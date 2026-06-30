# 06 · 推理 (Inference)

训练好的模型怎么"说话"？这一章拆解自回归生成的完整流程，并把 nano-LLM 的极简实现和工业级推理系统 (vLLM / SGLang / TensorRT-LLM / Mooncake) 的核心优化串起来。

## 1. 自回归生成的本质

LM 输出的是**下一个 token 的概率分布**。生成 N 个 token，就要做 N 次前向：

```
[BOS, 床,  前]                  → 明        (第 1 步)
[BOS, 床,  前, 明]              → 月        (第 2 步)
[BOS, 床,  前, 明, 月]          → 光        (第 3 步)
[BOS, 床,  前, 明, 月, 光]      → ，        (第 4 步)
...
```

朴素实现：每一步都把已生成的全部序列重新 forward 一遍，复杂度 **O(N²)** 次 token 计算。
这显然浪费——前面那些 token 的中间状态在上一步刚算过。

**KV cache** 就是把这些中间状态存下来，下一步直接用。

## 2. KV cache: 为什么只需要 cache K 和 V？

回顾 attention 的计算：

```
Q = X · W_Q    K = X · W_K    V = X · W_V
attn(Q, K, V) = softmax(Q · K^T / √d) · V
```

生成第 t+1 个 token 时，**Q 只需要最新位置的那一行**（因为我们只关心新位置看历史），
但 **K、V 需要全部历史位置**（因为新 Q 要跟所有历史 K 算 attention，然后加权所有历史 V）。

所以：
- **Q**: 只算新的一个 token，不需要 cache
- **K, V**: 把每一步算出来的 `[B, n_kv_heads, 1, head_dim]` 拼接进 cache，形状变成 `[B, n_kv_heads, T_total, head_dim]`

复杂度从 O(N²) 降到 **O(N)** 次 token 计算（每步仍是 O(T) 的 attention，但 token 算一次）。

### 在代码里的样子

`nanollm/model.py` 的 `Attention.forward()`:

```python
if kv_cache is not None:
    past_k, past_v = kv_cache
    k = torch.cat([past_k, k], dim=2)   # 沿序列维拼接
    v = torch.cat([past_v, v], dim=2)
new_cache = (k, v) if return_cache else None
```

简单粗暴——真实推理引擎用 PagedAttention 之类的内存管理（见第 7 节），但语义就是这个。

## 3. Prefill vs Decode: 两阶段计算模式

工业界把推理切成两个阶段，原因是它们的**计算特性完全不同**：

| 阶段 | 输入 | 输出 | 计算特性 | 瓶颈 |
|------|------|------|----------|------|
| **Prefill** | 全部 prompt (T_p 个 token) | 初始 KV cache + 第一个 token | 大矩阵乘法，T_p × d 维度 | **Compute-bound** (FLOPs) |
| **Decode** | 1 个新 token | 1 个 token + 更新 cache | 小矩阵 × 大 cache | **Memory-bound** (HBM 带宽) |

decode 阶段每步都是 `[B, 1, d]` 的小矩阵跟 cache 算 attention，**算术强度极低**（FLOPs/Byte 小），GPU 的 TFLOPs 用不上，瓶颈是把 KV cache 从 HBM 搬到 SRAM 的带宽。

这就是为什么：
- vLLM、SGLang 在 decode 阶段拼命做 **continuous batching**——多个请求的 decode 拼成大 batch，让 GPU 别闲着
- **Prefill / Decode 解耦**（Mooncake / DistServe / Splitwise）把两个阶段调度到不同的 GPU 集群，prefill 集群配高算力卡，decode 集群配大显存高带宽卡，各取所需

nano-LLM 的 `generate()` 里这两阶段写得很清楚：

```python
# Prefill: 一次性把整个 prompt 跑完，得到 cache
out = self.forward(input_ids, return_caches=True)   # [B, T_p, V]
kv_caches = out.kv_caches
next_logits = out.logits[:, -1, :]                  # 只要最后一位的 logits

for _ in range(max_new_tokens):
    next_token = sample(next_logits, ...)            # 采样
    # Decode: 只喂 1 个 token，复用 cache
    out = self.forward(next_token, kv_caches=kv_caches, return_caches=True)
    kv_caches = out.kv_caches
    next_logits = out.logits[:, -1, :]
```

## 4. KV cache 的显存占用

这是部署时绕不开的问题。单个 token 的 KV cache 大小：

$$\text{bytes/token} = 2 \times n_{layers} \times n_{kv\_heads} \times \text{head\_dim} \times \text{dtype\_bytes}$$

其中 `2` 是 K 和 V，`dtype_bytes` 通常是 2 (FP16/BF16) 或 1 (INT8/FP8)。

### nano-LLM 默认配置 (d_model=512, n_layers=8, n_kv_heads=2, head_dim=64, BF16)

$$2 \times 8 \times 2 \times 64 \times 2 = 4096 \text{ bytes/token} = 4 \text{ KB/token}$$

512 上下文 = 2 MB/序列。完全不是问题。

### 对比 Llama-3-70B (n_layers=80, n_kv_heads=8, head_dim=128, BF16)

$$2 \times 80 \times 8 \times 128 \times 2 = 327,680 \text{ bytes/token} = 320 \text{ KB/token}$$

- 8K context = 2.5 GB/序列
- 128K context = 40 GB/序列 ← **比模型权重还大**！

### 为什么 GQA 是部署的救命稻草

如果 Llama-3-70B 用 MHA (n_heads=64) 而不是 GQA (n_kv_heads=8)，KV cache 直接 ×8，
128K context 单序列 320 GB——单卡放不下，必须切。

GQA / MQA 几乎是部署导向的设计——训练时质量损失可控，推理时显存省 4~8 倍，吞吐量翻倍。
nano-LLM 也用了 GQA (n_heads=8, n_kv_heads=2)，可以在 `tests/test_model.py` 里验证 cache 形状只跟 `n_kv_heads` 有关。

## 5. 采样策略 (Sampling)

有了 logits 怎么挑下一个 token？这一步直接影响输出质量和"创造力"。

### 5.1 贪心 (Greedy)

```python
next_token = logits.argmax(dim=-1)
```

每次取最大概率。**确定性、可复现**，但容易掉进重复循环（"我爱你我爱你我爱你..."），尤其在小模型上。
**适合**：评测、需要可复现的场景。

### 5.2 温度 (Temperature)

```python
probs = softmax(logits / T)
next_token = multinomial(probs, 1)
```

- `T → 0`: 退化到贪心
- `T = 1`: 原始分布
- `T → ∞`: 趋近均匀分布（胡言乱语）

直觉：T 把分布"拉平"或"拉尖"。一般 0.7~1.0。

### 5.3 Top-k

```python
v, _ = topk(logits, k)
logits[logits < v[..., -1:]] = -inf
probs = softmax(logits / T)
```

只在 top-k 个候选里采样，截断长尾。k 一般 20~100。
**问题**：k 是固定的，分布很尖时 k=50 太大（会引入噪声），分布很平时 k=50 太小（太单调）。

### 5.4 Top-p (Nucleus sampling)

```python
sorted_p = cumsum(sorted softmax(logits))
# 保留累积概率 < p 的最小集合
```

动态选取累积概率达到 p 的候选集。p=0.9 表示"采样池占总概率质量的 90%"。
**好处**：分布尖时自动用小集合，分布平时自动扩大集合。

实践：top-k 和 top-p **可以叠加**，nano-LLM 的 `generate()` 实现了先 top-k 再 top-p。
社区常用配置：`temperature=0.7, top_p=0.9, top_k=50`。

### 5.5 其他

- **Repetition penalty**: 对已出现过的 token 降权，缓解循环
- **Min-p** (新): 比 top-p 更鲁棒，按最大 token 的某个比例做阈值
- **Beam search**: 同时维护 N 条候选路径，**不适合**开放式生成（缺乏多样性），但翻译、摘要还在用
- **Speculative decoding**: 小模型先猜 K 个 token，大模型一次性验证。验证通过则 K 个 token 一次产出 → 2-3× 加速

## 6. nano-LLM 推理实战

### 6.1 续写模式

```bash
python scripts/generate.py \
    --ckpt checkpoints/pretrain_final.pt \
    --tokenizer data/tokenizer.json \
    --prompt "床前明月光，" \
    --max_new_tokens 100 \
    --temperature 0.8 \
    --top_p 0.9
```

### 6.2 对话模式 (SFT 后)

```bash
python scripts/generate.py \
    --ckpt checkpoints/sft_final.pt \
    --tokenizer data/tokenizer.json \
    --chat
```

会进入交互式 REPL。代码里做的事:

```python
# 拼对话模板
ids = tokenizer.apply_chat_template([
    {"role": "user", "content": user_input},
])
# 模型生成
output_ids = model.generate(
    ids, max_new_tokens=256,
    temperature=0.8, top_p=0.9,
    eos_token_id=tokenizer.end_token_id,    # 注意是 <end>，不是 <eos>
)
# 解码新增部分
reply = tokenizer.decode(output_ids[0, len(ids):])
```

**关键点**: SFT 模式下 EOS 用 `<end>`（一轮回答结束符），不是 `<eos>`（整篇文本结束）。
否则模型一回答完就停，但实际上你可能想多轮对话——`<eos>` 是给 pretrain 用的边界标记。

## 7. 工业级推理引擎做了什么

nano-LLM 的 `generate()` 大概 50 行代码就能跑。真实部署要解决的问题：

### 7.1 PagedAttention (vLLM)

朴素 KV cache 在显存里是连续的，一个请求结束就留下碎片。
PagedAttention 把 cache 切成固定大小的 block（类似 OS 的虚拟内存分页），**用一张 block table 把逻辑序列映射到物理 block**：

- 显存利用率从 ~30% 提到 >90%
- 支持 prefix sharing：多个请求共享相同 prompt 前缀的 block (system prompt 复用)
- 支持 copy-on-write：beam search、parallel sampling 时分支

### 7.2 Continuous Batching

朴素 batching：等一个 batch 全部完成才能进下一个。慢序列拖快序列。
Continuous batching：每生成一步检查哪些序列结束了，立刻替换进新请求。GPU 利用率显著提升。
vLLM / SGLang / TGI 标配。

### 7.3 RadixAttention (SGLang)

把 KV cache 组织成 **Radix 树**（前缀树），自动识别和共享公共前缀。
适合 system prompt、few-shot、agent 场景里大量共享前缀的请求。

### 7.4 分层 KV cache (SGLang HiCache, Mooncake)

GPU HBM 太贵也太小。把 KV cache 做成 **GPU HBM → CPU DRAM → SSD/RDMA 远端** 的多级缓存：
- 热前缀放 GPU
- 冷前缀降到 CPU/SSD，需要时再调回
- 跨 GPU 节点共享 cache (Mooncake 用 RDMA)

直接把"长上下文/历史会话"从延迟和成本上变得可行。

### 7.5 Prefill / Decode 解耦 (Mooncake, DistServe)

前面提过——prefill 和 decode 是不同的负载特性，混在一张卡上互相干扰：
- prefill 长 → 阻塞 decode → TBT (Time Between Tokens) 抖动
- decode 多 → 浪费 prefill 集群的算力

**解法**：两类节点池物理分离，KV cache 通过高速网络（RDMA）从 prefill 节点搬到 decode 节点。
代价是搬运开销，收益是吞吐和 SLO 双提升。

### 7.6 量化推理 (Quantization)

把权重从 BF16 压到 INT8 / INT4 / FP8：
- 显存减半甚至 1/4
- 内存带宽瓶颈缓解，decode 阶段直接加速
- 质量损失通常 < 1% (PPL)

主流方案：
- **GPTQ / AWQ**: 权重量化到 INT4，激活保持 FP16
- **FP8** (H100 / B300 原生支持): 权重 + 激活都 FP8
- **KV cache 量化**: cache 单独压到 INT8 / FP8，长上下文场景收益巨大

### 7.7 投机解码 (Speculative Decoding)

- **Draft model + verify**: 小模型 (~1B) 先生成 K 个 token，大模型一次性 verify 全部 K 个 → 平均接受率 ~3
- **Medusa / EAGLE**: 在主模型上加几个轻量 head 直接预测多个未来 token
- **n-gram speculation**: 直接用 prompt 里的 n-gram 当 draft（适合代码、长文档续写）

## 8. 性能指标怎么看

部署 LLM 经常看的几个指标：

| 指标 | 含义 | 典型值 (Llama-70B / A100) |
|------|------|---------------------------|
| **TTFT** (Time To First Token) | 从请求到第一个 token | 100~500 ms (prefill 决定) |
| **TBT** (Time Between Tokens) | 相邻两个 token 间隔 | 20~50 ms (decode 决定) |
| **Throughput** | 全集群 tokens/s | 越高越省钱 |
| **吞吐 vs 延迟** | trade-off | batch 越大吞吐越高，延迟越差 |

**SLO 设计**: TTFT < 500ms, P99 TBT < 100ms 是常见的对话 SLA。

## 9. 接下来

最后一章看怎么把这个项目"玩坏"——各种实验、扩展方向、能学到什么。

→ `07-experiments.md`

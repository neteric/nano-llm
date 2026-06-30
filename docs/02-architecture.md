# 02 · 模型架构详解

本文逐组件讲清楚 `nanollm/model.py` 里的每一段代码。架构与 Llama-2/3、Qwen2、Mistral 同源。

## 0. 总览图

```
input_ids: [B, T]                      ← B=batch, T=seq_len
    │
    ▼
[Token Embedding]                      d_model 维向量
    │
    ▼
┌───────────────────────────┐
│  Block × N                │
│  ┌─────────────────────┐  │
│  │ x                   │  │
│  │ │                   │  │
│  │ ▼                   │  │
│  │ RMSNorm             │  │
│  │ │                   │  │
│  │ ▼                   │  │
│  │ Attention (GQA+RoPE)│  │
│  │ │                   │  │
│  │ +←──────────────────│  ← residual
│  │ │                   │  │
│  │ ▼                   │  │
│  │ RMSNorm             │  │
│  │ │                   │  │
│  │ ▼                   │  │
│  │ FFN (SwiGLU)        │  │
│  │ │                   │  │
│  │ +←──────────────────│  ← residual
│  └─────────────────────┘  │
└───────────────────────────┘
    │
    ▼
RMSNorm
    │
    ▼
LM Head (Linear, tied with token_emb)
    │
    ▼
logits: [B, T, vocab_size]
```

注意是 **Pre-Norm**: norm 在 attention/FFN **之前**，残差直通。
对比经典 Transformer 的 Post-Norm（norm 在残差之后），Pre-Norm 训练更稳，几乎是现代 LLM 的统一选择。

## 1. RMSNorm

```python
class RMSNorm(nn.Module):
    def forward(self, x):
        rms = x.pow(2).mean(-1, keepdim=True).add(eps).rsqrt()
        return x * rms * self.weight
```

**对比 LayerNorm**:

| 操作          | LayerNorm                              | RMSNorm                       |
|---------------|----------------------------------------|-------------------------------|
| 中心化        | 减去 mean                              | ❌ 不做                       |
| 归一化分母    | sqrt(var + eps)                        | sqrt(mean(x²) + eps)          |
| 可学习参数    | gamma, beta                            | 只有 gamma                    |
| 计算量        | 1×                                     | 约 0.5×                       |

经验上 RMSNorm 效果几乎与 LayerNorm 一致，但**节省一组参数**且**计算更快**。
代码里强制用 fp32 算 RMS，是为了在 bf16/fp16 训练时避免数值下溢。

## 2. RoPE: 旋转位置编码

**为什么不用 Learned PE / 正弦 PE？**

- Learned PE 不能外推到更长序列
- 正弦 PE 是加性的，与内容耦合
- **RoPE 把位置信息编码进 query/key 的旋转角度**，使得 `attention(q_m, k_n)` 自然只依赖相对位置 `(m-n)`

**核心思想**: 把 head_dim 维向量两两配对成复数，乘以 `e^{iθ_m}` 实现旋转。
不同维度对应不同频率 `θ_i = 1/base^(2i/d)`，base 通常取 10000（长上下文模型会调到 1e6 或更高，扩展上下文长度）。

```python
# 关键代码 (简化)
freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2) / head_dim))
pos = torch.arange(max_seq_len).float()
freqs = torch.outer(pos, freqs)        # [seq_len, head_dim/2]
cos, sin = freqs.cos(), freqs.sin()

# 应用到 q / k
x_even, x_odd = x[..., 0::2], x[..., 1::2]
out_even = x_even * cos - x_odd * sin
out_odd  = x_even * sin + x_odd * cos
```

**关键性质**:
- **正交变换**: 不改变向量的 L2 范数（测试 `test_rope_shapes` 验证）
- **相对位置感知**: 旋转 m 和 n 后内积 = 只与 m-n 相关的函数
- **可外推**: 训练时见过 max_seq_len=2048，推理时仍可以用更长的位置（虽有精度损失，可用 YaRN / NTK 等技术改善）

注意 RoPE 只作用在 **q 和 k** 上，**不作用在 v** 上 —— 旋转的目的是让点积感知相对位置，v 是被加权的内容，没必要旋转。

## 3. Attention 与 GQA

### 3.1 标准多头注意力 (MHA)

```
q, k, v 各有 n_heads 个头，head_dim = d_model / n_heads
attention(q,k,v) = softmax(q · k^T / sqrt(d)) · v
```

**Causal mask**: 训练时 q 看不到未来 k，用下三角 mask 屏蔽。

### 3.2 GQA: Grouped Query Attention

```
n_heads     = 8     ← query 头数
n_kv_heads  = 2     ← K/V 头数 (8/2=4 个 q 头共享一组 kv)
```

每 `n_heads / n_kv_heads = 4` 个 q 头共享一组 (k, v)。极端情况:

- `n_kv_heads = n_heads`: 标准 MHA
- `n_kv_heads = 1`:        MQA (Multi-Query Attention)
- 中间值:                   GQA

**为什么用 GQA**: 推理时 KV cache 是显存大头，把 kv 头数从 32 降到 8 能节省 4× 显存。
代价是模型表达力略降，但实测下降很小。Llama-2 70B、Qwen2、Mistral、Llama-3 都用 GQA。

### 3.3 实现要点

```python
q = self.wq(x).view(B, T, n_heads, head_dim).transpose(1, 2)
k = self.wk(x).view(B, T, n_kv_heads, head_dim).transpose(1, 2)
v = self.wv(x).view(B, T, n_kv_heads, head_dim).transpose(1, 2)

# RoPE 加到 q, k
q, k = apply_rope(q, ...), apply_rope(k, ...)

# 拼接 KV cache (推理增量解码时用)
if kv_cache is not None:
    k = torch.cat([past_k, k], dim=2)
    v = torch.cat([past_v, v], dim=2)

# 复制 kv 头以匹配 q 头
k = k.repeat_interleave(n_rep, dim=1)
v = v.repeat_interleave(n_rep, dim=1)

# 用 PyTorch 内置 SDPA: 会自动选 Flash-Attn / Memory-Efficient / Math kernel
out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

**`scaled_dot_product_attention`** 是 PyTorch 2.0+ 提供的融合 kernel，是写出 production-grade attention 最简单的方式。
内部会自动选用 Flash-Attention 2 / xFormers Memory-Efficient / 朴素实现 之一。

## 4. SwiGLU FFN

```python
class SwiGLU(nn.Module):
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
```

**对比经典 FFN**:

| FFN 类型 | 公式                                | 矩阵数 |
|---------|-------------------------------------|--------|
| GELU FFN | `w2(GELU(w1(x)))`                  | 2      |
| SwiGLU   | `w2(Swish(w1(x)) ⊙ w3(x))`         | 3      |

**Swish (= SiLU)**: `silu(x) = x * sigmoid(x)`

SwiGLU 多了一路 "gate" (w3)，让某些维度可以被门控关闭/放大，表达力更强。
论文 [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) 显示：在等参数量下 SwiGLU 显著优于 GELU FFN。

为了等参数量，d_ff 通常按 **2/3 缩放**：

```
经典 GELU FFN:  d_ff = 4 * d_model       (2 个 d_model × d_ff 矩阵)
SwiGLU:         d_ff ≈ 2.67 * d_model    (3 个 d_model × d_ff 矩阵, 等参数)
```

实践中常取 `d_ff = round_to_multiple_of(2.75 * d_model, 64)`，本项目默认 `d_model=512, d_ff=1408`。

## 5. 权重绑定 (Weight Tying)

```python
if cfg.tie_word_embeddings:
    self.lm_head.weight = self.tok_emb.weight
```

token embedding 和 lm_head 共享同一个权重矩阵：
- 直觉上: "把 token 映射成向量" 和 "把向量映射回 token logits" 是互逆操作
- 实际效果: 省下一个 `vocab_size × d_model` 的矩阵，对小模型显著（26M 模型中省了约 3M 参数）

Llama 系列默认 **不** tie embeddings（模型够大，参数不是瓶颈），但小模型如 GPT-2 small、Qwen 小模型都用 tying。

## 6. 初始化

```python
nn.init.normal_(linear.weight, std=0.02)
# 残差路径上的输出投影 (wo, w2) 用更小的 std:
nn.init.normal_(p, std=0.02 / math.sqrt(2 * n_layers))
```

最后那个缩放是 GPT-2 的经验：残差路径上每加一层，方差会增大，把输出投影的初始化方差按 `1/sqrt(2N)` 缩小可以保持训练初期的稳定。

## 7. 模型参数量怎么估？

对于本项目结构 (d=d_model, V=vocab_size, L=n_layers, F=d_ff, H_kv=n_kv_heads, head_dim=h):

```
Embedding (tied):   V × d
Attention per layer:
    wq:  d × (n_heads × h) = d × d
    wk:  d × (n_kv_heads × h) = d × (n_kv × h)
    wv:  d × (n_kv × h)
    wo:  d × d
    总计 ≈ 2d² + 2d·n_kv·h
FFN per layer (SwiGLU):
    w1, w3: d × F
    w2:     F × d
    总计 = 3 × d × F
RMSNorm: 微不足道 (~2d/layer)

总参数 ≈ V·d + L · (2d² + 2d·n_kv·h + 3dF)
```

代入默认值 `d=512, L=8, n_kv=2, h=64, F=1408, V=6400`:

```
embedding:    6400 × 512 = 3.3M
per layer:    2·512² + 2·512·2·64 + 3·512·1408
            = 524K + 131K + 2.16M = 2.82M
8 layers:     22.5M
norms:        ~10K
总计:         ≈ 25.8M  ← 与 tests/test_model.py 输出吻合
```

## 8. 与真实 LLM 的差距

本项目省略 / 简化的部分:

| 真实 LLM                            | 本项目                     |
|------------------------------------|---------------------------|
| Flash-Attention 2 显式集成          | 用 SDPA 自动选择          |
| Mixture-of-Experts (DeepSeek/Mixtral)| 纯稠密 FFN              |
| Sliding window / global attn 混合   | 全局 causal               |
| 注意力的 sink token / NTK 扩展      | 单纯 RoPE                 |
| 多查询并行 (Multi-Token Prediction) | 单 token 预测             |
| Tensor / Pipeline / Sequence 并行   | 单卡                      |
| ZeRO / FSDP                        | 单 device                 |

但**基础结构 (RMSNorm + RoPE + SwiGLU + GQA) 完全一致**，理解了本项目就能读懂 Llama / Qwen 的源码。

## 接下来

→ `03-tokenizer.md` 看 BPE 怎么训

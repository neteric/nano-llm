# 04 · 预训练详解

预训练是把模型从"随机初始化"变成"懂语言"的关键阶段。本文讲清楚目标、数据、loss、优化器和调度。

## 1. 训练目标: 自回归语言建模

**给定 token 序列** `x = [x_1, x_2, ..., x_n]`，模型最小化：

$$ \mathcal{L} = -\frac{1}{n-1} \sum_{i=1}^{n-1} \log P(x_{i+1} | x_1, ..., x_i; \theta) $$

实现上是一个**向左 shift 的交叉熵**:

```python
input_ids = tokens[:-1]   # [t_1, t_2, ..., t_{n-1}]
targets   = tokens[1:]    # [t_2, t_3, ..., t_n]
logits    = model(input_ids)            # [B, T, V]
loss      = F.cross_entropy(logits.view(-1, V), targets.view(-1))
```

注意三件事：

1. **不显式构造 (input, target) 对**，而是把同一段 tokens 错位一位 —— 这样一次 forward 就能在所有 T 个位置同时计算 loss。
2. **causal mask 保证**预测 `t_{i+1}` 时只看 `t_1..t_i`，否则会"作弊"。
3. **不需要 attention mask** —— 我们把语料拼成一条流，文档边界不做特殊处理（论文里有"document mask"做精细化处理，但效果差异通常很小）。

## 2. 数据流水线

### 2.1 离线 tokenize

```python
# scripts/prepare_data.py
for line in corpus_file:
    ids = tokenizer.encode(line, add_eos=True)
    bin_file.write(np.array(ids, dtype=np.uint16).tobytes())
```

把整个语料 tokenize 成一个 **uint16 二进制流**。词表 < 65536 时用 uint16，省一半磁盘。
单个文件可以是几十 GB，用 `np.memmap` 按需读取。

### 2.2 训练时随机切片

```python
class PretrainDataset(Dataset):
    def __getitem__(self, _idx):
        i = random.randint(0, len(self.data) - seq_len - 1)
        chunk = self.data[i : i + seq_len + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y
```

**为什么随机起点？**
- 训出来的模型不应依赖"位置 0 总是文档开头"这种偏置
- 不放回随机采样在大语料上等价于按顺序遍历，但更鲁棒

**为什么用 seq_len+1 而不是 seq_len？**
- 需要 `seq_len` 个 (input, target) 对，必须切 `seq_len + 1` 个 token

## 3. 优化器: AdamW

```python
optimizer = torch.optim.AdamW(
    param_groups, lr=lr, betas=(0.9, 0.95), weight_decay=0.1,
)
```

- **betas=(0.9, 0.95)**: beta2 比图像/通用 NLP 任务的默认 0.999 小，是 LLM 训练的经验值（GPT-3、Llama 均用）。
  beta2 小 → 二阶矩衰减快 → 对学习率变化更敏感，配合 warmup 更稳。
- **weight_decay=0.1**: 只作用在 2D 参数（Linear weight、Embedding），不作用在 norm 的 gamma 和 bias。
- **fused=True** (GPU only): 把优化器步骤融合成一个 CUDA kernel，提速 ~10%。

## 4. 学习率调度: warmup + cosine

```python
def cosine_lr(step):
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps   # 线性升
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(pi * progress))
```

**为什么 warmup？**
- 训练初期梯度估计不稳，直接用大 lr 容易爆炸
- 典型 warmup 步数: 总步数的 0.5% ~ 2%

**为什么 cosine 衰减？**
- 实践经验：cosine 比 linear / step decay 略好
- 注意 cosine 的 **min_lr 不为 0**（Llama-2 是 max_lr 的 10%）

## 5. 梯度裁剪

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**为什么**: 训练初期 loss 高，可能产生大梯度；clip 防止参数被一脚踹飞导致训练崩溃。
**典型值**: 1.0；调小到 0.5 会让训练更稳但收敛慢一点。

## 6. 混合精度训练

```python
with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
    out = model(x, targets=y)
    loss = out.loss
loss.backward()
```

- **bf16**: 8-bit 指数 + 7-bit 尾数。指数范围与 fp32 一致，**不会数值溢出**。Ampere (A100/H100) 及以后强烈推荐。
- **fp16**: 5-bit 指数 + 10-bit 尾数。范围窄，需要 `GradScaler` 防梯度下溢。仅在老 GPU (V100, T4) 上用。
- **fp32**: 慢但稳，调试时用。

bf16 + AdamW 时，**weight、grad、激活全是 bf16**，但**优化器状态是 fp32** —— 默认行为，不需要额外配置。

## 7. 梯度累积

```python
optimizer.zero_grad()
for _ in range(grad_accum_steps):
    x, y = next(loader)
    with autocast:
        loss = model(x, y).loss / grad_accum_steps
    loss.backward()
optimizer.step()
```

**用途**: 显存装不下大 batch 时，用多个 micro-batch 累积梯度模拟大 batch。
**等效**: `effective_batch = batch_size × grad_accum_steps`

**经验**: LLM 训练通常 effective_batch_size = 0.5M ~ 4M token 量级（如 1024 × 2048）。本项目玩具规模用 8 × 64 = 512 token / step 就够。

## 8. Loss 与 Perplexity

```python
ppl = exp(loss)
```

困惑度 = 模型在每个位置"平均不确定多少 token"。
- ppl = 1: 模型每次都准确预测下一个 token
- ppl = V: 完全随机
- 典型 baseline:
    - WikiText-103: GPT-2 small ppl ≈ 29, Llama-2 7B ≈ 7
    - 你的小模型在简单语料上预期 ppl 100 ~ 50（不要期望 < 10）

如果 loss 长时间不降:
- LR 太大 → loss 震荡 / NaN
- LR 太小 → loss 缓慢下降
- 数据太少 → 早期下降然后过拟合

## 9. 训练步数估算

经验公式（Chinchilla）:

```
推荐训练步数 = (20 × 参数量) / (batch_size × seq_len × grad_accum)
```

例：26M 模型，effective batch = 32 × 512 = 16384 token

```
20 × 26M / 16384 ≈ 32000 步
```

本项目默认 `max_steps=5000` 是远不够"训饱"的，仅供调通流程。

## 10. checkpoint 怎么存？

```python
ckpt = {
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "step": step,
    "model_config": cfg.__dict__,
}
torch.save(ckpt, path)
```

存 optimizer 是为了**断点续训**。如果只是为了推理，可以只存 model 部分。
生产场景还要存:
- `rng_state` (random / torch / cuda) —— 严格可复现
- `lr_scheduler.state_dict()` —— 如果用了 scheduler 对象
- `iteration` 而不是 `step` —— 区分 epoch / iteration

## 11. 多卡训练 (本项目未实现但务必了解)

| 并行方式             | 适合规模          | 工具                  |
|---------------------|------------------|----------------------|
| DDP (Data Parallel) | < 1B 参数          | `torch.nn.parallel.DistributedDataParallel` |
| ZeRO-1/2/3 / FSDP   | 1B - 100B+        | `torch.distributed.fsdp` / DeepSpeed |
| Tensor Parallel     | 单层放不下时       | Megatron / vLLM internal |
| Pipeline Parallel   | 极大模型           | DeepSpeed Pipeline / PiPPy |
| Sequence Parallel   | 超长上下文         | DeepSpeed / Megatron-LM |

对 26M 模型用 DDP 就够；上到 7B 必须 ZeRO-3 或 FSDP；上到 70B+ 通常 ZeRO + Tensor Parallel。

## 12. 接下来

→ `05-sft.md` 把 base model 变成 chat model

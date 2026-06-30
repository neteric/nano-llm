# 07 · 实验与扩展

代码跑通只是开始。这一章给一组**可上手的实验**，每个都能让你对 LLM 的某个具体方面有更深的体感；同时给一份从 nano-LLM 出发的扩展路线图。

## 1. 推荐实验清单

下面的实验按"投入 / 收益"由低到高排，挑感兴趣的做。

### 实验 1：换数据集

最低投入、最高回报的实验。当前的合成中文小故事词汇极有限，模型很快就 overfit。换上真实数据，loss 曲线、生成质量、收敛 token 数会立刻发生变化。

**英文小数据**：[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) — 470M tokens 的简单英文故事，专为 nano 级模型设计，10M 参数就能写出连贯故事。

**中文数据**：
- [Wikipedia-zh](https://dumps.wikimedia.org/zhwiki/) — 干净、高质量
- [SkyPile-150B](https://huggingface.co/datasets/Skywork/SkyPile-150B) — 大规模中文预训练语料
- [WuDaoCorpus](https://www.scidb.cn/en/detail?dataSetId=c6a3fe684227415a9db8e21bac4a15ab) — 200GB 中文

**实验问题**：
- 同样 token 量，TinyStories vs 合成数据的最终 PPL 差多少？
- 词表多大才合适？2K vs 6K vs 32K，在 100M token 数据上分别什么效果？

### 实验 2：调缩放 (Scaling)

把 `ModelConfig` 改一改，量一量参数量和 loss 的关系。

| 配置 | d_model | n_layers | 参数量 | 备注 |
|------|---------|----------|--------|------|
| tiny | 128 | 4 | ~1 M | CPU 能跑 |
| small (默认) | 512 | 8 | ~26 M | 单卡 GPU 几小时 |
| medium | 768 | 12 | ~110 M | GPT-2 small 规模 |
| large | 1024 | 24 | ~350 M | GPT-2 medium 规模 |

固定数据，画 **loss vs 参数量** 的曲线。理论上应该呈幂律下降（Chinchilla scaling laws）：

$$L(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}$$

Chinchilla 给出的最优 token : 参数比例约 20:1。你的小数据能验证这点吗？

### 实验 3：消融 GQA

GQA 到底省了多少、损失多少质量？

把 `n_kv_heads` 分别改成：
- `n_kv_heads = n_heads` → MHA (Multi-Head Attention，传统做法)
- `n_kv_heads = 2` → GQA (默认)
- `n_kv_heads = 1` → MQA (Multi-Query Attention)

测三件事：
1. **最终训练 loss / PPL** —— 质量损失
2. **生成时 KV cache 占用** —— 内存收益（直接看 cache 的 shape）
3. **推理吞吐** —— 跑 `generate()`，测 tokens/s

工业经验：MHA → GQA(8/64) 质量基本无损，KV cache 减少 8 倍；GQA → MQA 质量会有可见下降。

### 实验 4：Tokenizer 影响

训三个 tokenizer：
- `vocab_size=2000`（极小）
- `vocab_size=6400`（默认）
- `vocab_size=32000`（Llama 量级）

同一份数据，量：
- **压缩率**: tokens/字符（中文越接近 1.0 越好，理想约 0.6~0.7）
- **OOV / `<unk>` 率**: 应该接近 0（ByteLevel BPE 保证）
- **训练步速**: 大词表 embedding 矩阵 + 输出层都变大
- **生成质量**: 词表太小，每个 token 信息量少，模型要预测更长序列

### 实验 5：学习率与 warmup

调三个旋钮，看 loss 曲线：

- `lr` ∈ {1e-4, 3e-4, 1e-3, 3e-3} — peak LR
- `warmup_iters / total_iters` ∈ {1%, 5%, 10%}
- 是否做 cosine decay 到 0 vs decay 到 0.1 × peak

观察：
- LR 太大：loss 发散或剧烈震荡
- LR 太小：收敛慢，最终 loss 偏高
- 无 warmup：开头几步 loss 突刺，可能直接 NaN（尤其大模型）

### 实验 6：DPO 偏好对齐

SFT 之后做 DPO。**最实用的对齐实验**。

构造一份偏好数据 `(prompt, chosen, rejected)`，例如：
- `chosen`：礼貌、完整的回答
- `rejected`：粗鲁、敷衍、不安全的回答

用 [trl](https://github.com/huggingface/trl) 的 `DPOTrainer` 或自己写 ~100 行实现：

```python
import torch.nn.functional as F

def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps, beta=0.1):
    pi_logratio = policy_chosen_logps - policy_rejected_logps
    ref_logratio = ref_chosen_logps - ref_rejected_logps
    return -F.logsigmoid(beta * (pi_logratio - ref_logratio)).mean()
```

需要两份 nano-LLM 同时在显存：policy (训练的) + reference (冻结 SFT 模型)。
观察：DPO 训完，模型选择 chosen 风格的概率应该显著上升，PPL 在通用数据上略升。

### 实验 7：长上下文 / RoPE 外推

默认 `max_seq_len=512`。能不能让模型支持 2048+？

**方法 A — 训练时扩展**：直接把 `max_seq_len` 改 2048，从头训。代价是 attention O(T²)。

**方法 B — RoPE 频率插值**（不重新训）：
推理时把 `position_ids * (512 / 2048)`，等于"压缩"位置。短上下文质量略降，但可以处理长上下文。
这是 LongRoPE / Position Interpolation 的核心想法。

**方法 C — YaRN / NTK-aware scaling**：对 RoPE 的不同频段分别处理，比 PI 更平滑。

实验：选一种方法实现，比较模型在 1024 / 2048 长度的 PPL 退化程度。

## 2. 把 nano-LLM 升级到 100M+ 规模

实际工程关心从 toy 到 production 之间的 gap 是什么。下面是把这套代码扩展到能跑 GPT-2 medium / Llama-tiny 规模需要补的东西。

### 2.1 训练侧

| 当前 | 100M+ 规模需要 |
|------|----------------|
| 单 GPU | **DDP** (`torchrun --nproc_per_node`) |
| FP32 / 简单 AMP | **BF16 全程**，配合 grad scaler |
| AdamW 标准 | **Fused AdamW** (`apex` 或 PyTorch 2.x 自带) |
| 默认 attention | **FlashAttention-2** (`flash-attn` 包) 显存减半，速度翻倍 |
| 全量参数训 | 大于 1B 时考虑 **FSDP** 或 **DeepSpeed ZeRO-3** 分片优化器/梯度/参数 |
| 朴素 dataloader | **Streaming dataset** (mosaicml-streaming, webdataset)，避免一次性加载 |
| 单机 | **Slurm + torchrun**，多机多卡 |

### 2.2 数据侧

| 当前 | production |
|------|------------|
| `txt` 文件读全部 | 数据分 shard，每 shard 几 GB |
| 同步预处理 | **离线** tokenize 成 `.bin` 后 mmap |
| 单种数据 | **数据混合**：web / books / code / math / 多语言按比例混 |
| 无质量过滤 | **dedup** + **质量分类器** (启发式 + classifier) + **毒性过滤** |
| 训练顺序固定 | **课程学习** (curriculum): 简单 → 复杂 |

### 2.3 评测

只看 train loss 完全不够。最少补：
- **PPL on held-out** (你自己分一个 val set)
- **下游任务**: lm-evaluation-harness 跑 ARC / HellaSwag / MMLU / C-Eval 等
- **生成质量人工评测**: 一组 prompt 让人对比 baseline vs new

### 2.4 监控

- **wandb / tensorboard**: loss / lr / grad norm / param norm 全要记
- **grad norm 爆炸**: 早期诊断训练不稳的最重要信号
- **吞吐**: tokens/sec/GPU，对比理论 MFU (Model FLOPs Utilization)，A100 BF16 跑到 50%+ 算优秀

## 3. 模型压缩与部署

### 3.1 量化

`bitsandbytes` 或 `auto-gptq` 可以一键把模型量化到 INT8 / INT4：

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")
model = AutoModelForCausalLM.from_pretrained(..., quantization_config=bnb_config)
```

不过 nano-LLM 不是 HF 格式。要量化得：
1. 写一个转换脚本把 `nanollm.NanoLLM` 转成 HF `LlamaForCausalLM`（架构本来就对得上）
2. 用 HF 生态的量化工具

### 3.2 推理引擎接入

把 nano-LLM 训出来的权重转成 HF Llama 格式后，可以直接喂给：
- **vLLM**: `vllm.LLM("path/to/converted")` → 立即拿到 PagedAttention + continuous batching
- **SGLang**: 同上，自动 RadixAttention
- **llama.cpp**: 转 GGUF 格式后可以 CPU 推理，甚至手机推理

转换脚本的核心是把每层的 weight 名字对应起来。Llama 和 nano-LLM 的命名几乎一致（这是有意设计的），主要差别：
- nano-LLM 用 `wq/wk/wv/wo`，Llama 用 `q_proj/k_proj/v_proj/o_proj`
- nano-LLM 用 `w1/w2/w3`，Llama 用 `gate_proj/down_proj/up_proj`

### 3.3 知识蒸馏

让 nano-LLM 从大模型学：
1. 用大模型 (Qwen-7B) 生成大量 (prompt, response) 对
2. 用这些数据 SFT 你的小模型
3. 进阶：在 token level 做 **logit distillation** — 让 student 的输出分布拟合 teacher 的 softmax

社区做法：TinyLlama / MobileLLM / Phi 系列，都从大模型蒸馏。

## 4. 学习路径建议

如果你刚学完 nano-LLM 想再往深处走：

**论文必读 (按推荐顺序)**
1. *Attention Is All You Need* — Transformer 起点
2. *GPT-3 / Scaling Laws* — 为什么要堆参数
3. *Chinchilla* — 参数和数据怎么配
4. *LLaMA / LLaMA 2* — 现代 LM 工程实践
5. *FlashAttention* (1 和 2) — attention 工程化的关键
6. *RoFormer (RoPE)* — 位置编码
7. *GQA* — 推理优化
8. *vLLM (PagedAttention)* — 推理系统
9. *DPO* — 对齐
10. *Mooncake* — 大规模 LLM serving 系统设计

**代码必读**
- [nanoGPT](https://github.com/karpathy/nanoGPT) — Karpathy 的极简 GPT，本项目主要灵感来源
- [minimind](https://github.com/jingyaogong/minimind) — 中文社区类似项目，规模更完整
- [llama.c](https://github.com/karpathy/llama2.c) — Llama 推理 600 行 C
- [tinygrad](https://github.com/tinygrad/tinygrad) — 自己写一个深度学习框架
- [vLLM](https://github.com/vllm-project/vllm) — production 推理引擎源码
- [SGLang](https://github.com/sgl-project/sglang) — RadixAttention / HiCache 实现

**动手实验**
- 把 nano-LLM 训到能写出像样的中文故事 (TinyStories-zh 风格)
- 实现 DPO 对齐一个有"个性"的助手
- 把训出来的模型转 HF 格式，用 vLLM 上线一个本地 API
- 量化到 INT4 在 CPU 上跑

## 5. 这套代码可以怎么改

最后一些"魔改"方向，对应不同的研究/工程兴趣：

- **MoE 化**: 把 SwiGLU FFN 换成 8 个 expert + top-2 routing，参数量翻 7 倍但激活量不变。研究 routing 行为、负载均衡 loss
- **Mamba / RWKV / RetNet**: 把 Attention 替换成线性复杂度的 SSM，理论上无限长上下文
- **Mixture-of-Depths**: 让每个 token 决定自己要不要进某些层
- **Multi-token prediction**: 一次预测下 4 个 token (DeepSeek-V3 做的)，加速训练和推理
- **Diffusion LM**: 用扩散模型代替自回归，并行解码（学术前沿）

## 6. 结语

LLM 不是黑箱，**每一行代码都对应一个具体可推导的数学操作**。
nano-LLM 大约 800 行 Python 复现了一个 production-grade LLM 的全部核心组件——只是参数规模和工程优化不同。

读完这套文档 + 跑通这套代码，你应该能：

- 看懂 Llama / Qwen / GPT 系列模型的源码
- 理解 vLLM / SGLang / Mooncake 这类推理系统优化的根源
- 复现一篇 LLM 论文的 baseline
- 把你自己的领域数据训一个专用小模型

接下来：开始动手训自己的模型。

---

← 上一章：`06-inference.md`  ·  回到 [README](../README.md)

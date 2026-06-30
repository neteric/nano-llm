# 01 · 全流程概览

理解大模型，先建立一张**全局地图**。本文件回答两个问题：

1. 从一堆原始文本到一个能聊天的模型，中间发生了什么？
2. 这些阶段为什么是这个顺序，每一步在解决什么问题？

## 1. 全流程总览

```
[原始文本语料]
      │
      │  ① 训练分词器 (BPE)
      ▼
[tokenizer.json]                ←── 词表 + 合并规则
      │
      │  ② 用分词器把语料编码成 token 流
      ▼
[pretrain.bin]                  ←── uint16/uint32 二进制
      │
      │  ③ 预训练: 自回归语言建模
      │      目标: 给定前 i 个 token，预测第 i+1 个 token
      ▼
[base model]                    ←── 一个"懂语言"但不"懂指令"的模型
      │
      │  ④ 监督微调 (SFT): 对话数据上继续训
      │      目标: 学会对话格式 + 学会"按用户要求回答"
      ▼
[chat model]
      │
      │  ⑤ (可选) RLHF / DPO: 用人类偏好对齐
      │      目标: 让模型回答更符合人类口味、更安全
      ▼
[aligned model]
      │
      │  ⑥ 推理: KV cache + 采样
      ▼
[生成的文本]
```

本项目实现了 ① ② ③ ④ ⑥，⑤ (对齐) 在 `docs/07-experiments.md` 有进阶建议。

## 2. 每一步在解决什么核心问题

### ① 分词器: 把字符转成模型能吃的 ID

**问题**: 神经网络只能处理数字。但一个汉字 / 单词太"粗"（OOV 严重），一个字节又太"细"（序列变得很长）。

**解法**: BPE（Byte Pair Encoding）—— 从单字节开始，迭代合并最高频的相邻对，得到一个"中粒度"的词表。
常见词如 "的"、"and " 会变成单个 token；罕见词 / 中文罕见字 会拆成多个 token。

**输出**: 一个 vocab_size（典型 32k ~ 200k）的词表 + 合并规则文件。

### ② 数据准备: 文本流化

**问题**: 训练时要从一个超长的 token 序列里随机切片。如果每次都 tokenize，CPU 会成瓶颈。

**解法**: 离线一次性 tokenize 完整个语料，存为二进制（每个 token 用 uint16/uint32 表示）。
训练时用 `numpy.memmap` 按需读取，几乎零内存开销。

### ③ 预训练: 学语言本身

**问题**: 模型一开始什么都不会。怎么让它学到语言的统计规律 + 世界知识？

**解法**: **自回归语言建模 (Causal LM)**。给定一段文本 `[t1, t2, ..., tn]`：
- 输入: `[t1, t2, ..., t_{n-1}]`
- 目标: `[t2, t3, ..., tn]`（向左 shift 一位）
- 损失: 在每个位置上的交叉熵

由于注意力的 **causal mask**，预测 `t_{i+1}` 时模型只能看到 `t1..t_i`，这模拟了自回归生成的过程。
所有 token 同时计算 loss（"并行化的"自回归），这是 Transformer 比 RNN 快得多的关键。

**结果**: 一个 base model，它会根据前文续写后文，但不会按指令回答问题。
你给它 "请用一句话总结相对论："，它可能会续写 "我们在课堂上学过……"。

### ④ SFT: 学指令跟随

**问题**: base model 不会对话。我们需要它在看到 "user: 问题" 后输出 "assistant: 答案"。

**解法**: 在**对话格式数据**上继续训练，关键有三：
- **聊天模板**: 用特殊 token 包装每个 role 的内容，例如
  `<bos><user>你好<end><assistant>你也好<end>`
- **loss masking**: **只在 assistant 回答的 token 上算 loss**。
  user 输入和特殊 token 不算（让模型不要学着复述用户输入）。
- **较小的学习率**: SFT 学习率通常比预训练小 1-2 个量级（典型 5e-5）。
  目的是不破坏预训练学到的"语言能力"，只调整输出格式。

**结果**: 一个 chat model，能在对话格式下输出格式正确的回答。

### ⑤ 对齐 (本项目未实现)

**问题**: SFT 后的模型回答常常啰嗦、错误、不安全。如何用人类偏好进一步打磨？

**解法（两类）**:
- **RLHF (PPO)**: 训练一个 reward model 给回答打分，用 PPO 让 LM 最大化得分。复杂，难调。
- **DPO (Direct Preference Optimization)**: 直接用偏好对 `(chosen, rejected)` 优化，无需 reward model。
  目前事实上的工业标准之一。

### ⑥ 推理: 高效自回归生成

**问题**: 生成 N 个 token 时，每生成一个 token 都要把前面所有 token 重新跑一遍 attention，复杂度 O(N²)。

**解法**: **KV cache** —— 把每层的 K, V 缓存下来。
- **Prefill 阶段**: 一次性 forward 整个 prompt，得到完整 KV cache。
- **Decode 阶段**: 每次只 forward 1 个新 token，把它的 K, V append 到 cache 里。
  attention 的计算量从 O(N²) 降到 O(N)。

代价是显存：一个 token 的 KV 大小 = `2 * n_layers * n_kv_heads * head_dim * dtype_bytes`。
这也是 **GQA** 把 `n_kv_heads` 调小的核心动机 —— Llama-2 70B 用 GQA 把 KV cache 减少了 8 倍。

## 3. 参数规模与数据规模的关系

**Chinchilla scaling law** 的经验结论:

> 训练 token 数 ≈ 20 × 模型参数量

| 模型参数 | 推荐训练 token |
|---------|-------------|
| 1M      | 20M tokens  |
| 26M     | 520M tokens |
| 110M    | 2.2B tokens |
| 7B      | 140B tokens |

低于这个比例 → 模型"没吃饱"，能力不足。
高于这个比例 → 收益递减，但近年来（如 Llama-3）发现继续训仍有提升，业界做法已经远超 Chinchilla。

对于本项目的玩具规模，可以放心地"过训"（比如 26M 模型训 1B token），效果会更好。

## 4. 显存与算力的粗略估算

**显存** (训练，bf16 + AdamW + 激活检查点关闭):

```
显存 ≈ 参数量 × (2 byte 模型 + 2 byte 梯度 + 8 byte AdamW 状态) + 激活
     ≈ 参数量 × 12 byte + 激活
```

26M 模型 ≈ 312 MB 状态 + ~几百 MB 激活，单卡 8G 都够。
7B 模型 ≈ 84 GB 状态，必须 ZeRO / FSDP 切片。

**算力** (FLOPs):

```
训练 FLOPs ≈ 6 × 参数量 × 训练 token 数
```

26M × 100M token = 1.56e16 FLOPs ≈ 单卡 V100 (~15 TFLOPS bf16) 跑约 17 分钟。

## 5. 接下来读哪个？

- 想看模型怎么搭 → `02-architecture.md`
- 想搞懂分词器 → `03-tokenizer.md`
- 想动手训 → `04-pretrain.md` 起读
- 想了解推理优化 → `06-inference.md`

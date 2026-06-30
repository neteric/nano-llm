# 05 · 监督微调 (SFT) 详解

预训练让模型"懂语言"，SFT 让模型"会聊天"。本文讲清楚 SFT 数据格式、loss masking 和实践细节。

## 1. 为什么 SFT 不能用预训练数据继续训？

预训练数据是无结构文本，模型学到的是"续写"。

但用户的交互模式是：

```
user: 介绍一下相对论
assistant: 相对论是...
```

模型需要学会:
- **识别对话格式**: `<user>...<end><assistant>...<end>` 的结构
- **在正确位置停止**: 不要写完答案后继续编 user 的下一句
- **跟随指令**: 看到问题给答案，看到要求做总结，等等

这些是"格式 + 行为"的学习，必须用结构化的对话数据。

## 2. 对话模板 (Chat Template)

本项目用一个简化的 ChatML 风格模板：

```
<bos><user>你好<end><assistant>你好！<end><user>再见<end><assistant>再见！<end>
```

代码 (`tokenizer.py`):

```python
def apply_chat_template(messages, add_generation_prompt=False):
    ids = [bos_id]
    for msg in messages:
        if msg["role"] == "user":
            ids += [user_id] + encode(msg["content"]) + [end_id]
        elif msg["role"] == "assistant":
            ids += [assistant_id] + encode(msg["content"]) + [end_id]
    if add_generation_prompt:
        ids += [assistant_id]   # 让模型从这里开始生成
    return ids
```

**`add_generation_prompt`** 的作用：推理时给 prompt 末尾加上 `<assistant>` 后停止，让模型补全 assistant 的回答。这与训练时模型看到 `<assistant>` 后开始预测内容的模式一致。

**与主流模板对比**:

| 模型      | 模板格式 (简化)                                              |
|----------|--------------------------------------------------------------|
| ChatML   | `<\|im_start\|>user\n...<\|im_end\|>`                       |
| Llama-3  | `<\|start_header_id\|>user<\|end_header_id\|>\n\n...<\|eot_id\|>` |
| Qwen2    | `<\|im_start\|>user\n...<\|im_end\|>` (与 ChatML 相同)     |
| 本项目    | `<user>...<end><assistant>...<end>`                         |

结构本质相同：都是 role 标记 + 内容 + 结束符。

## 3. Loss Masking: 只在 assistant 上算 loss

这是 SFT 最重要的实现细节。

**错误做法**: 把整个对话当成一个序列做语言建模 (loss 算在每个 token 上)。
后果: 模型会学着复述用户的问题，输出风格变成"问答都自己说"。

**正确做法**: target 中 user 部分和特殊 token 部分写 `-100`，让 `cross_entropy` 忽略它们。

```python
# nanollm/data.py 的 SFTDataset._build()
ids: List[int] = [bos_id]
mask: List[int] = [0]              # 1 = 算 loss

for msg in messages:
    content_ids = encode(msg["content"])
    if msg["role"] == "user":
        ids += [user_id] + content_ids + [end_id]
        mask += [0, 0..., 0]                     # 全不算
    elif msg["role"] == "assistant":
        ids += [assistant_id] + content_ids + [end_id]
        mask += [0] + [1]*len(content_ids) + [1] # 内容 + <end> 算
```

**注意**:
- `<assistant>` 这个起始 token **不**算 loss（它是 prompt 的一部分）
- 但 `<end>` 这个**结束 token 必须算 loss** —— 不然模型学不会"什么时候停止"

## 4. SFT 数据格式

本项目用 jsonl，每行一个样本：

```json
{"messages": [
    {"role": "user", "content": "介绍一下相对论"},
    {"role": "assistant", "content": "相对论是由爱因斯坦提出的..."}
]}
```

支持多轮对话:

```json
{"messages": [
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好！"},
    {"role": "user", "content": "今天星期几？"},
    {"role": "assistant", "content": "我不知道，我没有实时信息。"}
]}
```

多轮时，所有 assistant 回合都参与 loss（不只是最后一个）。

## 5. SFT 与预训练的关键差异

| 维度          | 预训练                  | SFT                              |
|--------------|------------------------|---------------------------------|
| 数据格式      | 自然文本流              | 对话 jsonl                       |
| 损失计算位置  | 所有 token              | 只有 assistant 的内容 token       |
| 学习率        | 1e-4 ~ 3e-4            | 1e-5 ~ 5e-5 (小一个量级)         |
| 步数 / 数据量 | 几十亿 ~ 万亿 token      | 几万 ~ 几十万样本                 |
| Warmup        | 几百 ~ 几千步           | 50 ~ 200 步                      |
| Batch size    | 大 (effective 500K+ tok)| 小 (32 ~ 128 样本)              |
| 是否需要 mask | 不需要                  | 必须 mask user 部分              |
| Weight decay  | 0.1                    | 通常 0 或 0.01                  |

**学习率为什么必须小**:
- 大 lr 会破坏预训练学到的"语言能力"
- SFT 数据量小，大 lr 容易过拟合 + 灾难性遗忘

## 6. SFT 数据从哪来？

**公开数据集**:

- **英文**: [Alpaca](https://github.com/tatsu-lab/stanford_alpaca) (52K, GPT-3.5 蒸馏的指令)，[OpenHermes](https://huggingface.co/datasets/teknium/OpenHermes-2.5) (1M, 高质量混合)，[ShareGPT](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered) (多轮对话)
- **中文**: [Belle](https://github.com/LianjiaTech/BELLE) (开放双语)，[COIG](https://huggingface.co/datasets/BAAI/COIG)、[MOSS](https://huggingface.co/datasets/fnlp/moss-002-sft-data)、[Firefly](https://huggingface.co/datasets/YeungNLP/firefly-train-1.1M)

数据规模经验:
- 10K 样本：模型开始"像"在对话
- 50K - 100K：基本可用
- 500K+：跨任务泛化变好

**质量 vs 数量**: LIMA 论文证明 1K 高质量人工样本可以胜过 50K 弱标签样本。SFT 阶段**质量 >> 数量**。

## 7. 常见 pitfall

1. **EOS / end token 没学会**
   - 症状: 模型一直生成不停
   - 原因: 数据里 `<end>` 没参与 loss，或采样时没设 `eos_token_id`

2. **复述用户输入**
   - 症状: 模型先重复一遍用户的问题，再回答
   - 原因: loss mask 没正确实现，user 部分也参与了 loss

3. **回答里出现 `<user>` 标记**
   - 症状: 模型自己 hallucinate 出下一轮对话
   - 原因: SFT 数据里出现过这种模式，或模板渲染时引入了泄漏

4. **灾难性遗忘**
   - 症状: SFT 后 base 语言能力下降，answer 经常胡言乱语
   - 解决: 学习率调小；混入 ~5% 的预训练数据一起训

5. **过拟合**
   - 症状: train loss 持续下降但 eval 上回答模板化、无创意
   - 解决: 减少 epoch（SFT 通常只训 1-3 个 epoch）；增加数据多样性

## 8. SFT 之后：对齐 (本项目未实现)

SFT 后的模型已经能"按格式回答"，但回答经常:
- 啰嗦、模板化
- 错误自信 (hallucination)
- 不够安全 / 礼貌

**对齐 (Alignment)** 阶段解决这些。两种主流方法：

### RLHF (Reinforcement Learning from Human Feedback)
1. 训一个 reward model: 用人类对 (回答 A, 回答 B) 的偏好打分
2. 用 PPO 让 LM 最大化 reward，同时 KL 约束不要偏离 SFT 模型太远

**特点**: 强大但复杂、不稳定、显存翻 4 倍（actor / critic / reward / reference 四份模型）。

### DPO (Direct Preference Optimization)
直接用偏好对 `(prompt, chosen, rejected)` 做监督学习，跳过 reward model 和 RL：

$$\mathcal{L}_{DPO} = -\log \sigma\left(\beta \log \frac{\pi(\text{chosen})}{\pi_{ref}(\text{chosen})} - \beta \log \frac{\pi(\text{rejected})}{\pi_{ref}(\text{rejected})}\right)$$

**特点**: 简单（一份 LM + 一份冻结 ref），稳定，效果接近 PPO。
社区主流选择。Llama-3 chat、Qwen2-chat 等均用 DPO 或其变体 (KTO / SimPO / IPO)。

想自己实验，可以参考 [trl](https://github.com/huggingface/trl) 库的 `DPOTrainer`。

## 9. 接下来

→ `06-inference.md` 看模型怎么"说话"

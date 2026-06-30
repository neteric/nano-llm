# 03 · 分词器: BPE 原理与实现

## 1. 为什么需要分词器

模型只能处理整数 ID。从字符 / 字节到 ID 的映射，是一个**粒度选择**问题：

| 粒度           | 优点                       | 缺点                                  |
|---------------|---------------------------|--------------------------------------|
| 字符级         | 词表小（几千），无 OOV     | 序列变长，相同含义需要更多 token       |
| 单词级         | 序列短，语义直观           | 词表巨大，OOV 严重（new 词没法编码）   |
| **子词 BPE**   | 词表可控，常见词单 token  | 复杂罕见词拆成多个，但不会 OOV         |

BPE（Byte Pair Encoding）是当前事实上的标准（GPT 系列、Llama、Qwen 全是 BPE 变种）。

## 2. BPE 算法

**训练**:

1. 初始化词表 = 所有单字节 / 单字符
2. 在语料里统计**相邻 token 对**的频次
3. 把最高频的一对合并成一个新 token，加入词表
4. 重复 2-3，直到词表达到目标大小

**编码**:

把文本拆成最细粒度的 token，然后**贪心地按训练时学到的合并规则**逐步合并，直到无法再合并。

例：训练时学到合并规则 `(' ', 't') → ' t'`, `(' t', 'he') → ' the'`

```
"the cat" 编码过程:
  ['t', 'h', 'e', ' ', 'c', 'a', 't']
→ apply (' ', 't'):           不适用（' t' 没出现）
→ apply ('t', 'h')→'th':      ['th', 'e', ' ', 'c', 'a', 't']
→ apply ('th', 'e')→'the':    ['the', ' ', 'c', 'a', 't']
→ ...
```

## 3. ByteLevel BPE

直接在 unicode 字符上做 BPE，会遇到中文/日文/emoji 等字符集巨大的问题。
**ByteLevel BPE** 先把文本转成 UTF-8 字节流（256 种），再做 BPE。优点：

- 任何 unicode 都能表示，**绝不会 OOV**
- 基础字母表只有 256，可控

GPT-2 / GPT-3 / Llama 全用 ByteLevel BPE。本项目也用。

## 4. 本项目的特殊 token

```python
SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>", "<user>", "<assistant>", "<end>"]
                   [0]      [1]     [2]      [3]      [4]       [5]            [6]
```

- `<pad>`: 把变长 batch pad 到统一长度
- `<bos>`: 序列起始
- `<eos>`: 预训练阶段标记一段文档结束（让模型学会"结束"这个概念）
- `<unk>`: ByteLevel BPE 理论上不该出现，保留作为兜底
- `<user>`, `<assistant>`, `<end>`: SFT 阶段的对话模板，详见 `05-sft.md`

## 5. 训练

```bash
python scripts/train_tokenizer.py \
    --corpus data/pretrain_sample.txt \
    --vocab_size 6400 \
    --out data/tokenizer.json
```

底层用 huggingface `tokenizers` 库（Rust 实现），训练几 MB 文本只需几秒。
对于真实场景的 1B+ token 语料，训 vocab=32k 大约 10-30 分钟。

**vocab_size 怎么选？**

经验法则：
- 玩具模型 (< 100M)：2k - 8k
- 小模型 (100M - 1B)：16k - 32k
- 中大模型 (> 1B)：32k - 128k
- 多语言模型：128k+ (Llama-3 用 128256, Qwen2 用 151936)

太小 → 序列变长，长上下文吃显存；
太大 → embedding 占参数比例过高，且每个 token 看到的次数变少（学不充分）。

## 6. tokens 的 "形状感觉"

训出 tokenizer 后，重要的是建立一种对 token 的**直觉**：

```python
from nanollm.tokenizer import NanoTokenizer
tk = NanoTokenizer.load("data/tokenizer.json")

print(tk.encode("Hello, world!"))
# 英文常见词通常 1 token

print(tk.encode("Antidisestablishmentarianism"))
# 罕见英文长词 → 拆 4-5 个 token

print(tk.encode("我爱北京天安门"))
# 中文常见字 1-2 token，罕见字可能 3+ token
```

**一个粗略经验** (英文 + 中文混合训练的 GPT-4 风格 tokenizer):

- 1 个英文 token ≈ 0.75 个英文单词
- 1 个中文 token ≈ 0.6 个汉字（GPT-4 早期，Qwen 等中文优化 tokenizer ≈ 1 个汉字）

## 7. 训练数据要不要先洗？

**强烈建议**:

1. **去重** —— 重复文档严重伤害训练效果（甚至导致模型背诵）
2. **质量过滤** —— 去除乱码 / 网页样板 / 重复 n-gram
3. **混合配比** —— 不同来源数据（网页 / 书 / 代码 / 中文 / 英文）的比例显著影响最终能力

本项目脚本不做这些，只是把单文件 tokenize。生产场景请用 [datatrove](https://github.com/huggingface/datatrove)、[CCNet](https://github.com/facebookresearch/cc_net) 或自研 pipeline。

## 8. 接下来

→ `04-pretrain.md` 看 token 流如何变成模型权重

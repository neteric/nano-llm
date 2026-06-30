# nano-llm: 麻雀虽小、五脏俱全的 LLM 全流程实现

一个用于**理解大模型训练 / 推理全过程**的最小可用实现。约 26M 参数（可调小至 1M），完整覆盖：

1. **BPE 分词器训练** —— 从原始文本到 token ID
2. **预训练 (Pretraining)** —— 在自然语料上做自回归语言建模
3. **监督微调 (SFT)** —— 对话格式数据 + loss masking
4. **推理 (Inference)** —— 带 KV cache 的自回归生成，支持 top-k / top-p / temperature 采样

架构与现代 LLM 对齐：**RMSNorm + RoPE + SwiGLU + GQA + Pre-Norm**，是 Llama / Qwen / Mistral 同款。
代码总量约 1500 行 Python，可读完。

## 项目结构

```
nano-llm/
├── nanollm/                  # 核心库
│   ├── config.py             # 模型与训练配置
│   ├── model.py              # Transformer 实现 (RMSNorm/RoPE/GQA/SwiGLU/KV cache)
│   ├── tokenizer.py          # BPE 分词器封装 + chat template
│   ├── data.py               # 预训练 & SFT 数据加载
│   └── utils.py              # LR 调度 / checkpoint / 设备检测
├── scripts/                  # 入口脚本
│   ├── make_sample_data.py   # 生成合成语料 (无需联网)
│   ├── train_tokenizer.py    # 训练 BPE
│   ├── prepare_data.py       # 文本 → 二进制 token 流
│   ├── pretrain.py           # 预训练
│   ├── sft.py                # SFT
│   └── generate.py           # 推理 / 对话
├── docs/                     # 详细文档（强烈建议阅读）
│   ├── 01-overview.md        # 全流程概览
│   ├── 02-architecture.md    # 模型架构详解
│   ├── 03-tokenizer.md       # 分词器原理与实现
│   ├── 04-pretrain.md        # 预训练详解
│   ├── 05-sft.md             # SFT 详解
│   ├── 06-inference.md       # KV cache 与采样
│   └── 07-experiments.md     # 推荐实验
├── tests/
│   └── test_model.py         # 模型烟雾测试 (含 KV cache 正确性)
└── requirements.txt          # torch + tokenizers + numpy
```

## 安装

```bash
pip install -r requirements.txt
# 验证模型可跑通
PYTHONPATH=. python tests/test_model.py
```

## 5 分钟跑通完整流水线 (CPU 即可)

```bash
cd nano-llm
export PYTHONPATH=.

# 1) 生成合成语料 (约 1 MB 文本)
python scripts/make_sample_data.py

# 2) 训练 BPE 分词器 (vocab=2000)
python scripts/train_tokenizer.py \
    --corpus data/pretrain_sample.txt \
    --vocab_size 2000 \
    --out data/tokenizer.json

# 3) 把文本打包成二进制 token 流
python scripts/prepare_data.py \
    --tokenizer data/tokenizer.json \
    --corpus data/pretrain_sample.txt \
    --out data/pretrain.bin

# 4) 预训练（CPU 上 ~5 分钟训完 500 步）
python scripts/pretrain.py \
    --data data/pretrain.bin --tokenizer data/tokenizer.json \
    --device cpu --dtype float32 \
    --d_model 128 --n_layers 4 --n_heads 4 --n_kv_heads 2 --d_ff 256 \
    --batch_size 8 --seq_len 64 \
    --max_steps 500 --warmup_steps 50

# 5) SFT
python scripts/sft.py \
    --pretrain_ckpt checkpoints/pretrain_final.pt \
    --tokenizer data/tokenizer.json \
    --data data/sft_sample.jsonl \
    --device cpu --dtype float32 \
    --batch_size 4 --seq_len 128 \
    --max_steps 300

# 6) 对话
python scripts/generate.py \
    --ckpt checkpoints/sft_final.pt \
    --tokenizer data/tokenizer.json \
    --device cpu --dtype float32 \
    --chat
```

> 注：合成语料只用于验证流水线，模型不会有真正的能力。要训出有意思的小模型，参考 `docs/07-experiments.md` 换 TinyStories / WikiText / 中文维基百科子集等真实语料。

## GPU 配置参考

| 模型规模 | d_model | n_layers | 参数量 | 单卡 (24G) seq_len | 大致语料量 |
|---------|---------|----------|--------|-------------------|-----------|
| Toy     | 128     | 4        | 1M     | 256, bs=64        | 10 MB     |
| Default | 512     | 8        | 26M    | 512, bs=32        | 500 MB    |
| Small   | 768     | 12       | 110M   | 1024, bs=8        | 5 GB+     |

## 学习路径建议

新手按文档编号 01 → 07 顺序读，每读完一个文档对照 `nanollm/*.py` 中对应模块的代码再读一遍：

1. **`docs/01-overview.md`** —— 先建立全局图景
2. **`docs/02-architecture.md`** —— 配合 `model.py` 读
3. **`docs/03-tokenizer.md`** —— 配合 `tokenizer.py` 读
4. **`docs/04-pretrain.md`** —— 配合 `data.py` + `scripts/pretrain.py`
5. **`docs/05-sft.md`** —— 配合 `data.py` 的 `SFTDataset` + `scripts/sft.py`
6. **`docs/06-inference.md`** —— 配合 `model.py` 的 `generate()`
7. **`docs/07-experiments.md`** —— 上手改代码做实验

## License

MIT —— 学习用，随便改、随便用。

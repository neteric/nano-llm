#!/usr/bin/env bash
# TinyStories 完整训练流水线
# 执行方式: bash run_tinystories.sh 2>&1 | tee logs/run_tinystories.log
set -euo pipefail

PYTHON=".venv/bin/python"
TORCHRUN=".venv/bin/torchrun"
DATA_DIR="data"
CKPT_DIR="checkpoints/tinystories"
LOG_DIR="logs"

mkdir -p "$CKPT_DIR" "$LOG_DIR"

echo "============================================================"
echo " nano-llm × TinyStories 训练流水线"
echo " 开始时间: $(date)"
echo "============================================================"

# ── Step 1: 下载 TinyStories ──────────────────────────────────
echo ""
echo "[Step 1] 下载 TinyStories 数据集"
$PYTHON - <<'EOF'
from datasets import load_dataset
import os

out_path = "data/tinystories_train.txt"
if os.path.exists(out_path):
    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"  已存在 {out_path} ({size_mb:.1f} MB)，跳过下载")
else:
    print("  正在从 HuggingFace 下载 TinyStories train split ...")
    ds = load_dataset("roneneldan/TinyStories", split="train")
    print(f"  共 {len(ds):,} 条故事，写入 {out_path} ...")
    with open(out_path, "w", encoding="utf-8") as f:
        for row in ds:
            text = row["text"].strip()
            if text:
                f.write(text + "\n")
    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"  完成: {size_mb:.1f} MB")
EOF

# ── Step 2: 训练 BPE 分词器 ───────────────────────────────────
echo ""
echo "[Step 2] 训练 BPE 分词器 (vocab_size=6400)"
if [ -f "$DATA_DIR/tinystories_tokenizer.json" ]; then
    echo "  已存在，跳过"
else
    $PYTHON scripts/train_tokenizer.py \
        --corpus "$DATA_DIR/tinystories_train.txt" \
        --vocab_size 6400 \
        --out "$DATA_DIR/tinystories_tokenizer.json"
fi

# ── Step 3: 文本 → 二进制 token 流 ───────────────────────────
echo ""
echo "[Step 3] 预处理语料 → 二进制 token 流"
if [ -f "$DATA_DIR/tinystories_pretrain.bin" ]; then
    echo "  已存在，跳过"
else
    $PYTHON scripts/prepare_data.py \
        --tokenizer "$DATA_DIR/tinystories_tokenizer.json" \
        --corpus "$DATA_DIR/tinystories_train.txt" \
        --out "$DATA_DIR/tinystories_pretrain.bin"
fi

# ── Step 4: 预训练 ────────────────────────────────────────────
echo ""
echo "[Step 4] 预训练 (26M params, 2×L20, bfloat16, 5000 steps)"
if [ -f "$CKPT_DIR/pretrain_final.pt" ]; then
    echo "  已存在 $CKPT_DIR/pretrain_final.pt，跳过"
else
$TORCHRUN --nproc_per_node=2 scripts/pretrain.py \
    --data "$DATA_DIR/tinystories_pretrain.bin" \
    --tokenizer "$DATA_DIR/tinystories_tokenizer.json" \
    --device cuda --dtype bfloat16 \
    --d_model 512 --n_layers 8 --n_heads 8 --n_kv_heads 2 --d_ff 1408 \
    --batch_size 32 --seq_len 256 \
    --max_steps 5000 --warmup_steps 200 \
    --save_interval 1000 \
    --out_dir "$CKPT_DIR"
fi

# ── Step 5: 生成示例 ──────────────────────────────────────────
echo ""
echo "[Step 5] 生成示例"
for PROMPT in "Once upon a time" "There was a little girl" "The dog and the cat"; do
    echo ""
    echo "  prompt: \"$PROMPT\""
    $PYTHON scripts/generate.py \
        --ckpt "$CKPT_DIR/pretrain_final.pt" \
        --tokenizer "$DATA_DIR/tinystories_tokenizer.json" \
        --device cuda --dtype bfloat16 \
        --prompt "$PROMPT" \
        --max_new_tokens 150 --temperature 0.8 --top_p 0.9
done

# ── Step 6: 生成 SFT 数据 ─────────────────────────────────────
echo ""
echo "[Step 6] 生成 TinyStories SFT 数据 (20000 条)"
if [ -f "$DATA_DIR/tinystories_sft.jsonl" ]; then
    echo "  已存在，跳过"
else
    $PYTHON - <<'EOF'
import json, re, random

TEMPLATES_NAMED = [
    "Tell me a short story about {name}.",
    "Can you tell a story featuring a character named {name}?",
    "Write a simple bedtime story with {name} as the main character.",
    "Please tell a children's story about {name}.",
    "Make up a story for kids about {name}.",
    "Create a short story where {name} is the hero.",
    "Tell a fun story for young children about {name}.",
    "Write a story for a child that involves a character called {name}.",
]
TEMPLATES_GENERIC = [
    "Tell me a short story for children.",
    "Write a simple bedtime story.",
    "Tell a fun story for kids.",
    "Make up a short children's story.",
]
STOP = {'Once','One','There','A','The','In','On','It','He','She','They','His','Her'}

def extract_name(story):
    m = re.search(r'\bnamed\s+([A-Z][a-z]+)', story)
    if m:
        return m.group(1)
    for word in story.split()[:30]:
        w = re.sub(r"[^A-Za-z]", "", word)
        if w and w[0].isupper() and w not in STOP and len(w) > 2:
            return w
    return None

rng = random.Random(42)
out_path = "data/tinystories_sft.jsonl"
n_target = 20000
count = 0
with open("data/tinystories_train.txt", encoding="utf-8") as fin, \
     open(out_path, "w", encoding="utf-8") as fout:
    for line in fin:
        if count >= n_target:
            break
        story = line.strip()
        if len(story) < 50:
            continue
        if len(story) > 2000:
            cut = story[:2000].rfind('.')
            story = story[:cut+1] if cut > 100 else story[:2000]
        name = extract_name(story)
        if name and rng.random() < 0.8:
            question = rng.choice(TEMPLATES_NAMED).format(name=name)
        else:
            question = rng.choice(TEMPLATES_GENERIC)
        fout.write(json.dumps({
            "messages": [
                {"role": "user",      "content": question},
                {"role": "assistant", "content": story},
            ]
        }, ensure_ascii=False) + "\n")
        count += 1
print(f"  生成 {count} 条 -> {out_path}")
EOF
fi

# ── Step 7: SFT 微调 ──────────────────────────────────────────
echo ""
echo "[Step 7] SFT 微调 (2000 steps)"
$PYTHON scripts/sft.py \
    --pretrain_ckpt "$CKPT_DIR/pretrain_final.pt" \
    --tokenizer "$DATA_DIR/tinystories_tokenizer.json" \
    --data "$DATA_DIR/tinystories_sft.jsonl" \
    --device cuda --dtype bfloat16 \
    --batch_size 16 --seq_len 256 \
    --learning_rate 3e-5 --min_lr 3e-6 \
    --warmup_steps 100 --max_steps 2000 \
    --save_interval 500 \
    --out_dir "$CKPT_DIR"

# ── Step 8: 对比验证 ──────────────────────────────────────────
echo ""
echo "[Step 8] 对比验证：预训练 vs SFT"
$PYTHON - <<'EOF'
import torch
from nanollm.config import ModelConfig
from nanollm.model import NanoLLM
from nanollm.tokenizer import NanoTokenizer

CKPT_DIR  = "checkpoints/tinystories"
TOKENIZER = "data/tinystories_tokenizer.json"
DEVICE    = "cuda"

tk = NanoTokenizer.load(TOKENIZER)

def load_model(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    m = NanoLLM(ModelConfig(**ckpt["model_config"])).to(DEVICE)
    m.load_state_dict(ckpt["model"])
    return m.eval()

def gen_pretrain(model, prompt, max_new=150):
    ids = tk.encode(prompt, add_bos=True)
    x = torch.tensor([ids], device=DEVICE)
    with torch.no_grad():
        out = model.generate(x, max_new_tokens=max_new, temperature=0.8, top_p=0.9)
    return tk.decode(out[0, len(ids):].tolist())

def gen_sft(model, question, max_new=200):
    ids = tk.apply_chat_template(
        [{"role": "user", "content": question}],
        add_generation_prompt=True)
    x = torch.tensor([ids], device=DEVICE)
    with torch.no_grad():
        out = model.generate(x, max_new_tokens=max_new, temperature=0.8,
                             top_p=0.9, eos_token_id=tk.end_id)
    tokens = out[0, len(ids):].tolist()
    if tokens and tokens[-1] == tk.end_id:
        tokens = tokens[:-1]
    return tk.decode(tokens)

pretrain = load_model(f"{CKPT_DIR}/pretrain_final.pt")
sft      = load_model(f"{CKPT_DIR}/sft_final.pt")

TESTS = [
    "Tell me a short story about a girl named Emma.",
    "Write a bedtime story for children.",
    "Tell a story about a dog who learns something new.",
]
for q in TESTS:
    print(f"\n{'='*60}")
    print(f"问题:        {q}")
    print(f"[预训练续写] {gen_pretrain(pretrain, q)[:300]}")
    print(f"[SFT 对话]   {gen_sft(sft, q)[:300]}")
EOF

echo ""
echo "============================================================"
echo " 全部完成: $(date)"
echo "============================================================"

# ── 对话测试（单独运行此段）────────────────────────────────────
# ssh root@10.138.0.26
# cd /root/nano-llm
# .venv/bin/python scripts/generate.py \
#     --ckpt checkpoints/tinystories/sft_final.pt \
#     --tokenizer data/tinystories_tokenizer.json \
#     --device cuda --dtype bfloat16 \
#     --chat
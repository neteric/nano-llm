#!/usr/bin/env bash
# TinyStories 完整训练流水线
# 执行方式: bash run_tinystories.sh 2>&1 | tee logs/run_tinystories.log
set -euo pipefail

PYTHON=".venv/bin/python"
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
echo "[Step 4] 预训练 (26M params, cuda, bfloat16, 5000 steps)"
$PYTHON scripts/pretrain.py \
    --data "$DATA_DIR/tinystories_pretrain.bin" \
    --tokenizer "$DATA_DIR/tinystories_tokenizer.json" \
    --device cuda --dtype bfloat16 \
    --d_model 512 --n_layers 8 --n_heads 8 --n_kv_heads 2 --d_ff 1408 \
    --batch_size 32 --seq_len 256 \
    --max_steps 5000 --warmup_steps 200 \
    --save_interval 1000 \
    --out_dir "$CKPT_DIR"

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

echo ""
echo "============================================================"
echo " 全部完成: $(date)"
echo "============================================================"

"""训练 BPE 分词器。

用法:
    python scripts/train_tokenizer.py \
        --corpus data/pretrain_sample.txt \
        --vocab_size 6400 \
        --out data/tokenizer.json
"""
import argparse
from pathlib import Path

from nanollm.tokenizer import NanoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", nargs="+", required=True, help="一个或多个文本文件")
    ap.add_argument("--vocab_size", type=int, default=6400)
    ap.add_argument("--min_frequency", type=int, default=2)
    ap.add_argument("--out", default="data/tokenizer.json")
    args = ap.parse_args()

    print(f"[1/3] 训练 BPE：vocab_size={args.vocab_size}, files={args.corpus}")
    tk = NanoTokenizer.train_from_files(
        args.corpus,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    tk.save(args.out)
    print(f"[2/3] 已保存 -> {args.out}（实际词表大小 {tk.vocab_size}）")

    # sanity check
    text = "你好，世界！Hello, world!"
    ids = tk.encode(text)
    back = tk.decode(ids)
    print(f"[3/3] sanity check:\n  text  : {text!r}\n  tokens: {ids}\n  decode: {back!r}")


if __name__ == "__main__":
    main()

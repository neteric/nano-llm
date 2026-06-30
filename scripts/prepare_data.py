"""把文本语料 tokenize 并打包成二进制 token 流。

用法:
    python scripts/prepare_data.py \
        --tokenizer data/tokenizer.json \
        --corpus data/pretrain_sample.txt \
        --out data/pretrain.bin
"""
import argparse
from pathlib import Path

from nanollm.tokenizer import NanoTokenizer
from nanollm.data import tokenize_corpus_to_bin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="data/tokenizer.json")
    ap.add_argument("--corpus", nargs="+", required=True)
    ap.add_argument("--out", default="data/pretrain.bin")
    ap.add_argument("--dtype", choices=["uint16", "uint32"], default="uint16",
                    help="词表 < 65536 时用 uint16 省一半磁盘空间")
    args = ap.parse_args()

    tk = NanoTokenizer.load(args.tokenizer)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    tokenize_corpus_to_bin(args.corpus, tk, args.out, dtype=args.dtype)


if __name__ == "__main__":
    main()

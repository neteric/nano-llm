"""在验证集上计算 perplexity。

用法:
    python scripts/eval_ppl.py \
        --ckpt checkpoints/tinystories/pretrain_final.pt \
        --tokenizer data/tinystories_tokenizer.json \
        --data data/tinystories_val.bin \
        --seq_len 256 --device cuda
"""
import argparse
import math
import torch
import numpy as np
from nanollm.model import NanoLLM
from nanollm.config import ModelConfig
from nanollm.utils import load_checkpoint_into


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data", required=True, help="二进制 token 文件 (.bin)")
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--bin_dtype", default="uint16")
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]

    # 加载模型
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**ckpt["model_config"])
    model = NanoLLM(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model = model.to(dtype).eval()
    print(f"== loaded {args.ckpt} ({model.num_parameters()/1e6:.1f}M params) ==")

    # 加载验证数据
    np_dtype = np.uint16 if args.bin_dtype == "uint16" else np.uint32
    data = np.memmap(args.data, dtype=np_dtype, mode="r")
    total_tokens = len(data)
    print(f"== val tokens: {total_tokens:,} ==")

    # 滑动窗口，不重叠
    chunk = args.seq_len + 1
    n_chunks = total_tokens // chunk
    total_loss = 0.0
    total_toks = 0

    with torch.no_grad():
        for start in range(0, n_chunks * chunk, args.batch_size * chunk):
            batch_indices = range(start, min(start + args.batch_size * chunk, n_chunks * chunk), chunk)
            if not batch_indices:
                break
            xs, ys = [], []
            for i in batch_indices:
                seg = data[i: i + chunk].astype(np.int64)
                xs.append(seg[:-1])
                ys.append(seg[1:])
            x = torch.tensor(np.stack(xs), device=device)
            y = torch.tensor(np.stack(ys), device=device)
            with torch.autocast(device_type=args.device, dtype=dtype):
                out = model(x, targets=y)
            # cross_entropy 返回的是 mean，还原 sum
            valid = (y != -100).sum().item()
            total_loss += out.loss.item() * valid
            total_toks += valid

    ppl = math.exp(total_loss / total_toks)
    print(f"\n{'='*40}")
    print(f"  val loss : {total_loss/total_toks:.4f}")
    print(f"  val PPL  : {ppl:.2f}")
    print(f"  tokens   : {total_toks:,}")
    print(f"{'='*40}")


if __name__ == "__main__":
    main()

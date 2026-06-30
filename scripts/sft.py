"""监督微调 (SFT) 入口。

与预训练的差异:
    1. 数据集是对话格式 (jsonl)，用 SFTDataset 而非二进制 token 流
    2. loss 只算在 assistant 回答 token 上 (其他位置 target=-100)
    3. 学习率显著更小 (通常 1e-5 ~ 5e-5)，步数也更少
    4. 必须从一个已有的预训练 checkpoint 开始

用法:
    python scripts/sft.py \
        --pretrain_ckpt checkpoints/pretrain_final.pt \
        --tokenizer data/tokenizer.json \
        --data data/sft_sample.jsonl \
        --device cuda --dtype bfloat16 \
        --max_steps 1000 --batch_size 8
"""
import argparse
import math
import os
import time
from pathlib import Path

import torch

from nanollm.config import ModelConfig, SFTConfig
from nanollm.model import NanoLLM
from nanollm.tokenizer import NanoTokenizer
from nanollm.data import make_sft_loader
from nanollm.utils import (
    set_seed, detect_device, get_dtype, cosine_lr,
    configure_optimizer, save_checkpoint, load_checkpoint_into,
    human_count,
)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrain_ckpt", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--learning_rate", type=float, default=5e-5)
    ap.add_argument("--min_lr", type=float, default=5e-6)
    ap.add_argument("--max_steps", type=int, default=1000)
    ap.add_argument("--warmup_steps", type=int, default=50)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--log_interval", type=int, default=10)
    ap.add_argument("--save_interval", type=int, default=500)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"])
    ap.add_argument("--out_dir", default="checkpoints")
    ap.add_argument("--seed", type=int, default=1337)
    return ap.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = detect_device(args.device)
    dtype = get_dtype(args.dtype)

    # 1) 加载预训练 checkpoint 中的模型结构
    print(f"== loading pretrain ckpt: {args.pretrain_ckpt} ==")
    ckpt = torch.load(args.pretrain_ckpt, map_location="cpu", weights_only=False)
    mcfg = ModelConfig(**ckpt["model_config"])
    model = NanoLLM(mcfg)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    print(f"== params: {human_count(model.num_parameters())} ==")

    # 2) tokenizer 必须与预训练时一致
    tk = NanoTokenizer.load(args.tokenizer)
    assert tk.vocab_size == mcfg.vocab_size, "tokenizer 与模型 vocab_size 不匹配"

    # 3) 数据 & 优化器
    tcfg = SFTConfig(
        data_path=args.data, batch_size=args.batch_size, seq_len=args.seq_len,
        learning_rate=args.learning_rate, min_lr=args.min_lr,
        weight_decay=args.weight_decay, grad_clip=args.grad_clip,
        max_steps=args.max_steps, warmup_steps=args.warmup_steps,
        device=device, dtype=args.dtype, out_dir=args.out_dir,
        pretrain_ckpt=args.pretrain_ckpt,
    )
    loader = make_sft_loader(tcfg, tk)
    optimizer = configure_optimizer(
        model, weight_decay=tcfg.weight_decay,
        lr=tcfg.learning_rate, betas=(tcfg.beta1, tcfg.beta2), device=device,
    )

    autocast_ctx = torch.amp.autocast(device_type=device, dtype=dtype) if device != "cpu" or dtype != torch.float32 \
        else torch.amp.autocast(device_type="cpu", enabled=False)
    scaler = torch.amp.GradScaler(device=device, enabled=(dtype == torch.float16))

    Path(tcfg.out_dir).mkdir(parents=True, exist_ok=True)
    model.train()
    step = 0
    t0 = time.time()
    data_iter = iter(loader)
    while step < tcfg.max_steps:
        lr = cosine_lr(step, warmup_steps=tcfg.warmup_steps, max_steps=tcfg.max_steps,
                       max_lr=tcfg.learning_rate, min_lr=tcfg.min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            x, y = next(data_iter)
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast_ctx:
            out = model(x, targets=y)
            loss = out.loss
        scaler.scale(loss).backward()
        if tcfg.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        step += 1
        if step % tcfg.log_interval == 0:
            dt = time.time() - t0
            print(f"step {step:5d} | loss {loss.item():7.4f} | lr {lr:.2e} "
                  f"| ppl {math.exp(min(20, loss.item())):8.2f} | dt {dt:.2f}s")
            t0 = time.time()

        if step % tcfg.save_interval == 0 or step == tcfg.max_steps:
            ckpt_path = os.path.join(tcfg.out_dir, f"sft_step{step}.pt")
            save_checkpoint(ckpt_path, model, optimizer, step, mcfg)
            print(f"  -> saved {ckpt_path}")

    final_path = os.path.join(tcfg.out_dir, "sft_final.pt")
    save_checkpoint(final_path, model, optimizer, step, mcfg)
    print(f"\nSFT 完成 -> {final_path}")


if __name__ == "__main__":
    main()

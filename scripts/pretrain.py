"""预训练入口。

最小用法 (在 CPU 上验证流程，~ 几分钟):
    python scripts/pretrain.py \
        --data data/pretrain.bin \
        --tokenizer data/tokenizer.json \
        --device cpu --dtype float32 \
        --max_steps 200 --batch_size 4 --seq_len 128

GPU 用法:
    python scripts/pretrain.py \
        --data data/pretrain.bin \
        --tokenizer data/tokenizer.json \
        --device cuda --dtype bfloat16 \
        --max_steps 5000 --batch_size 16 --seq_len 512
"""
import argparse
import math
import os
import time
from pathlib import Path

import torch

from nanollm.config import ModelConfig, TrainConfig
from nanollm.model import NanoLLM
from nanollm.tokenizer import NanoTokenizer
from nanollm.data import make_pretrain_loader
from nanollm.utils import (
    set_seed, detect_device, get_dtype, cosine_lr,
    configure_optimizer, save_checkpoint, human_count,
)


def parse_args():
    ap = argparse.ArgumentParser()
    # 数据
    ap.add_argument("--data", required=True)
    ap.add_argument("--tokenizer", required=True)
    # 模型（如果想覆盖默认，可在这里改；不写就用 ModelConfig 的默认值）
    ap.add_argument("--d_model", type=int, default=512)
    ap.add_argument("--n_layers", type=int, default=8)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--n_kv_heads", type=int, default=2)
    ap.add_argument("--d_ff", type=int, default=1408)
    # 训练
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--grad_accum_steps", type=int, default=1)
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--learning_rate", type=float, default=3e-4)
    ap.add_argument("--min_lr", type=float, default=3e-5)
    ap.add_argument("--max_steps", type=int, default=5000)
    ap.add_argument("--warmup_steps", type=int, default=100)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--log_interval", type=int, default=10)
    ap.add_argument("--save_interval", type=int, default=1000)
    # 系统
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"])
    ap.add_argument("--out_dir", default="checkpoints")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--bin_dtype", default="uint16", choices=["uint16", "uint32"])
    return ap.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = detect_device(args.device)
    dtype = get_dtype(args.dtype)
    print(f"== device = {device}, dtype = {args.dtype} ==")

    # 1) tokenizer & model config
    tk = NanoTokenizer.load(args.tokenizer)
    mcfg = ModelConfig(
        vocab_size=tk.vocab_size,
        d_model=args.d_model, n_layers=args.n_layers,
        n_heads=args.n_heads, n_kv_heads=args.n_kv_heads,
        d_ff=args.d_ff, max_seq_len=max(args.seq_len, 256),
    )
    tcfg = TrainConfig(
        data_path=args.data, batch_size=args.batch_size, seq_len=args.seq_len,
        learning_rate=args.learning_rate, min_lr=args.min_lr,
        weight_decay=args.weight_decay, grad_clip=args.grad_clip,
        max_steps=args.max_steps, warmup_steps=args.warmup_steps,
        device=device, dtype=args.dtype, out_dir=args.out_dir,
    )

    # 2) model
    model = NanoLLM(mcfg).to(device)
    n_total = model.num_parameters()
    n_non_emb = model.num_parameters(exclude_embedding=True)
    print(f"== params: total = {human_count(n_total)} | non-embedding = {human_count(n_non_emb)} ==")

    # 3) optimizer
    optimizer = configure_optimizer(
        model, weight_decay=tcfg.weight_decay,
        lr=tcfg.learning_rate, betas=(tcfg.beta1, tcfg.beta2), device=device,
    )

    # 4) data
    loader = make_pretrain_loader(tcfg, dtype=args.bin_dtype)

    # 5) mixed precision setup
    autocast_ctx = torch.amp.autocast(device_type=device, dtype=dtype) if device != "cpu" or dtype != torch.float32 \
        else torch.amp.autocast(device_type="cpu", enabled=False)
    scaler = torch.amp.GradScaler(device=device, enabled=(dtype == torch.float16))

    # 6) 训练循环
    Path(tcfg.out_dir).mkdir(parents=True, exist_ok=True)
    model.train()
    step = 0
    t0 = time.time()
    loss_accum = 0.0
    data_iter = iter(loader)

    while step < tcfg.max_steps:
        # 学习率
        lr = cosine_lr(step, warmup_steps=tcfg.warmup_steps, max_steps=tcfg.max_steps,
                       max_lr=tcfg.learning_rate, min_lr=tcfg.min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # 梯度累积
        optimizer.zero_grad(set_to_none=True)
        micro_loss = 0.0
        for _ in range(args.grad_accum_steps):
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                x, y = next(data_iter)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            with autocast_ctx:
                out = model(x, targets=y)
                loss = out.loss / args.grad_accum_steps
            scaler.scale(loss).backward()
            micro_loss += loss.item()

        # 梯度裁剪
        if tcfg.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        loss_accum = micro_loss * args.grad_accum_steps
        step += 1

        if step % tcfg.log_interval == 0:
            dt = time.time() - t0
            tok_per_step = tcfg.batch_size * args.grad_accum_steps * tcfg.seq_len
            tok_per_sec = tok_per_step * tcfg.log_interval / dt
            print(f"step {step:5d} | loss {loss_accum:7.4f} | lr {lr:.2e} "
                  f"| ppl {math.exp(min(20, loss_accum)):8.2f} | {tok_per_sec:,.0f} tok/s")
            t0 = time.time()

        if step % tcfg.save_interval == 0 or step == tcfg.max_steps:
            ckpt_path = os.path.join(tcfg.out_dir, f"pretrain_step{step}.pt")
            save_checkpoint(ckpt_path, model, optimizer, step, mcfg)
            print(f"  -> saved {ckpt_path}")

    final_path = os.path.join(tcfg.out_dir, "pretrain_final.pt")
    save_checkpoint(final_path, model, optimizer, step, mcfg)
    print(f"\n训练完成 -> {final_path}")


if __name__ == "__main__":
    main()

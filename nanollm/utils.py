"""通用工具: 学习率调度、checkpoint 存取、设备检测。"""
from __future__ import annotations
import math
import os
import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def detect_device(prefer: str = "cuda") -> str:
    if prefer == "cuda" and torch.cuda.is_available():
        return "cuda"
    if prefer == "mps" and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_dtype(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[name]


def cosine_lr(step: int, *, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    """带 warmup 的 cosine 衰减学习率。

    阶段 1 [0, warmup_steps): 线性从 0 升到 max_lr
    阶段 2 [warmup_steps, max_steps]: cosine 从 max_lr 衰减到 min_lr
    阶段 3 (max_steps, ...): 保持 min_lr
    """
    if step < warmup_steps:
        return max_lr * (step + 1) / max(1, warmup_steps)
    if step >= max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def configure_optimizer(model: torch.nn.Module, weight_decay: float, lr: float,
                        betas: tuple, device: str) -> torch.optim.Optimizer:
    """AdamW with weight decay only on 2D+ params (典型做法，避免对 norm/bias 衰减)。"""
    decay_params, no_decay_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() >= 2:
            decay_params.append(p)
        else:
            no_decay_params.append(p)
    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    fused = device == "cuda" and "fused" in torch.optim.AdamW.__init__.__code__.co_varnames
    extra = {"fused": True} if fused else {}
    return torch.optim.AdamW(param_groups, lr=lr, betas=betas, **extra)


def save_checkpoint(path: str, model, optimizer, step: int, cfg, extra: Optional[dict] = None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "step": step,
        "model_config": cfg.__dict__,
    }
    if extra:
        ckpt.update(extra)
    torch.save(ckpt, path)


def load_checkpoint_into(model, path: str, map_location: str = "cpu"):
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model"])
    return ckpt


def human_count(n: int) -> str:
    for unit in ["", "K", "M", "B", "T"]:
        if abs(n) < 1000:
            return f"{n:.1f}{unit}" if unit else f"{n}"
        n /= 1000
    return f"{n:.1f}P"

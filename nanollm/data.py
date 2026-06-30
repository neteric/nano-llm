"""
数据集与数据加载。

两种数据格式:

  1) 预训练:
     - 输入: 普通文本文件
     - 流程: tokenize -> 拼成一条巨大的 token 流 -> 存为 uint16/uint32 二进制
     - 训练时随机从流中切 seq_len+1 长度的片段，前 seq_len 个是输入，后 seq_len 个是 target
     - 这种 "packing" 做法是 GPT 系列标准操作，效率远高于 padding

  2) SFT:
     - 输入: jsonl，每行是 {"messages": [{"role": "...", "content": "..."}, ...]}
     - 用 chat_template 渲染成 ids，然后用 loss_mask 标记哪些位置参与 loss
     - 只有 assistant 回合的 token 计入 loss (典型做法)
"""
from __future__ import annotations
import json
import os
import random
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .tokenizer import NanoTokenizer, USER_ID, ASSISTANT_ID, END_ID, PAD_ID, BOS_ID


# ============================================================================
# 预训练
# ============================================================================
def tokenize_corpus_to_bin(
    text_files: List[str],
    tokenizer: NanoTokenizer,
    out_path: str,
    dtype: str = "uint16",
) -> int:
    """把若干文本文件 tokenize 并拼接成一个二进制文件。

    Returns:
        总 token 数。
    """
    assert dtype in ("uint16", "uint32")
    np_dtype = np.uint16 if dtype == "uint16" else np.uint32
    max_id = 2**16 - 1 if dtype == "uint16" else 2**32 - 1
    assert tokenizer.vocab_size <= max_id + 1, (
        f"词表 {tokenizer.vocab_size} 超过了 {dtype} 的表达范围，请改用 uint32"
    )

    total = 0
    with open(out_path, "wb") as f:
        for path in text_files:
            print(f"  tokenizing {path} ...")
            with open(path, "r", encoding="utf-8") as g:
                # 按行处理，避免一次性读入大文件
                for line in g:
                    line = line.strip()
                    if not line:
                        continue
                    ids = tokenizer.encode(line, add_bos=False, add_eos=True)
                    arr = np.array(ids, dtype=np_dtype)
                    f.write(arr.tobytes())
                    total += len(ids)
    print(f"  写入 {total:,} 个 token -> {out_path}")
    return total


class PretrainDataset(Dataset):
    """从二进制 token 流中随机采样固定长度片段。

    用 memmap 避免把大文件一次性读进内存。
    """

    def __init__(self, bin_path: str, seq_len: int, dtype: str = "uint16"):
        np_dtype = np.uint16 if dtype == "uint16" else np.uint32
        self.data = np.memmap(bin_path, dtype=np_dtype, mode="r")
        self.seq_len = seq_len
        assert len(self.data) > seq_len + 1, "数据太短"

    def __len__(self):
        # "epoch" 概念在 LLM 预训练里其实没那么重要，这里给一个能让 dataloader 跑起来的数字
        return max(1, len(self.data) // self.seq_len)

    def __getitem__(self, _idx):
        # 随机起点（不放回采样在大语料上等价于按顺序遍历，且更稳）
        i = random.randint(0, len(self.data) - self.seq_len - 1)
        chunk = self.data[i : i + self.seq_len + 1].astype(np.int64)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        return x, y


# ============================================================================
# SFT
# ============================================================================
class SFTDataset(Dataset):
    """加载并 tokenize jsonl 格式的对话数据。

    每行 jsonl 形如:
        {"messages": [
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."},
            ...
        ]}

    生成的 (input_ids, targets) 中:
        - input_ids 是完整对话的 token 序列
        - targets[i] = input_ids[i+1] 但只在 assistant 的回答 token 处计入 loss，
          其他位置写 -100 (cross_entropy 的 ignore_index)
    """

    def __init__(self, jsonl_path: str, tokenizer: NanoTokenizer, max_len: int = 512):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.samples: List[List[dict]] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                self.samples.append(obj["messages"])

    def __len__(self):
        return len(self.samples)

    def _build(self, messages: List[dict]) -> Tuple[List[int], List[int]]:
        """渲染对话并构造 loss_mask。

        关键: 我们逐段构造，知道每段属于哪个 role，从而精确地把 mask 打在 assistant 的回答上。
        """
        tk = self.tokenizer
        ids: List[int] = [tk.bos_id]
        # mask[i] = 1 表示 ids[i] 是 assistant 回答内容的一部分，参与 loss
        mask: List[int] = [0]

        for msg in messages:
            content_ids = tk.encode(msg["content"])
            if msg["role"] == "user":
                ids += [tk.user_id] + content_ids + [tk.end_id]
                mask += [0] * (1 + len(content_ids) + 1)
            elif msg["role"] == "assistant":
                # 起始标记 <assistant> 本身不算 loss；内容与结尾 <end> 算 loss
                # (让模型学会在合适位置输出 <end> 来停止)
                ids += [tk.assistant_id] + content_ids + [tk.end_id]
                mask += [0] + [1] * len(content_ids) + [1]
            else:
                raise ValueError(msg["role"])

        # 截断
        ids = ids[: self.max_len + 1]
        mask = mask[: self.max_len + 1]
        return ids, mask

    def __getitem__(self, idx):
        ids, mask = self._build(self.samples[idx])

        x = ids[:-1]
        y_full = ids[1:]
        m = mask[1:]  # mask 对齐到 target 位置

        # 把 mask=0 的位置改成 -100，使其在 cross_entropy 中被忽略
        y = [t if mi == 1 else -100 for t, mi in zip(y_full, m)]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


def sft_collate(batch, pad_id: int = PAD_ID):
    """SFT 用的 collate: 把不等长样本 pad 到 batch 内最长。"""
    xs, ys = zip(*batch)
    max_len = max(x.size(0) for x in xs)
    bsz = len(xs)
    x_pad = torch.full((bsz, max_len), pad_id, dtype=torch.long)
    y_pad = torch.full((bsz, max_len), -100, dtype=torch.long)  # -100 = ignore
    for i, (x, y) in enumerate(zip(xs, ys)):
        x_pad[i, : x.size(0)] = x
        y_pad[i, : y.size(0)] = y
    return x_pad, y_pad


def make_pretrain_loader(cfg, dtype: str = "uint16"):
    ds = PretrainDataset(cfg.data_path, cfg.seq_len, dtype=dtype)
    return DataLoader(
        ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=0, pin_memory=(cfg.device == "cuda"), drop_last=True,
    )


def make_sft_loader(cfg, tokenizer):
    ds = SFTDataset(cfg.data_path, tokenizer, max_len=cfg.seq_len)
    return DataLoader(
        ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=0, pin_memory=(cfg.device == "cuda"),
        collate_fn=lambda b: sft_collate(b, pad_id=tokenizer.pad_id),
    )

"""
BPE 分词器: 训练 + 加载的薄封装。

底层用 huggingface `tokenizers` 库（纯 Rust，训练飞快），不依赖 transformers。

特殊 token 约定:
    <pad>      [0]   填充
    <bos>      [1]   序列起始
    <eos>      [2]   序列结束 / 生成停止
    <unk>      [3]   未登录词
    <user>     [4]   SFT: 用户回合起始
    <assistant>[5]   SFT: 助手回合起始
    <end>      [6]   SFT: 单回合结束
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Iterable

from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers


SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>", "<user>", "<assistant>", "<end>"]
PAD_ID, BOS_ID, EOS_ID, UNK_ID, USER_ID, ASSISTANT_ID, END_ID = range(len(SPECIAL_TOKENS))


class NanoTokenizer:
    """对 huggingface Tokenizer 的薄封装，给出 encode/decode 与 chat template。"""

    def __init__(self, tk: Tokenizer):
        self.tk = tk
        self.pad_id = PAD_ID
        self.bos_id = BOS_ID
        self.eos_id = EOS_ID
        self.unk_id = UNK_ID
        self.user_id = USER_ID
        self.assistant_id = ASSISTANT_ID
        self.end_id = END_ID

    @property
    def vocab_size(self) -> int:
        return self.tk.get_vocab_size()

    # —— 编码 / 解码 ——
    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        ids = self.tk.encode(text).ids
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        return self.tk.decode(ids, skip_special_tokens=skip_special)

    # —— 对话模板 ——
    def apply_chat_template(self, messages: List[dict], add_generation_prompt: bool = False) -> List[int]:
        """把 [{"role": "user"/"assistant", "content": "..."}] 渲染成 token 序列。

        模板形式 (简化版，借鉴 ChatML)：
            <bos> <user>   你好 <end>
                  <assistant> 你也好 <end>
                  <user>   再见 <end>
                  <assistant>            <-- add_generation_prompt=True 时停在这里
        """
        ids: List[int] = [self.bos_id]
        for msg in messages:
            role = msg["role"]
            content_ids = self.encode(msg["content"])
            if role == "user":
                ids += [self.user_id] + content_ids + [self.end_id]
            elif role == "assistant":
                ids += [self.assistant_id] + content_ids + [self.end_id]
            else:
                raise ValueError(f"未知 role: {role}")
        if add_generation_prompt:
            ids += [self.assistant_id]
        return ids

    # —— 持久化 ——
    def save(self, path: str | Path) -> None:
        self.tk.save(str(path))

    @classmethod
    def load(cls, path: str | Path) -> "NanoTokenizer":
        return cls(Tokenizer.from_file(str(path)))

    # —— 训练 ——
    @classmethod
    def train_from_files(
        cls,
        files: Iterable[str],
        vocab_size: int = 6400,
        min_frequency: int = 2,
    ) -> "NanoTokenizer":
        """在给定的文本文件上训练一个 BPE 分词器。

        Args:
            files:        训练语料文件路径列表 (utf-8 文本，每行一个样本)
            vocab_size:   目标词表大小（含特殊 token）
            min_frequency: 合并对最少出现次数
        """
        tk = Tokenizer(models.BPE(unk_token="<unk>"))
        # ByteLevel 预处理: 让任意 unicode 都能被表示（中文/emoji/代码符号都 OK），
        # 且与 GPT-2 / Llama BPE 思路一致
        tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tk.decoder = decoders.ByteLevel()

        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=SPECIAL_TOKENS,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=True,
        )
        tk.train(list(files), trainer=trainer)
        return cls(tk)

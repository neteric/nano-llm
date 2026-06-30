"""
模型与训练配置。

所有可调超参集中在此处，便于在不同尺度（CPU 玩具 / 单卡 GPU / 多卡）之间切换。
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """Llama 风格 decoder-only Transformer 的结构超参。"""

    # —— 词表 ——
    vocab_size: int = 6400          # 词表大小，需与 tokenizer.json 对齐

    # —— 主干 ——
    d_model: int = 512              # hidden size / embedding 维度
    n_layers: int = 8               # Transformer block 层数
    n_heads: int = 8                # 多头注意力的 query 头数
    n_kv_heads: int = 2             # K/V 头数 (GQA: n_kv_heads < n_heads)
    d_ff: int = 1408                # FFN 中间层维度；SwiGLU 经验值 ≈ 2.75 * d_model
    max_seq_len: int = 512          # 训练/推理支持的最大上下文

    # —— 归一化与位置编码 ——
    rms_norm_eps: float = 1e-5      # RMSNorm 的 epsilon
    rope_theta: float = 10000.0     # RoPE 的频率底数；长上下文模型常调高到 1e6

    # —— 训练 trick ——
    dropout: float = 0.0            # 推理时务必置 0
    tie_word_embeddings: bool = True  # embedding 与 lm_head 共享权重，能显著降低参数量

    @property
    def head_dim(self) -> int:
        assert self.d_model % self.n_heads == 0, "d_model 必须能被 n_heads 整除"
        return self.d_model // self.n_heads

    def __post_init__(self):
        assert self.n_heads % self.n_kv_heads == 0, (
            f"n_heads({self.n_heads}) 必须能被 n_kv_heads({self.n_kv_heads}) 整除"
        )


@dataclass
class TrainConfig:
    """训练超参。"""

    # —— 数据 ——
    data_path: str = "data/pretrain.bin"     # 预 tokenize 后的二进制语料
    val_data_path: Optional[str] = None
    seq_len: int = 512                       # 训练样本长度，应 ≤ model.max_seq_len

    # —— 优化器 ——
    batch_size: int = 16                     # micro batch size
    grad_accum_steps: int = 1                # 梯度累积，等效 batch = batch_size * grad_accum
    learning_rate: float = 3e-4
    min_lr: float = 3e-5                     # cosine schedule 的下限
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # —— 训练步数与日程 ——
    max_steps: int = 5000
    warmup_steps: int = 100
    eval_interval: int = 500
    save_interval: int = 1000
    log_interval: int = 10

    # —— 系统 ——
    device: str = "cuda"                     # "cuda" / "cpu" / "mps"
    dtype: str = "bfloat16"                  # "float32" / "bfloat16" / "float16"
    compile_model: bool = False              # torch.compile 在小模型上收益有限
    out_dir: str = "checkpoints"
    seed: int = 1337


@dataclass
class SFTConfig(TrainConfig):
    """SFT 阶段的训练参数。与预训练共用大部分字段，只覆盖几个差异。"""

    data_path: str = "data/sft.jsonl"
    learning_rate: float = 5e-5              # SFT 学习率通常比预训练小一个量级
    max_steps: int = 1000
    warmup_steps: int = 50
    seq_len: int = 512

    pretrain_ckpt: str = "checkpoints/pretrain_final.pt"  # 必须先有预训练 checkpoint

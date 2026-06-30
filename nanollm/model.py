"""
Llama 风格的 decoder-only Transformer 实现。

设计取舍（与现代 LLM 对齐）：
    - RMSNorm 替代 LayerNorm                     -> 少一组参数、更稳定
    - RoPE 旋转位置编码替代 Learned PE             -> 外推性好、无可学习参数
    - SwiGLU FFN 替代 GELU                       -> 表达力更强，是 Llama/Qwen 等的标配
    - GQA (Grouped Query Attention)              -> 推理时显著降低 KV cache 内存
    - Pre-Norm 结构                               -> 训练更稳定
    - KV cache                                    -> 推理 O(n) 而非 O(n²)
    - 权重绑定 (embedding ↔ lm_head)              -> 小模型省参数

阅读顺序建议: RMSNorm -> RoPE -> Attention -> SwiGLU -> Block -> Transformer
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


# ============================================================================
# 1) RMSNorm
# ============================================================================
class RMSNorm(nn.Module):
    """Root Mean Square LayerNorm。

    与 LayerNorm 的区别：
        LayerNorm:  y = (x - mean) / sqrt(var + eps) * gamma + beta
        RMSNorm:    y = x / sqrt(mean(x^2) + eps) * gamma
    省掉了均值中心化和 beta，少 ~50% 计算与一组参数，效果几乎无损。
    """

    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 计算时用 fp32 防止 bf16 下数值不稳
        dtype = x.dtype
        x_fp32 = x.float()
        rms = x_fp32.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x_fp32 * rms).to(dtype) * self.weight


# ============================================================================
# 2) RoPE: Rotary Position Embedding
# ============================================================================
def precompute_rope_cache(
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """预计算 RoPE 用到的 cos / sin 表。

    返回:
        cos: [max_seq_len, head_dim/2]
        sin: [max_seq_len, head_dim/2]
    """
    # 论文里的 theta_i = 1 / (base^(2i/d))，i ∈ [0, d/2)
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(max_seq_len, device=device).float()
    freqs = torch.outer(t, freqs)  # [seq_len, head_dim/2]
    return freqs.cos(), freqs.sin()


def apply_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """对 q 或 k 张量施加旋转。

    x: [B, n_heads, T, head_dim]
    cos/sin: [T, head_dim/2]

    将 head_dim 维拆成 (head_dim/2) 对，每对 (x_even, x_odd) 看成复数 x_even + i*x_odd，
    然后乘以 e^{iθ} = cos + i*sin，等价于 2D 平面上的旋转。
    """
    # 取偶数下标与奇数下标
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    # 广播: cos/sin 形状 [T, head_dim/2] -> [1, 1, T, head_dim/2]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    # 复数乘法的实部 / 虚部
    rot_even = x_even * cos - x_odd * sin
    rot_odd = x_even * sin + x_odd * cos
    # 重新交错回去
    out = torch.empty_like(x)
    out[..., 0::2] = rot_even
    out[..., 1::2] = rot_odd
    return out


# ============================================================================
# 3) Attention with GQA + KV cache
# ============================================================================
class Attention(nn.Module):
    """多头注意力，支持 GQA 与 KV cache。

    GQA: query 有 n_heads 个头，但 key/value 只有 n_kv_heads 个头，每组 query 共享一组 kv。
        当 n_kv_heads = n_heads 时退化为标准 MHA；
        当 n_kv_heads = 1 时退化为 MQA。
    Llama-2 70B、Qwen2、Mistral 等都用 GQA，主要动机是省 KV cache 显存。
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.n_rep = cfg.n_heads // cfg.n_kv_heads  # 每个 kv 头被几个 q 头共享

        # 一次性产出 q/k/v，q 维度 = n_heads * head_dim, k/v 维度 = n_kv_heads * head_dim
        self.wq = nn.Linear(cfg.d_model, cfg.n_heads * cfg.head_dim, bias=False)
        self.wk = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.wv = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.wo = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.d_model, bias=False)

        self.dropout = cfg.dropout
        # 是否启用 PyTorch SDPA (Flash-Attention 兼容路径)
        self.use_sdpa = hasattr(F, "scaled_dot_product_attention")

    def forward(
        self,
        x: torch.Tensor,                                  # [B, T, d_model]
        cos: torch.Tensor,                                # [T_pos, head_dim/2]
        sin: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, T, _ = x.shape

        # 1) 线性映射
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)       # [B, n_h, T, d_h]
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)    # [B, n_kvh, T, d_h]
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # 2) 施加 RoPE 到 q, k（注意 v 不需要旋转）
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # 3) 拼接 KV cache（推理时增量生成）
        if kv_cache is not None:
            past_k, past_v = kv_cache
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        new_cache = (k, v)

        # 4) GQA: 把 kv 头复制 n_rep 份以匹配 q 头数
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        # 5) 计算注意力
        # 训练时 q.shape[2] == k.shape[2]，需要 causal mask；
        # 推理增量解码时 q.shape[2] == 1，理论上不需要 mask（只有它一个 token），
        # 但 SDPA 的 is_causal 仍可放心置 True，对 T_q=1 的情况是 no-op。
        is_causal = q.shape[2] > 1
        if self.use_sdpa:
            # PyTorch SDPA 内部会选择 Flash-Attn / Memory-Efficient / Math 三种 kernel
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=is_causal,
            )
        else:
            # Fallback: 手动实现，便于阅读
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if is_causal:
                T_q, T_k = q.shape[2], k.shape[2]
                # 下三角 mask: 第 i 行只允许看到前 i 个 key
                mask = torch.ones(T_q, T_k, dtype=torch.bool, device=q.device).tril(diagonal=T_k - T_q)
                scores = scores.masked_fill(~mask, float("-inf"))
            attn = F.softmax(scores.float(), dim=-1).type_as(q)
            if self.training and self.dropout > 0:
                attn = F.dropout(attn, p=self.dropout)
            out = torch.matmul(attn, v)

        # 6) 合并头并输出投影
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(out), new_cache


# ============================================================================
# 4) SwiGLU FFN
# ============================================================================
class SwiGLU(nn.Module):
    """SwiGLU(x) = (Swish(W1 x)) ⊙ (W3 x), 然后 W2(·)

    比 ReLU/GELU FFN 多了一路 "gate"，3 个矩阵代替原来的 2 个，
    但中间维度通常会按 2/3 缩放以保持总参数不变。
    """

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)  # gate
        self.w3 = nn.Linear(d_model, d_ff, bias=False)  # up
        self.w2 = nn.Linear(d_ff, d_model, bias=False)  # down

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


# ============================================================================
# 5) Transformer Block
# ============================================================================
class Block(nn.Module):
    """Pre-Norm 风格的 Transformer Block。

        x -> RMSNorm -> Attn -> + (residual) -> RMSNorm -> FFN -> + (residual)
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.attn = Attention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.ffn = SwiGLU(cfg.d_model, cfg.d_ff)

    def forward(self, x, cos, sin, kv_cache=None):
        attn_out, new_cache = self.attn(self.attn_norm(x), cos, sin, kv_cache)
        x = x + attn_out
        x = x + self.ffn(self.ffn_norm(x))
        return x, new_cache


# ============================================================================
# 6) Full Transformer
# ============================================================================
@dataclass
class ModelOutput:
    logits: torch.Tensor                          # [B, T, vocab_size]
    loss: Optional[torch.Tensor] = None           # 标量
    kv_caches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None


class NanoLLM(nn.Module):
    """完整模型: token embedding -> N x Block -> RMSNorm -> lm_head"""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.tok_emb.weight

        # RoPE 表只算一次，作为 buffer 跟随 .to(device) 走
        cos, sin = precompute_rope_cache(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        # 初始化
        self.apply(self._init_weights)
        # 残差路径上的输出投影做一次额外缩小，GPT-2 经验
        for name, p in self.named_parameters():
            if name.endswith("wo.weight") or name.endswith("w2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layers))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_parameters(self, exclude_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if exclude_embedding:
            n -= self.tok_emb.weight.numel()
            if not self.cfg.tie_word_embeddings:
                n -= self.lm_head.weight.numel()
        return n

    def forward(
        self,
        input_ids: torch.Tensor,                                  # [B, T]
        targets: Optional[torch.Tensor] = None,                   # [B, T]，-100 为忽略
        kv_caches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        return_caches: bool = False,
    ) -> ModelOutput:
        B, T = input_ids.shape

        # 推理增量解码: 当前 token 在序列中的真实位置 = 已缓存长度
        if kv_caches is not None and kv_caches[0] is not None:
            past_len = kv_caches[0][0].shape[2]
        else:
            past_len = 0
        assert past_len + T <= self.cfg.max_seq_len, (
            f"超过 max_seq_len: past_len={past_len}, new={T}, max={self.cfg.max_seq_len}"
        )

        cos = self.rope_cos[past_len : past_len + T]
        sin = self.rope_sin[past_len : past_len + T]

        x = self.tok_emb(input_ids)
        new_caches: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for i, block in enumerate(self.blocks):
            past = kv_caches[i] if kv_caches is not None else None
            x, new_cache = block(x, cos, sin, past)
            new_caches.append(new_cache)
        x = self.final_norm(x)

        if targets is not None:
            # 训练: 计算所有 token 的 logits 后做交叉熵
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, self.cfg.vocab_size),
                targets.view(-1),
                ignore_index=-100,
            )
        else:
            # 推理: 增量解码时只需算最后一个位置
            logits = self.lm_head(x[:, -1:, :]) if not return_caches or T == 1 else self.lm_head(x)
            loss = None

        return ModelOutput(
            logits=logits,
            loss=loss,
            kv_caches=new_caches if return_caches else None,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,                       # [B, T_prompt]
        max_new_tokens: int = 200,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """带 KV cache 的自回归生成。

        采样策略:
            temperature = 0: 贪心
            top_k:           只在 top-k 之内归一化采样
            top_p:           nucleus / top-p 采样
            两者可以同时使用，先 top_k 再 top_p。
        """
        self.eval()
        device = input_ids.device

        # 1) Prefill: 一次性把 prompt 喂进去，得到初始 cache
        out = self.forward(input_ids, return_caches=True)
        kv_caches = out.kv_caches
        # 取最后一个位置的 logits
        next_logits = out.logits[:, -1, :]

        generated = [input_ids]
        for _ in range(max_new_tokens):
            # 2) 采样下一个 token
            if temperature == 0.0:
                next_token = next_logits.argmax(dim=-1, keepdim=True)
            else:
                logits = next_logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float("-inf")
                if top_p is not None:
                    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                    cum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    mask = cum > top_p
                    mask[..., 1:] = mask[..., :-1].clone()
                    mask[..., 0] = False
                    sorted_logits[mask] = float("-inf")
                    logits = torch.empty_like(logits).scatter_(1, sorted_idx, sorted_logits)
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            generated.append(next_token)
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

            # 3) Decode: 只喂最新一个 token，复用 KV cache
            out = self.forward(next_token, kv_caches=kv_caches, return_caches=True)
            kv_caches = out.kv_caches
            next_logits = out.logits[:, -1, :]

        return torch.cat(generated, dim=1)

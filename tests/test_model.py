"""模型前向、反向、生成的烟雾测试。

直接运行:
    python -m tests.test_model
"""
import torch

from nanollm.config import ModelConfig
from nanollm.model import NanoLLM, precompute_rope_cache, apply_rope


def test_rope_shapes():
    cos, sin = precompute_rope_cache(head_dim=64, max_seq_len=128)
    assert cos.shape == (128, 32)
    assert sin.shape == (128, 32)
    x = torch.randn(2, 4, 16, 64)
    y = apply_rope(x, cos[:16], sin[:16])
    assert y.shape == x.shape
    # 验证 RoPE 是正交变换：施加后 L2 范数不变
    assert torch.allclose(y.norm(dim=-1), x.norm(dim=-1), atol=1e-5)
    print("[ok] RoPE shapes & norm-preservation")


def test_forward_backward():
    cfg = ModelConfig(
        vocab_size=256, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2,
        d_ff=128, max_seq_len=64,
    )
    model = NanoLLM(cfg)
    x = torch.randint(0, 256, (3, 16))
    y = torch.randint(0, 256, (3, 16))
    out = model(x, targets=y)
    assert out.logits.shape == (3, 16, 256)
    assert out.loss.dim() == 0
    out.loss.backward()
    grad_ok = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert grad_ok
    print(f"[ok] forward+backward, loss = {out.loss.item():.4f}")


def test_generate_with_kv_cache():
    cfg = ModelConfig(
        vocab_size=256, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2,
        d_ff=128, max_seq_len=64,
    )
    model = NanoLLM(cfg)
    model.eval()
    prompt = torch.randint(0, 256, (1, 5))
    out = model.generate(prompt, max_new_tokens=10, temperature=0.0)
    assert out.shape == (1, 15)
    print(f"[ok] generate, output shape = {tuple(out.shape)}")


def test_kv_cache_correctness():
    """关键正确性测试: 一次前向 vs 增量前向应得到相同的最终 hidden。"""
    cfg = ModelConfig(
        vocab_size=128, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2,
        d_ff=128, max_seq_len=64,
    )
    model = NanoLLM(cfg).eval()
    torch.manual_seed(0)
    x = torch.randint(0, 128, (1, 8))

    # 路径 A: 一次性前向，拿到每个位置的 logits
    with torch.no_grad():
        logits_a = model.lm_head(_full_hidden(model, x))

    # 路径 B: 增量前向，每次喂一个 token
    kv = None
    last_logits = None
    with torch.no_grad():
        for i in range(x.shape[1]):
            out = model(x[:, i:i+1], kv_caches=kv, return_caches=True)
            kv = out.kv_caches
            last_logits = out.logits[:, -1, :]

    # 取一次性前向最后一个位置的 logits（注意一次性前向时只返回最后位置，除非 targets 不为 None）
    # 为了对比，重新算完整 logits
    diff = (logits_a[:, -1, :] - last_logits).abs().max().item()
    assert diff < 1e-4, f"KV cache 实现有 bug, diff = {diff}"
    print(f"[ok] KV cache correctness, max diff = {diff:.2e}")


def _full_hidden(model, x):
    """跑一遍 forward 但返回 final_norm 之前的 hidden。"""
    past_len = 0
    cos = model.rope_cos[past_len : past_len + x.shape[1]]
    sin = model.rope_sin[past_len : past_len + x.shape[1]]
    h = model.tok_emb(x)
    for block in model.blocks:
        h, _ = block(h, cos, sin, None)
    return model.final_norm(h)


def test_param_count():
    cfg = ModelConfig()  # 默认配置
    model = NanoLLM(cfg)
    n = model.num_parameters()
    print(f"[info] 默认配置参数量 = {n/1e6:.1f}M")


if __name__ == "__main__":
    test_rope_shapes()
    test_forward_backward()
    test_generate_with_kv_cache()
    test_kv_cache_correctness()
    test_param_count()
    print("\nAll tests passed.")

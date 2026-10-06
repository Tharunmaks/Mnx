import pytest
import torch

from mnx.configs import PRESETS, get_config
from mnx.model import MNXCoder, MNXConfig, RMSNorm, apply_rope, precompute_rope


@pytest.fixture
def small_cfg() -> MNXConfig:
    return MNXConfig(vocab_size=300, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, d_ff=64, max_seq_len=64)


@pytest.fixture
def model(small_cfg) -> MNXCoder:
    return MNXCoder(small_cfg).eval()


def test_forward_shapes_and_loss(model, small_cfg):
    x = torch.randint(0, small_cfg.vocab_size, (3, 17))
    logits, loss, cache = model(x, targets=x)
    assert logits.shape == (3, 17, small_cfg.vocab_size)
    assert loss is not None and loss.ndim == 0 and torch.isfinite(loss)
    assert cache is None
    # Untrained model: loss should be near ln(vocab).
    assert abs(loss.item() - torch.log(torch.tensor(float(small_cfg.vocab_size))).item()) < 1.0


def test_causality(model, small_cfg):
    """Changing a future token must not change logits at earlier positions."""
    torch.manual_seed(1)
    x = torch.randint(0, small_cfg.vocab_size, (1, 20))
    y = x.clone()
    y[0, 12] = (y[0, 12] + 1) % small_cfg.vocab_size
    with torch.no_grad():
        lx, _, _ = model(x)
        ly, _, _ = model(y)
    assert torch.allclose(lx[0, :12], ly[0, :12], atol=1e-5)
    assert not torch.allclose(lx[0, 12:], ly[0, 12:], atol=1e-5)


def test_kv_cache_matches_full_forward(model, small_cfg):
    x = torch.randint(0, small_cfg.vocab_size, (2, 24))
    with torch.no_grad():
        full, _, _ = model(x)
        l1, _, kv = model(x[:, :10], use_cache=True)
        l2, _, kv = model(x[:, 10:20], past_kv=kv, use_cache=True)
        l3, _, kv = model(x[:, 20:], past_kv=kv, use_cache=True)
    assert len(kv) == small_cfg.n_layers
    assert kv[0][0].shape == (2, small_cfg.n_kv_heads, 24, small_cfg.head_dim)
    assert torch.allclose(torch.cat([l1, l2, l3], dim=1), full, atol=1e-4)


def test_generate_respects_max_new_tokens(model, small_cfg):
    prompt = torch.randint(0, small_cfg.vocab_size, (2, 5))
    out = model.generate(prompt, max_new_tokens=12, temperature=1.0, top_k=10)
    assert out.shape == (2, 17)
    assert torch.equal(out[:, :5], prompt)
    assert out.min() >= 0 and out.max() < small_cfg.vocab_size


def test_generate_greedy_is_deterministic(model, small_cfg):
    prompt = torch.randint(0, small_cfg.vocab_size, (1, 4))
    a = model.generate(prompt, max_new_tokens=8, temperature=0.0)
    b = model.generate(prompt, max_new_tokens=8, temperature=0.0)
    assert torch.equal(a, b)


def test_generate_stops_at_eos(model, small_cfg):
    """Use the greedy first token as eos: generation must stop right after it."""
    prompt = torch.randint(0, small_cfg.vocab_size, (1, 3))
    first = model.generate(prompt, max_new_tokens=1, temperature=0.0)[0, -1].item()
    out = model.generate(prompt, max_new_tokens=20, temperature=0.0, eos_id=first)
    assert out.shape[1] == 4  # stopped right after the first eos
    assert out[0, -1].item() == first


def test_generate_pads_finished_rows_in_batch(model, small_cfg):
    torch.manual_seed(2)
    prompt = torch.randint(0, small_cfg.vocab_size, (2, 3))
    greedy = model.generate(prompt, max_new_tokens=6, temperature=0.0)
    # Pick the second generated token of row 0 as eos; row 1 may run longer.
    eos = greedy[0, 4].item()
    out = model.generate(prompt, max_new_tokens=6, temperature=0.0, eos_id=eos)
    assert out[0, 4].item() == eos
    assert (out[0, 5:] == eos).all()  # finished row is padded with eos
    assert out.shape[1] <= 9


def test_generate_sliding_window_beyond_max_seq_len(small_cfg):
    cfg = small_cfg.replace(max_seq_len=16)
    model = MNXCoder(cfg).eval()
    prompt = torch.randint(0, cfg.vocab_size, (1, 10))
    out = model.generate(prompt, max_new_tokens=30, temperature=0.0)
    assert out.shape == (1, 40)


def test_sampling_top_p_and_top_k_run(model, small_cfg):
    prompt = torch.randint(0, small_cfg.vocab_size, (1, 3))
    out = model.generate(prompt, max_new_tokens=5, temperature=0.7, top_k=5, top_p=0.9)
    assert out.shape == (1, 8)


def test_tied_embeddings_and_param_count(model):
    assert model.lm_head.weight is model.tok_emb.weight
    total = sum(p.numel() for p in model.parameters())  # tied params counted once by torch too
    assert model.num_parameters() == total
    assert model.num_parameters(non_embedding=True) == total - model.tok_emb.weight.numel()


def test_rmsnorm():
    norm = RMSNorm(8)
    x = torch.randn(2, 3, 8) * 5
    y = norm(x)
    rms = y.pow(2).mean(-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-4)


def test_rope_preserves_norm_and_is_relative():
    cos, sin = precompute_rope(8, 32)
    q = torch.randn(1, 1, 32, 8)
    k = torch.randn(1, 1, 32, 8)
    rq, rk = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
    assert torch.allclose(rq.norm(dim=-1), q.norm(dim=-1), atol=1e-5)
    # Dot product depends only on relative offset: <q_m, k_n> == <q_{m+s}, k_{n+s}>.
    q0, k0 = q[..., 3, :], k[..., 5, :]
    shifted_q = apply_rope(q0.expand(1, 1, 32, 8).contiguous(), cos, sin)
    shifted_k = apply_rope(k0.expand(1, 1, 32, 8).contiguous(), cos, sin)
    d1 = (shifted_q[..., 3, :] * shifted_k[..., 5, :]).sum()
    d2 = (shifted_q[..., 10, :] * shifted_k[..., 12, :]).sum()
    assert torch.allclose(d1, d2, atol=1e-4)


def test_config_validation():
    with pytest.raises(ValueError):
        MNXConfig(d_model=30, n_heads=4)
    with pytest.raises(ValueError):
        MNXConfig(n_heads=4, n_kv_heads=3)
    cfg = MNXConfig()
    assert MNXConfig.from_dict(cfg.to_dict()) == cfg


def test_presets_construct_and_tiny_is_about_1m():
    for name in ("tiny", "small", "base", "max"):
        cfg = PRESETS[name]
        assert cfg.d_model % cfg.n_heads == 0
    tiny = MNXCoder(get_config("tiny"))
    assert 0.8e6 < tiny.num_parameters() < 1.3e6
    with pytest.raises(KeyError):
        get_config("nope")
    assert get_config("tiny", n_layers=1).n_layers == 1

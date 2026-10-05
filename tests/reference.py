"""An independent float64 numpy transformer (Qwen2 / Llama shapes), forward and backward, written without hive.lpu.

The LPU tests compare the simulated chips against this; this in turn is checked against finite differences.
"""

from __future__ import annotations

import math

import numpy as np


def rms(x, w, eps):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * w


def rope_tables(seq: int, head_dim: int, theta: float = 10000.0):
    inv = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim))
    t = np.arange(seq, dtype=np.float64)[:, None] * inv[None, :]
    emb = np.concatenate([t, t], -1)
    return np.cos(emb), np.sin(emb)


def _rope(t, cos, sin, D):
    S = t.shape[0]
    t = t.reshape(S, -1, D)
    h = D // 2
    rot = np.concatenate([-t[..., h:], t[..., :h]], -1)
    return (t * cos[:, None] + rot * sin[:, None]).reshape(S, -1)


def _rope_bwd(dy, cos, sin, D):
    S = dy.shape[0]
    t = dy.reshape(S, -1, D)
    h = D // 2
    u = t * sin[:, None]
    return (t * cos[:, None] + np.concatenate([u[..., h:], -u[..., :h]], -1)).reshape(S, -1)


def layer_forward(x, w, cfg, cos, sin):
    """One decoder layer. w holds full matrices (biases and norms as 1-D). Returns the cache the backward pass needs."""
    S = x.shape[0]
    nh, nkv = cfg["num_attention_heads"], cfg.get("num_key_value_heads") or cfg["num_attention_heads"]
    D = cfg["hidden_size"] // nh
    eps = cfg.get("rms_norm_eps", 1e-6)
    bias = lambda k: w.get(k, 0.0)
    c = {"x": x}
    c["xn"] = rms(x, w["input_layernorm.weight"], eps)
    q = c["xn"] @ w["self_attn.q_proj.weight"].T + bias("self_attn.q_proj.bias")
    k = c["xn"] @ w["self_attn.k_proj.weight"].T + bias("self_attn.k_proj.bias")
    c["v"] = c["xn"] @ w["self_attn.v_proj.weight"].T + bias("self_attn.v_proj.bias")
    c["qr"], c["kr"] = _rope(q, cos, sin, D), _rope(k, cos, sin, D)
    mask = np.triu(np.ones((S, S), bool), 1)
    attn = np.zeros((S, nh * D))
    c["probs"] = []
    for h in range(nh):
        g = h // (nh // nkv)
        s = c["qr"][:, h * D:(h + 1) * D] @ c["kr"][:, g * D:(g + 1) * D].T / math.sqrt(D)
        s = np.where(mask, -1e30, s)
        p = np.exp(s - s.max(-1, keepdims=True))
        p /= p.sum(-1, keepdims=True)
        c["probs"].append(p)
        attn[:, h * D:(h + 1) * D] = p @ c["v"][:, g * D:(g + 1) * D]
    c["attn"] = attn
    c["h1"] = x + attn @ w["self_attn.o_proj.weight"].T
    c["hn"] = rms(c["h1"], w["post_attention_layernorm.weight"], eps)
    c["gate"] = c["hn"] @ w["mlp.gate_proj.weight"].T
    c["up"] = c["hn"] @ w["mlp.up_proj.weight"].T
    c["act"] = c["gate"] / (1 + np.exp(-c["gate"])) * c["up"]
    c["out"] = c["h1"] + c["act"] @ w["mlp.down_proj.weight"].T
    return c


def _rms_bwd(dy, x, w, eps):
    r = 1 / np.sqrt((x * x).mean(-1, keepdims=True) + eps)
    g = dy * w
    return r * (g - x * r * r * (g * x).mean(-1, keepdims=True)), (dy * x * r).sum(0)


def layer_backward(c, w, cfg, cos, sin, dout):
    """Gradients of one layer's weights and of its input, given d(loss)/d(out)."""
    S = dout.shape[0]
    nh, nkv = cfg["num_attention_heads"], cfg.get("num_key_value_heads") or cfg["num_attention_heads"]
    D = cfg["hidden_size"] // nh
    eps = cfg.get("rms_norm_eps", 1e-6)
    gr = {}
    d_act = dout @ w["mlp.down_proj.weight"]
    gr["mlp.down_proj.weight"] = dout.T @ c["act"]
    sg = 1 / (1 + np.exp(-c["gate"]))
    d_gate = d_act * c["up"] * (sg + c["gate"] * sg * (1 - sg))
    d_up = d_act * c["gate"] * sg
    gr["mlp.gate_proj.weight"] = d_gate.T @ c["hn"]
    gr["mlp.up_proj.weight"] = d_up.T @ c["hn"]
    dx, gr["post_attention_layernorm.weight"] = _rms_bwd(d_gate @ w["mlp.gate_proj.weight"] + d_up @ w["mlp.up_proj.weight"],
                                                         c["h1"], w["post_attention_layernorm.weight"], eps)
    d_h1 = dout + dx
    d_attn = d_h1 @ w["self_attn.o_proj.weight"]
    gr["self_attn.o_proj.weight"] = d_h1.T @ c["attn"]
    d_qr, d_kr, d_v = np.zeros((S, nh * D)), np.zeros((S, nkv * D)), np.zeros((S, nkv * D))
    for h in range(nh):
        g = h // (nh // nkv)
        p, da = c["probs"][h], d_attn[:, h * D:(h + 1) * D]
        d_p = da @ c["v"][:, g * D:(g + 1) * D].T
        d_v[:, g * D:(g + 1) * D] += p.T @ da
        d_s = p * (d_p - (d_p * p).sum(-1, keepdims=True)) / math.sqrt(D)
        d_qr[:, h * D:(h + 1) * D] = d_s @ c["kr"][:, g * D:(g + 1) * D]
        d_kr[:, g * D:(g + 1) * D] += d_s.T @ c["qr"][:, h * D:(h + 1) * D]
    d_q, d_k = _rope_bwd(d_qr, cos, sin, D), _rope_bwd(d_kr, cos, sin, D)
    d_xn = d_q @ w["self_attn.q_proj.weight"] + d_k @ w["self_attn.k_proj.weight"] + d_v @ w["self_attn.v_proj.weight"]
    for d_, key in ((d_q, "self_attn.q_proj"), (d_k, "self_attn.k_proj"), (d_v, "self_attn.v_proj")):
        gr[key + ".weight"] = d_.T @ c["xn"]
        if key + ".bias" in w:
            gr[key + ".bias"] = d_.sum(0)
    dx, gr["input_layernorm.weight"] = _rms_bwd(d_xn, c["x"], w["input_layernorm.weight"], eps)
    return d_h1 + dx, gr


def logits(layers, E, norm, head, ids, cfg, cos, sin):
    x = E[np.asarray(ids)]
    for w in layers:
        x = layer_forward(x, w, cfg, cos, sin)["out"]
    return rms(x, norm, cfg.get("rms_norm_eps", 1e-6)) @ head.T


def loss(layers, E, norm, head, ids, targets, cfg, cos, sin):
    z = logits(layers, E, norm, head, ids, cfg, cos, sin)
    z = z - z.max(-1, keepdims=True)
    lp = z - np.log(np.exp(z).sum(-1, keepdims=True))
    return -lp[np.arange(len(targets)), np.asarray(targets)].mean()


def sgd_step(layers, E, norm, ids, targets, cfg, cos, sin, lr, head=None):
    """One plain SGD step; head None means tied to E. Returns (loss, new layers, new E, new norm, new head)."""
    eps = cfg.get("rms_norm_eps", 1e-6)
    tied = head is None
    head = E if tied else head
    S = len(ids)
    x = E[np.asarray(ids)]
    caches = []
    for w in layers:
        c = layer_forward(x, w, cfg, cos, sin)
        caches.append(c)
        x = c["out"]
    xn = rms(x, norm, eps)
    z = xn @ head.T
    p = np.exp(z - z.max(-1, keepdims=True))
    p /= p.sum(-1, keepdims=True)
    value = -np.log(p[np.arange(S), np.asarray(targets)]).mean()
    onehot = np.zeros_like(z)
    onehot[np.arange(S), np.asarray(targets)] = 1
    d_z = (p - onehot) / S
    d_head = d_z.T @ xn
    dx, d_norm = _rms_bwd(d_z @ head, x, norm, eps)
    new_layers = []
    for w, c in zip(reversed(layers), reversed(caches)):
        dx, gr = layer_backward(c, w, cfg, cos, sin, dx)
        new_layers.append({k: w[k] - lr * gr[k] for k in w})
    new_layers.reverse()
    E2 = E.copy()
    np.add.at(E2, np.asarray(ids), -lr * dx)
    if tied:
        E2 -= lr * d_head
        return value, new_layers, E2, norm - lr * d_norm, E2
    return value, new_layers, E2, norm - lr * d_norm, head - lr * d_head


def random_model(cfg: dict, seed: int = 0):
    """Random float64 weights in the layout reference.py uses (1-D norms and biases)."""
    rng = np.random.default_rng(seed)
    H, I = cfg["hidden_size"], cfg["intermediate_size"]
    nh, nkv = cfg["num_attention_heads"], cfg.get("num_key_value_heads") or cfg["num_attention_heads"]
    D = H // nh
    bias = cfg.get("attention_bias", cfg.get("model_type") == "qwen2")
    layers = []
    for _ in range(cfg["num_hidden_layers"]):
        w = {"input_layernorm.weight": 1 + 0.1 * rng.standard_normal(H), "post_attention_layernorm.weight": 1 + 0.1 * rng.standard_normal(H),
             "self_attn.q_proj.weight": 0.15 * rng.standard_normal((H, H)), "self_attn.k_proj.weight": 0.15 * rng.standard_normal((nkv * D, H)),
             "self_attn.v_proj.weight": 0.15 * rng.standard_normal((nkv * D, H)), "self_attn.o_proj.weight": 0.15 * rng.standard_normal((H, H)),
             "mlp.gate_proj.weight": 0.15 * rng.standard_normal((I, H)), "mlp.up_proj.weight": 0.15 * rng.standard_normal((I, H)),
             "mlp.down_proj.weight": 0.15 * rng.standard_normal((H, I))}
        if bias:
            w.update({"self_attn.q_proj.bias": 0.5 * rng.standard_normal(H), "self_attn.k_proj.bias": 0.5 * rng.standard_normal(nkv * D),
                      "self_attn.v_proj.bias": 0.5 * rng.standard_normal(nkv * D)})
        layers.append(w)
    E = 0.3 * rng.standard_normal((cfg["vocab_size"], H))
    norm = 1 + 0.1 * rng.standard_normal(H)
    head = None if cfg.get("tie_word_embeddings", True) else 0.3 * rng.standard_normal((cfg["vocab_size"], H))
    return layers, E, norm, head

"""Architect Bee: turn "a 50M parameter model" into a real transformer design,
and say honestly what it takes to train it (memory, GPUs, time, cost).

Shapes follow the Llama / Qwen2 family (RMSNorm, SwiGLU, rotary attention, GQA on big
models), so the result loads with plain `transformers`. The maths is also used by the
from-scratch training recipe, which gets a copy of this file.
"""

from __future__ import annotations

import math
import re

# Parameter counts people usually mean, shown as choices. Anything in between works too.
SIZES = [
    ("1M", 1e6), ("5M", 5e6), ("10M", 1e7), ("25M", 2.5e7), ("50M", 5e7), ("125M", 1.25e8),
    ("350M", 3.5e8), ("1B", 1e9), ("3B", 3e9), ("7B", 7e9), ("13B", 1.3e10), ("30B", 3e10),
    ("70B", 7e10), ("175B", 1.75e11), ("300B", 3e11), ("500B", 5e11),
]
MAX_PARAMS = 5e11  # the biggest model the Hive will design and train

# Depth anchors (params → layers), log-interpolated. Based on published model shapes.
_LAYER_ANCHORS = [
    (1e6, 4), (1e7, 6), (1e8, 12), (3.5e8, 24), (1.3e9, 24), (2.7e9, 32), (7e9, 32),
    (1.3e10, 40), (3e10, 60), (7e10, 80), (1.75e11, 96), (4e11, 126), (1e12, 128), (2e12, 160),
]

# GPUs you can rent: memory GB, effective training speed (FLOP/s at ~35% utilisation), rental $/hour.
GPUS = {
    "H100 80GB": (80, 350e12, 2.5),
    "A100 80GB": (80, 110e12, 1.6),
    "A100 40GB": (40, 110e12, 1.2),
    "L40S 48GB": (48, 120e12, 1.0),
    "RTX 4090 24GB": (24, 60e12, 0.45),
    "RTX 3090 24GB": (24, 30e12, 0.3),
    "L4 24GB": (24, 40e12, 0.5),
    "T4 16GB": (16, 12e12, 0.2),
}
CPU_FLOPS_PER_CORE = 2e10  # what PyTorch fp32 training typically gets per core
TRAIN_BYTES_PER_PARAM_GPU = 18  # bf16 weights + grads + fp32 Adam + master weights, before activations
TRAIN_BYTES_PER_PARAM_CPU = 16
TOKENS_PER_PARAM = 20  # Chinchilla: a well-trained model sees ~20 tokens per parameter

_SIZE_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*(k|m|b|t|thousand|million|billion|trillion)\b", re.I)
_UNIT = {"k": 1e3, "thousand": 1e3, "m": 1e6, "million": 1e6, "b": 1e9, "billion": 1e9, "t": 1e12, "trillion": 1e12}


def parse_size(text: str) -> float | None:
    """'50M' → 5e7, '2 trillion' → 2e12, '1.5b params' → 1.5e9."""
    m = _SIZE_RE.search(text or "")
    if not m:
        return None
    return float(m.group(1).replace(",", ".")) * _UNIT[m.group(2).lower()]


def clamp_size(n: float) -> tuple[float, bool]:
    """(size to use, was it capped). Sizes above 500B are brought down to 500B."""
    return (MAX_PARAMS, True) if n > MAX_PARAMS else (max(2e5, n), False)


def gpus_needed(n_params: float, gpu_memory_gb: float = 80) -> int:
    """How many GPUs of that memory hold a training run (weights, grads, optimizer; activations extra)."""
    return max(1, math.ceil(TRAIN_BYTES_PER_PARAM_GPU * n_params / (gpu_memory_gb * 1e9 * 0.85)))


def fmt_params(n: float) -> str:
    for unit, div in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if n >= div:
            v = n / div
            return f"{v:.2f}{unit}" if v < 10 else f"{v:.1f}{unit}" if v < 100 else f"{v:.0f}{unit}"
    return str(int(n))


def fmt_bytes(n: float) -> str:
    for unit, div in (("PB", 1e15), ("TB", 1e12), ("GB", 1e9), ("MB", 1e6), ("KB", 1e3)):
        if n >= div:
            return f"{n / div:.1f} {unit}"
    return f"{n:.0f} B"


def fmt_time(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f} seconds"
    if seconds < 5400:
        return f"{seconds / 60:.0f} minutes"
    if seconds < 172800:
        return f"{seconds / 3600:.1f} hours"
    if seconds < 365 * 86400 * 2:
        return f"{seconds / 86400:.0f} days"
    return f"{seconds / (365 * 86400):.0f} years"


def fmt_money(usd: float) -> str:
    if usd < 1:
        return f"${usd:.2f}"
    if usd < 1000:
        return f"${usd:.0f}"
    if usd < 1e6:
        return f"${usd / 1e3:.0f}K"
    return f"${usd / 1e6:.1f}M"


def _interp_layers(n: float) -> int:
    xs = [math.log10(a[0]) for a in _LAYER_ANCHORS]
    ys = [a[1] for a in _LAYER_ANCHORS]
    x = math.log10(max(n, 1e5))
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    for i in range(1, len(xs)):
        if x <= xs[i]:
            t = (x - xs[i - 1]) / (xs[i] - xs[i - 1])
            return int(round(ys[i - 1] + t * (ys[i] - ys[i - 1])))
    return ys[-1]


def default_vocab(n: float) -> int:
    """A vocabulary that doesn't swamp a small model's parameter budget."""
    if n <= 2e6:
        return 4096
    if n <= 2e7:
        return 8192
    if n <= 1.5e8:
        return 16384
    if n <= 3e9:
        return 32000
    if n <= 5e10:
        return 65536
    return 131072


def count_params(d: int, layers: int, heads: int, kv_heads: int, d_ff: int, vocab: int, tied: bool) -> int:
    head_dim = d // heads
    kv_dim = kv_heads * head_dim
    attn = d * d * 2 + d * kv_dim * 2 + d + kv_dim * 2  # q, o, k, v (+ Qwen-style biases on q/k/v)
    mlp = 3 * d * d_ff
    norms = 2 * d
    per_layer = attn + mlp + norms
    emb = vocab * d * (1 if tied else 2)
    return emb + layers * per_layer + d


def design(n_target: float, vocab: int | None = None) -> dict:
    """Pick layers, width, heads and feed-forward size for about n_target parameters."""
    n_target = max(2e5, float(n_target))
    vocab = vocab or default_vocab(n_target)
    layers = _interp_layers(n_target)
    head_dim = 64 if n_target < 1e9 else 128
    tied = n_target < 1e9
    step = 64 if n_target < 1e9 else 128
    best = None
    d = step
    while d <= 65536:
        heads = max(1, d // head_dim)
        kv_heads = heads if n_target < 5e9 else max(1, min(8, heads))
        d_ff = int(round(d * 8 / 3 / 64)) * 64 or 64
        n = count_params(d, layers, heads, kv_heads, d_ff, vocab, tied)
        if best is None or abs(n - n_target) < abs(best[0] - n_target):
            best = (n, d, heads, kv_heads, d_ff)
        if n > n_target * 1.6:
            break
        d += step
    n, d, heads, kv_heads, d_ff = best
    seq_len = 512 if n < 2e7 else 1024 if n < 5e8 else 2048 if n < 2e10 else 4096
    return {
        "params": int(n), "params_text": fmt_params(n), "hidden": d, "layers": layers, "heads": heads,
        "kv_heads": kv_heads, "head_dim": head_dim, "intermediate": d_ff, "vocab": vocab, "tied": tied,
        "seq_len": seq_len, "family": "qwen2",
    }


def estimate(arch: dict, tokens: float | None = None, target: dict | None = None) -> dict:
    """What training this design costs.

    target: {"gpu": bool, "gpus": [{"name", "memory_gb"}], "cores": int, "ram_gb": float}
    """
    n = arch["params"]
    tokens = tokens or TOKENS_PER_PARAM * n
    flops = 6.0 * n * tokens
    weights_bytes = 2 * n
    train_bytes_gpu = TRAIN_BYTES_PER_PARAM_GPU * n
    train_bytes_cpu = TRAIN_BYTES_PER_PARAM_CPU * n
    # Reference: the cheapest sensible GPU setup for this size.
    ref_name = "RTX 4090 24GB" if n < 2e9 else "A100 80GB" if n < 2e10 else "H100 80GB"
    ref_mem, ref_flops, ref_price = GPUS[ref_name]
    ref_gpus = max(1, math.ceil(train_bytes_gpu / (ref_mem * 1e9 * 0.85)))
    ref_gpus = max(ref_gpus, math.ceil(flops / (ref_flops * 30 * 86400)))  # keep a run under ~30 days
    ref_seconds = max(60.0, flops / (ref_flops * ref_gpus))  # startup, data loading etc. never take 0
    ref_cost = ref_seconds / 3600 * ref_price * ref_gpus
    out = {
        "tokens": int(tokens), "tokens_text": fmt_params(tokens).replace("K", "K").replace("M", "M"),
        "flops": flops, "weights": fmt_bytes(weights_bytes), "train_memory_gpu": fmt_bytes(train_bytes_gpu),
        "reference": {"gpu": ref_name, "count": ref_gpus, "time": fmt_time(ref_seconds), "cost": fmt_money(ref_cost),
                      "seconds": ref_seconds},
    }
    if target:
        if target.get("gpu") and target.get("gpus"):
            mem = sum(g.get("memory_gb") or 0 for g in target["gpus"]) * 1e9 * 0.85
            speed = 0.0
            for g in target["gpus"]:
                name = g.get("name", "")
                known = next((v for k, v in GPUS.items() if k.split()[0].lower() in name.lower()), None)
                speed += known[1] if known else 60e12
            fits = train_bytes_gpu <= mem
            seconds = max(60.0, flops / max(speed, 1))
            out["here"] = {"fits": fits, "memory": fmt_bytes(mem), "needed": fmt_bytes(train_bytes_gpu),
                           "time": fmt_time(seconds), "seconds": seconds, "device": "GPU"}
        else:
            ram = (target.get("ram_gb") or 0) * 1e9 * 0.85
            cores = target.get("cores") or 1
            fits = train_bytes_cpu <= ram
            seconds = max(60.0, flops / (CPU_FLOPS_PER_CORE * cores))
            out["here"] = {"fits": fits, "memory": fmt_bytes(ram), "needed": fmt_bytes(train_bytes_cpu),
                           "time": fmt_time(seconds), "seconds": seconds, "device": "CPU"}
    return out


def to_hf_config(arch: dict, tokenizer_ids: dict | None = None) -> dict:
    """A transformers Qwen2Config as a dict."""
    ids = tokenizer_ids or {}
    return {
        "architectures": ["Qwen2ForCausalLM"], "model_type": "qwen2",
        "hidden_size": arch["hidden"], "num_hidden_layers": arch["layers"], "num_attention_heads": arch["heads"],
        "num_key_value_heads": arch["kv_heads"], "intermediate_size": arch["intermediate"],
        "vocab_size": arch["vocab"], "max_position_embeddings": arch["seq_len"] * 4,
        "tie_word_embeddings": arch["tied"], "hidden_act": "silu", "rms_norm_eps": 1e-6, "rope_theta": 10000.0,
        "initializer_range": 0.02, "use_sliding_window": False, "torch_dtype": "bfloat16",
        "bos_token_id": ids.get("bos"), "eos_token_id": ids.get("eos"), "pad_token_id": ids.get("pad"),
    }


def summary_lines(arch: dict) -> list[list[str]]:
    return [
        ["Parameters", f"{arch['params_text']} ({arch['params']:,})"],
        ["Layers", str(arch["layers"])],
        ["Width", f"{arch['hidden']} (feed-forward {arch['intermediate']})"],
        ["Attention heads", f"{arch['heads']}" + (f" (KV heads {arch['kv_heads']})" if arch["kv_heads"] != arch["heads"] else "")],
        ["Vocabulary", f"{arch['vocab']:,} tokens"],
        ["Context", f"{arch['seq_len']} tokens"],
    ]

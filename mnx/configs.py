"""Named model presets for the MNX 2.0 Max Coder family.

Only ``tiny`` is meant to be trained on a laptop CPU. The other presets are
pure configuration describing progressively larger models; they are provided
so the same code path scales from a toy demo to a flagship-size architecture.
"""

from __future__ import annotations

from typing import Dict

from mnx.model import MNXConfig

PRESETS: Dict[str, MNXConfig] = {
    # ~1M parameters: trains for a few hundred steps in minutes on CPU.
    "tiny": MNXConfig(
        name="mnx-2.0-max-coder-tiny",
        vocab_size=1024,
        d_model=128,
        n_layers=4,
        n_heads=4,
        n_kv_heads=2,
        d_ff=448,
        max_seq_len=256,
        dropout=0.0,
    ),
    # ~125M-class model, comparable to GPT-2 small.
    "small": MNXConfig(
        name="mnx-2.0-max-coder-small",
        vocab_size=32768,
        d_model=768,
        n_layers=12,
        n_heads=12,
        n_kv_heads=4,
        d_ff=2048,
        max_seq_len=4096,
        dropout=0.0,
    ),
    # ~1.3B-class model.
    "base": MNXConfig(
        name="mnx-2.0-max-coder-base",
        vocab_size=32768,
        d_model=2048,
        n_layers=24,
        n_heads=16,
        n_kv_heads=8,
        d_ff=5632,
        max_seq_len=8192,
        dropout=0.0,
    ),
    # Flagship ~7B-class "MNX 2.0 Max Coder".
    "max": MNXConfig(
        name="mnx-2.0-max-coder-max",
        vocab_size=32768,
        d_model=4096,
        n_layers=32,
        n_heads=32,
        n_kv_heads=8,
        d_ff=11008,
        max_seq_len=16384,
        dropout=0.0,
    ),
}

# Friendly aliases.
PRESETS["mnx-2.0-max-coder-tiny"] = PRESETS["tiny"]
PRESETS["mnx-2.0-max-coder-small"] = PRESETS["small"]
PRESETS["mnx-2.0-max-coder-base"] = PRESETS["base"]
PRESETS["mnx-2.0-max-coder-max"] = PRESETS["max"]


def get_config(name: str, **overrides) -> MNXConfig:
    """Return a copy of a named preset, optionally overriding fields."""
    try:
        cfg = PRESETS[name]
    except KeyError as e:
        raise KeyError(f"unknown preset {name!r}; available: {sorted(set(PRESETS))}") from e
    return cfg.replace(**overrides)

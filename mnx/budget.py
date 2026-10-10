"""Back-of-the-envelope compute, time and memory budget for a pretraining run.

Example::

    python -m mnx.budget --config max --tokens 140e9 --gpus 64

Uses the standard estimate of ~6 FLOPs per parameter per token (plus causal
attention), the GPU's dense bf16 peak and an assumed model FLOPs utilisation.
The model is built on the ``meta`` device, so even the 7B preset costs no
memory to inspect.
"""

from __future__ import annotations

import argparse
from typing import Optional, Sequence

import torch

from mnx.configs import PRESETS, get_config
from mnx.model import MNXCoder

# Dense bf16 peak TFLOP/s and typical on-demand cloud price per GPU-hour (USD).
GPUS = {
    "h100": (989.0, 2.5),
    "h200": (989.0, 3.5),
    "a100": (312.0, 1.5),
    "b200": (2250.0, 5.0),
}


def budget(config: str, tokens: float, gpus: int, gpu: str = "h100", mfu: float = 0.4,
           seq_len: Optional[int] = None, price: Optional[float] = None) -> dict:
    cfg = get_config(config, **({"max_seq_len": seq_len} if seq_len else {}))
    with torch.device("meta"):
        model = MNXCoder(cfg)
    n = model.num_parameters()
    peak, default_price = GPUS[gpu]
    flops_tok = 6.0 * n + 12.0 * cfg.n_layers * cfg.d_model * cfg.max_seq_len
    total_flops = flops_tok * tokens
    gpu_hours = total_flops / (peak * 1e12 * mfu) / 3600
    # fp32 master weights + fp32 grads + two fp32 Adam moments = 16 bytes/param.
    state_gb = 16 * n / 1e9
    return {
        "name": cfg.name,
        "params": n,
        "tokens": tokens,
        "tokens_per_param": tokens / n,
        "total_flops": total_flops,
        "gpu_hours": gpu_hours,
        "wall_hours": gpu_hours / gpus,
        "cost_usd": gpu_hours * (price if price is not None else default_price),
        "train_state_gb": state_gb,
        "train_state_gb_per_gpu": state_gb / gpus,
        "bf16_weights_gb": 2 * n / 1e9,
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Estimate the compute budget of an MNX pretraining run.")
    p.add_argument("--config", default="max", help=f"preset, one of {sorted(set(PRESETS))}")
    p.add_argument("--tokens", type=float, default=None, help="training tokens (default: 20 per parameter)")
    p.add_argument("--gpus", type=int, default=8)
    p.add_argument("--gpu", choices=sorted(GPUS), default="h100")
    p.add_argument("--mfu", type=float, default=0.4, help="assumed model FLOPs utilisation")
    p.add_argument("--seq-len", type=int, default=None)
    p.add_argument("--price", type=float, default=None, help="USD per GPU-hour")
    a = p.parse_args(argv)

    tokens = a.tokens
    if tokens is None:
        with torch.device("meta"):
            tokens = 20.0 * MNXCoder(get_config(a.config)).num_parameters()
    b = budget(a.config, tokens, a.gpus, a.gpu, a.mfu, a.seq_len, a.price)
    print(f"model            {b['name']}  ({MNXCoder.format_params(b['params'])} params)")
    print(f"tokens           {b['tokens'] / 1e9:,.1f}B  ({b['tokens_per_param']:.0f} tokens/param)")
    print(f"compute          {b['total_flops']:.2e} FLOPs")
    print(f"GPU-hours        {b['gpu_hours']:,.0f} {a.gpu.upper()}-hours at {a.mfu:.0%} MFU")
    print(f"wall-clock       {b['wall_hours']:,.1f} h on {a.gpus} GPUs ({b['wall_hours'] / 24:,.1f} days)")
    print(f"est. cost        ${b['cost_usd']:,.0f}")
    print(f"training state   {b['train_state_gb']:,.0f} GB total, {b['train_state_gb_per_gpu']:,.1f} GB/GPU with FSDP (+ activations)")
    print(f"bf16 weights     {b['bf16_weights_gb']:,.1f} GB")


if __name__ == "__main__":
    main()

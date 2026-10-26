"""Quantize an MNXCoder checkpoint and export it for phones with ExecuTorch.

Train on a PC (``python -m mnx.train``), then::

    pip install executorch
    python -m mnx.export_mobile --ckpt runs/cpu --out mobile/

writes ``mobile/mnx.pte`` (the program, lowered to the XNNPACK backend that
runs on Android and iOS CPUs, with 8-bit weights unless ``--no-quantize``)
and ``mobile/tokenizer.json``. The program maps token ids ``(1, T)`` to the
next-token logits ``(1, vocab)``; the app runs the sampling loop. On a phone,
load it with ExecuTorch's Android library (``org.pytorch:executorch-android``)
or iOS framework. ``--check`` reruns the exported program with the ExecuTorch
runtime on this machine and compares it with PyTorch.

Vendor NPU backends (Qualcomm QNN, MediaTek NeuroPilot) need their SDKs and
are not used here; XNNPACK works on every phone.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch

from mnx.train import TOKENIZER_NAME, load_checkpoint


class NextTokenLogits(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        logits, _, _ = self.model(idx)
        return logits[:, -1, :]


def export(ckpt: str, out: str, quantize: bool = True, check: bool = False) -> dict:
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    from executorch.exir import to_edge_transform_and_lower

    model, tok, _ = load_checkpoint(ckpt)
    wrapper = NextTokenLogits(model).eval()
    max_len = model.cfg.max_seq_len
    example = (torch.randint(0, tok.vocab_size, (1, 16)),)
    if quantize:
        from torchao.quantization import Int8DynamicActivationIntxWeightConfig, quantize_
        from torchao.quantization.granularity import PerAxis

        quantize_(wrapper, Int8DynamicActivationIntxWeightConfig(weight_dtype=torch.int8, weight_granularity=PerAxis(0)))
    seq = torch.export.Dim("seq", min=2, max=max_len)
    with torch.no_grad():
        ep = torch.export.export(wrapper, example, dynamic_shapes=({1: seq},), strict=False)
    program = to_edge_transform_and_lower(ep, partitioner=[XnnpackPartitioner()]).to_executorch()
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    pte = out_dir / "mnx.pte"
    pte.write_bytes(program.buffer)
    shutil.copy(Path(ckpt) / TOKENIZER_NAME if Path(ckpt).is_dir() else Path(ckpt).parent / TOKENIZER_NAME,
                out_dir / TOKENIZER_NAME)
    meta = {"pte": str(pte), "bytes": pte.stat().st_size, "quantized": quantize, "max_seq_len": max_len,
            "vocab_size": tok.vocab_size, "bos_id": tok.bos_id, "eos_id": tok.eos_id}
    if check:
        from executorch.runtime import Runtime

        method = Runtime.get().load_program(pte).load_method("forward")
        ids = torch.tensor([tok.encode("def add(a, b):\n", add_bos=True)])
        got = method.execute([ids])[0]
        with torch.no_grad():
            want = NextTokenLogits(load_checkpoint(ckpt)[0]).eval()(ids)
        meta["check_same_top_token"] = bool(got.argmax().item() == want.argmax().item())
        meta["check_max_abs_diff"] = float((got - want).abs().max())
    (out_dir / "mnx.json").write_text(json.dumps(meta, indent=2))
    return meta


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Export an MNXCoder checkpoint to ExecuTorch for phones.")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", default="mobile")
    p.add_argument("--no-quantize", action="store_true")
    p.add_argument("--check", action="store_true", help="run the .pte with the ExecuTorch runtime and compare")
    a = p.parse_args(argv)
    print(json.dumps(export(a.ckpt, a.out, not a.no_quantize, a.check), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

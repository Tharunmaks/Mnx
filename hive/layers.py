"""Layer-by-layer streaming: a saved model split into one file per transformer layer.

A phone (or any small machine) never holds the whole model: it fetches layer 0, runs it,
drops it, fetches layer 1, … so memory is one layer plus activations. Training works the
same way in reverse (recompute a layer, back-propagate through it, update it, send it back).

This module does the server half with no PyTorch: it reads the safetensors header, slices
the raw bytes per layer, writes a manifest, and merges trained layers back into the model.
`hive/recipes/llm-pretrain/stream.py` is the runner that lives on the phone.
"""

from __future__ import annotations

import json
import math
import re
import shutil
import struct
import time
from pathlib import Path

from . import architect

DTYPE_BYTES = {"F64": 8, "I64": 8, "F32": 4, "I32": 4, "BF16": 2, "F16": 2, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1, "F8_E4M3": 1, "F8_E5M2": 1}
LAYER_RE = re.compile(r"^(?:model\.|transformer\.)?layers\.(\d+)\.")
SMALL_FILES = ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
               "special_tokens_map.json", "vocab.json", "merges.txt", "tokenizer.model", "mnx_model.json")
RUNTIME_BYTES = 1.2e9  # Python + PyTorch + the tokenizer on a phone
PHONE_STORAGE_READ = 1.2e9  # bytes/s a phone reads its own storage (UFS 3/4)
PHONE_FLOPS_PER_CORE = 1.5e10  # fp16/bf16 matmul per big core, roughly


# ---------- safetensors without torch ----------
def _read_header(path: Path) -> tuple[dict, int]:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    return header, 8 + n


def tensor_table(model_dir: Path) -> list[dict]:
    """Every tensor of a model: name, dtype, shape, file and byte range (handles sharded models)."""
    out = []
    files = sorted(model_dir.glob("*.safetensors"))
    if not files:
        raise ValueError("No safetensors weights in this model folder")
    for file in files:
        header, base = _read_header(file)
        for name, info in header.items():
            if name == "__metadata__":
                continue
            s, e = info["data_offsets"]
            out.append({"name": name, "dtype": info["dtype"], "shape": info["shape"], "file": file.name, "start": base + s, "end": base + e})
    return out


def _write_safetensors(dest: Path, tensors: list[dict], model_dir: Path) -> int:
    """Write the given tensors (slices of the source files) as one safetensors file; returns its size."""
    header, offset = {}, 0
    for t in tensors:
        n = t["end"] - t["start"]
        header[t["name"]] = {"dtype": t["dtype"], "shape": t["shape"], "data_offsets": [offset, offset + n]}
        offset += n
    header["__metadata__"] = {"format": "pt", "mnx": "layer"}
    hb = json.dumps(header, separators=(",", ":")).encode()
    hb += b" " * (-len(hb) % 8)
    with open(dest, "wb") as out:
        out.write(struct.pack("<Q", len(hb)) + hb)
        for t in tensors:
            with open(model_dir / t["file"], "rb") as src:
                src.seek(t["start"])
                remaining = t["end"] - t["start"]
                while remaining:
                    chunk = src.read(min(remaining, 1 << 24))
                    out.write(chunk)
                    remaining -= len(chunk)
    return dest.stat().st_size


def group_of(name: str) -> str:
    m = LAYER_RE.match(name)
    return f"layer_{int(m.group(1)):04d}" if m else "shared"


def split(model_dir: Path, layers_dir: Path, force: bool = False) -> dict:
    """Split model_dir/*.safetensors into layers_dir/layer_NNNN.safetensors + shared.safetensors + manifest.json."""
    manifest_path = layers_dir / "manifest.json"
    if manifest_path.exists() and not force:
        return json.loads(manifest_path.read_text())
    table = tensor_table(model_dir)
    groups: dict[str, list[dict]] = {}
    for t in table:
        groups.setdefault(group_of(t["name"]), []).append(t)
    layers_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for g in sorted(groups):
        size = _write_safetensors(layers_dir / f"{g}.safetensors", groups[g], model_dir)
        params = sum(math.prod(t["shape"]) for t in groups[g])
        files.append({"name": g, "file": f"{g}.safetensors", "bytes": size, "params": params, "tensors": len(groups[g])})
    for name in SMALL_FILES:
        if (model_dir / name).exists():
            shutil.copy(model_dir / name, layers_dir / name)
    config = json.loads((model_dir / "config.json").read_text()) if (model_dir / "config.json").exists() else {}
    manifest = {
        "version": 1, "created": int(time.time()), "layers": sum(1 for f in files if f["name"].startswith("layer_")),
        "files": files, "small_files": [n for n in SMALL_FILES if (layers_dir / n).exists()],
        "params": sum(f["params"] for f in files), "bytes": sum(f["bytes"] for f in files),
        "largest_layer_bytes": max((f["bytes"] for f in files if f["name"].startswith("layer_")), default=0),
        "dtype": table[0]["dtype"], "model_type": config.get("model_type"), "hidden": config.get("hidden_size"),
        "tied": bool(config.get("tie_word_embeddings", False)), "trained_steps": 0,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest


def merge(layers_dir: Path, model_dir: Path) -> int:
    """Rebuild model_dir/model.safetensors from the (possibly retrained) layer files."""
    manifest = json.loads((layers_dir / "manifest.json").read_text())
    tensors = []
    for f in manifest["files"]:
        path = layers_dir / f["file"]
        header, base = _read_header(path)
        for name, info in header.items():
            if name == "__metadata__":
                continue
            s, e = info["data_offsets"]
            tensors.append({"name": name, "dtype": info["dtype"], "shape": info["shape"], "file": path.name, "start": base + s, "end": base + e})
    for old in model_dir.glob("*.safetensors"):
        old.unlink()
    (model_dir / "model.safetensors.index.json").unlink(missing_ok=True)
    return _write_safetensors(model_dir / "model.safetensors", tensors, layers_dir)


def check_layer_file(path: Path, expected: dict) -> None:
    """An uploaded layer must hold exactly the tensors (names, dtypes, shapes) of the one it replaces."""
    header, _ = _read_header(path)
    got = {k: (v["dtype"], list(v["shape"])) for k, v in header.items() if k != "__metadata__"}
    if got != expected:
        raise ValueError("The uploaded layer doesn't match the model's layer (tensor names, dtypes or shapes differ)")


def expected_tensors(layers_dir: Path, name: str) -> dict:
    header, _ = _read_header(layers_dir / f"{name}.safetensors")
    return {k: (v["dtype"], list(v["shape"])) for k, v in header.items() if k != "__metadata__"}


# ---------- what layer streaming costs on a given phone ----------
def plan(params: float, layers: int, hidden: int, seq_len: int = 512, dtype_bytes: int = 2, ram_gb: float = 8,
         storage_free_gb: float | None = None, link_mb_s: float = 40, cores: int = 8, train_tokens: int = 512) -> dict:
    """Honest numbers for running / training a model layer by layer on a device with that much memory."""
    layers = max(1, int(layers))
    total_bytes = params * dtype_bytes
    layer_bytes = total_bytes / layers * 0.92  # embeddings and the head live in the shared file
    shared_bytes = total_bytes - layer_bytes * layers
    activations = seq_len * hidden * dtype_bytes * 12  # attention scores, MLP width, KV for one layer
    need = max(layer_bytes, shared_bytes) + activations + RUNTIME_BYTES
    fits = need <= ram_gb * 1e9 * 0.9
    cached = storage_free_gb is not None and total_bytes * 1.05 <= storage_free_gb * 1e9
    read_speed = PHONE_STORAGE_READ if cached else link_mb_s * 1e6
    compute_flops = cores * PHONE_FLOPS_PER_CORE
    run_sec = total_bytes / read_speed + 2 * params / compute_flops  # one new token: stream every layer once, compute once
    # one training step: forward + recompute-forward + backward over train_tokens, read every layer 2×, write it back once
    train_sec = (2 * total_bytes / read_speed + total_bytes / (link_mb_s * 1e6) + 6 * params * train_tokens / compute_flops)
    return {
        "fits": fits, "memory_needed": architect.fmt_bytes(need), "memory_needed_bytes": need, "layer": architect.fmt_bytes(layer_bytes),
        "shared": architect.fmt_bytes(shared_bytes), "total": architect.fmt_bytes(total_bytes),
        "cached_on_phone": cached, "storage_needed": architect.fmt_bytes(total_bytes * 1.05),
        "seconds_per_token": run_sec, "per_token": "under a second" if run_sec < 1 else architect.fmt_time(run_sec),
        "seconds_per_step": train_sec, "per_step": "under a second" if train_sec < 1 else architect.fmt_time(train_sec), "train_tokens": train_tokens,
        "source": "the phone's storage" if cached else f"the Hive over the network ({link_mb_s:.0f} MB/s)",
        "ram_gb": ram_gb, "layers": layers,
    }


def plan_text(p: dict, name: str) -> str:
    where = p["source"]
    s = (f"Layer by layer, {name} needs {p['memory_needed']} of memory on the phone (one layer is {p['layer']}; "
         f"the whole model is {p['total']}), streamed from {where}. ")
    s += f"About {p['per_token']} per generated token and {p['per_step']} per training step of {p['train_tokens']:,} tokens."
    if not p["fits"]:
        s += f" That doesn't fit a phone with {p['ram_gb']:.0f} GB: one layer alone is bigger than its memory."
    return s

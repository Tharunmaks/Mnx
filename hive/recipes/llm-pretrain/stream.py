#!/usr/bin/env python3
"""Mnx layer-by-layer streaming runner: run or train a model that is far bigger than this device.

Only one transformer layer is in memory at a time. Weights are fetched from the Hive
(GET /api/lab/models/<id>/layers/<layer>) one layer at a time, used, and dropped; trained
layers are sent back (PUT) so the Hive's copy of the model is the one that learns.

  python stream.py --hive http://server:8000 --token T --model m-xxxx run --prompt "Once upon a time" --max-new 40
  python stream.py ... train --text story.txt --steps 20 --lr 1e-4      (or --sample for the Hive's demo text)
  python stream.py ... serve --port 8765                                 (OpenAI-style /v1/chat/completions)

Needs: torch, transformers, safetensors (pip install torch transformers safetensors). Works on a
phone in Termux, a laptop, or any server. --cache DIR keeps fetched layers on local storage when
there is room, so the next pass reads from disk instead of the network.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from safetensors.torch import load_file, save_file


def log(*a):
    print(*a, flush=True)


def metric(**kw):
    log("MNX_METRIC " + json.dumps(kw))


# ---------- the layer store on the Hive ----------
class Store:
    def __init__(self, hive: str, token: str, model: str, cache: str | None):
        self.base = hive.rstrip("/") + f"/api/lab/models/{model}/layers"
        self.headers = {"Authorization": f"Bearer {token}"}
        self.cache = cache
        if cache:
            os.makedirs(cache, exist_ok=True)
        self.manifest = json.loads(self._get("").decode())
        self.fetched = 0
        self.sent = 0

    def _get(self, path: str) -> bytes:
        req = urllib.request.Request(self.base + ("/" + path if path else ""), headers=self.headers)
        with urllib.request.urlopen(req, timeout=600) as r:
            return r.read()

    def small(self, name: str) -> str:
        path = self.path(name)
        if not os.path.exists(path):
            with open(path, "wb") as f:
                f.write(self._get(name))
        return path

    def path(self, name: str) -> str:
        return os.path.join(self.cache or ".mnx_layers", name)

    def layer(self, name: str) -> dict:
        """Tensors of one layer file; fetched from the Hive unless cached locally."""
        path = self.path(name + ".safetensors")
        if not os.path.exists(path):
            data = self._get(name)
            self.fetched += len(data)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as f:
                f.write(data)
            keep = self.cache and shutil.disk_usage(self.cache).free > 2 * self.manifest["bytes"]
            tensors = load_file(path)
            if not keep:
                os.remove(path)  # no room to keep the model locally: stream it again next pass
            return tensors
        return load_file(path)

    def put_layer(self, name: str, tensors: dict) -> None:
        path = self.path(name + ".safetensors")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        save_file(tensors, path, metadata={"format": "pt", "mnx": "layer"})
        with open(path, "rb") as f:
            data = f.read()
        req = urllib.request.Request(self.base + "/" + name, data=data, method="PUT",
                                     headers={**self.headers, "Content-Type": "application/octet-stream"})
        with urllib.request.urlopen(req, timeout=600) as r:
            r.read()
        self.sent += len(data)
        if not (self.cache and shutil.disk_usage(self.cache).free > 2 * self.manifest["bytes"]):
            os.remove(path)

    def merge(self, steps: int, loss: float | None) -> dict:
        body = json.dumps({"steps": steps, "loss": loss}).encode()
        req = urllib.request.Request(self.base + "/merge", data=body, method="POST",
                                     headers={**self.headers, "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=3600) as r:
            return json.loads(r.read().decode())


# ---------- the model, one layer at a time ----------
class Streamer:
    def __init__(self, store: Store, dtype: torch.dtype, device: str):
        from transformers import AutoConfig, AutoTokenizer
        from transformers.models.qwen2 import modeling_qwen2 as q

        for name in store.manifest["small_files"]:
            store.small(name)
        folder = os.path.dirname(store.path("config.json"))
        self.config = AutoConfig.from_pretrained(folder)
        try:  # SDPA applies the causal mask itself when no mask is passed; eager would not
            self.config._attn_implementation = "sdpa"
        except Exception:
            pass
        self.tok = AutoTokenizer.from_pretrained(folder)
        self.store, self.dtype, self.device = store, dtype, device
        self.n_layers = int(self.config.num_hidden_layers)
        self.q = q
        self.rotary = q.Qwen2RotaryEmbedding(self.config).to(device)
        # One decoder layer module, reused for every layer (same shapes): load weights in, run, load the next.
        self.layer = q.Qwen2DecoderLayer(self.config, layer_idx=0).to(device=device, dtype=dtype)
        self.layer.eval()
        self.norm = q.Qwen2RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps).to(device=device, dtype=dtype)
        self.shared = None  # embed / norm / head tensors (small for tiny models, 2× vocab×hidden for big ones)
        # trained tensors go back in the dtype the Hive stores the model in, whatever we compute in
        self._stored_dtype = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}.get(store.manifest.get("dtype"), torch.float32)
        self._shared_dtype = self._stored_dtype

    def _shared(self) -> dict:
        if self.shared is None:
            self.shared = {k: v.to(self.device, self.dtype) for k, v in self.store.layer("shared").items()}
            self.norm.weight.data.copy_(self._find("norm.weight"))
        return self.shared

    def _find(self, suffix: str):
        for k, v in self.shared.items():
            if k.endswith(suffix) and "layers." not in k:
                return v
        raise KeyError(suffix)

    def embed(self, ids):
        w = self._find("embed_tokens.weight")
        return torch.nn.functional.embedding(ids, w)

    def head(self, hidden):
        self._shared()
        w = self._find("lm_head.weight") if any(k.endswith("lm_head.weight") for k in self.shared) else self._find("embed_tokens.weight")
        return torch.nn.functional.linear(self.norm(hidden), w)

    def load_layer(self, i: int, grad: bool = False) -> None:
        tensors = self.store.layer(f"layer_{i:04d}")
        prefix = next(k for k in tensors if ".layers." in k).split(".layers.")[0] + f".layers.{i}."
        state = {k[len(prefix):]: v.to(self.device, self.dtype) for k, v in tensors.items()}
        self.layer.load_state_dict(state, strict=True)
        for p in self.layer.parameters():
            p.requires_grad_(grad)
            p.grad = None
        self.layer.self_attn.layer_idx = i
        self._prefix = prefix

    def layer_tensors(self) -> dict:
        return {self._prefix + k: v.detach().to("cpu", self._stored_dtype).contiguous() for k, v in self.layer.state_dict().items()}

    def run_layer(self, hidden, position_ids, cache=None, cache_position=None):
        pos = self.rotary(hidden, position_ids)
        kw = {"position_ids": position_ids, "position_embeddings": pos, "use_cache": cache is not None}
        if cache is not None:
            kw.update(past_key_values=cache, cache_position=cache_position)
        out = self.layer(hidden, attention_mask=None, **kw)
        return out[0] if isinstance(out, tuple) else out

    # ----- generation -----
    @torch.no_grad()
    def generate(self, prompt: str, max_new: int = 40, temperature: float = 0.7) -> str:
        from transformers import DynamicCache
        self._shared()
        ids = self.tok(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(self.device)
        if ids.shape[1] == 0:
            ids = torch.tensor([[self.tok.bos_token_id or 0]], device=self.device)
        cache = DynamicCache()
        out_ids = []
        pos0 = 0
        cur = ids
        for step in range(max_new):
            t0 = time.time()
            n = cur.shape[1]
            position_ids = torch.arange(pos0, pos0 + n, device=self.device).unsqueeze(0)
            hidden = self.embed(cur)
            for i in range(self.n_layers):
                self.load_layer(i)
                hidden = self.run_layer(hidden, position_ids, cache, position_ids[0])
            logits = self.head(hidden[:, -1:, :]).float()[0, -1]
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                nxt = int(torch.multinomial(probs, 1))
            else:
                nxt = int(torch.argmax(logits))
            out_ids.append(nxt)
            metric(token=step + 1, layers=self.n_layers, seconds=round(time.time() - t0, 2), fetched_mb=round(self.store.fetched / 1e6, 1))
            if nxt == self.tok.eos_token_id:
                break
            pos0 += n
            cur = torch.tensor([[nxt]], device=self.device)
        return self.tok.decode(out_ids, skip_special_tokens=True)

    # ----- one training step, layer by layer (forward, then backward from the top with recomputation) -----
    def train_step(self, ids, lr: float, optimizer: str = "sgd") -> float:
        self._shared()
        n = ids.shape[1]
        position_ids = torch.arange(0, n, device=self.device).unsqueeze(0)
        inputs, labels = ids[:, :-1], ids[:, 1:]
        position_ids = position_ids[:, :-1]
        # forward: keep the input of every layer (activations), weights come and go
        acts = []
        with torch.no_grad():
            hidden = self.embed(inputs)
            for i in range(self.n_layers):
                acts.append(hidden.to("cpu"))
                self.load_layer(i)
                hidden = self.run_layer(hidden, position_ids)
        # loss and the gradient flowing into the top layer's output (also updates norm + head)
        hidden = hidden.detach().requires_grad_(True)
        self.norm.weight.requires_grad_(True)
        head_w = (self._find("lm_head.weight") if any(k.endswith("lm_head.weight") for k in self.shared)
                  else self._find("embed_tokens.weight"))
        head_w.requires_grad_(True)
        logits = torch.nn.functional.linear(self.norm(hidden), head_w).float()
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
        loss.backward()
        grad_out = hidden.grad
        self._sgd([self.norm.weight, head_w], lr)
        # backward: recompute each layer from its saved input, push the gradient through, update it, send it back
        for i in reversed(range(self.n_layers)):
            x = acts[i].to(self.device).detach().requires_grad_(True)
            self.load_layer(i, grad=True)
            with torch.enable_grad():
                y = self.run_layer(x, position_ids)
                y.backward(grad_out)
            grad_out = x.grad
            self._sgd(list(self.layer.parameters()), lr)
            self.store.put_layer(f"layer_{i:04d}", self.layer_tensors())
        # embeddings get the gradient that reached layer 0's input (tied weights already took the head's share)
        emb = self._find("embed_tokens.weight")
        if emb.grad is None:
            emb.requires_grad_(True)
        g = torch.zeros_like(emb)
        g.index_add_(0, inputs.reshape(-1), grad_out.reshape(-1, grad_out.shape[-1]).to(g.dtype))
        emb.data.add_(g, alpha=-lr)
        self.store.put_layer("shared", {k: v.detach().to("cpu", self._shared_dtype).contiguous() for k, v in self.shared.items()})
        for t in self.shared.values():
            t.grad = None
        self.norm.weight.grad = None
        return float(loss.detach())

    @staticmethod
    def _sgd(params, lr: float) -> None:
        with torch.no_grad():
            for p in params:
                if p.grad is not None:
                    g = p.grad.float()
                    g = g / max(1.0, float(g.norm()))  # clip each tensor to norm 1: safe at any size, no optimizer state
                    p.add_(g.to(p.dtype), alpha=-lr)
                    p.grad = None


# ---------- commands ----------
def cmd_run(st: Streamer, a) -> None:
    t0 = time.time()
    extra = {}
    if getattr(a, "engine", "torch") == "lpu":
        text, extra = run_lpu(st, a)
    else:
        text = st.generate(a.prompt, a.max_new, a.temperature)
    log("MNX_RESULT " + json.dumps({"prompt": a.prompt, "text": text, "seconds": round(time.time() - t0, 1),
                                    "fetched_mb": round(st.store.fetched / 1e6, 1), "layers": st.n_layers, **extra}))
    log(text)


def _lpu(a):
    """The Hive's hive/lpu.py (the virtual chip), fetched to sit next to this file and imported."""
    import importlib.util
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lpu.py")
    try:
        req = urllib.request.Request(a.hive.rstrip("/") + "/lpu.py", headers={"Authorization": f"Bearer {a.token}"})
        with urllib.request.urlopen(req, timeout=120) as r:
            open(path, "wb").write(r.read())
    except Exception as exc:
        if not os.path.exists(path):
            raise SystemExit(f"couldn't fetch lpu.py from the Hive: {exc}")
    spec = importlib.util.spec_from_file_location("lpu", path)
    lpu = importlib.util.module_from_spec(spec)
    sys.modules["lpu"] = lpu  # dataclasses resolve their annotations through sys.modules
    spec.loader.exec_module(lpu)
    return lpu


def _lpu_loaders(st: Streamer, lpu):
    def load_layer(i: int) -> dict:
        return lpu.strip_layer_prefix({k: v.float().numpy() for k, v in st.store.layer(f"layer_{i:04d}").items()}, i)

    def shared() -> dict:
        return {k: v.float().numpy() for k, v in st.store.layer("shared").items()}
    return load_layer, shared


def run_lpu(st: Streamer, a) -> tuple[str, dict]:
    """Generate on the virtual LPU: each streamed layer's weights are cut to the chips' shards and the compiled schedule
    runs cycle by cycle, with real multiplies."""
    lpu = _lpu(a)
    load_layer, shared = _lpu_loaders(st, lpu)
    ids = st.tok(a.prompt, add_special_tokens=False)["input_ids"] or [st.tok.bos_token_id or 0]
    m = lpu.Model(st.config.to_dict(), load_layer, shared, seq=len(ids) + a.max_new, chips=a.chips)
    P = m.plan
    log(f"Virtual LPU: {P.chips} chip(s), {P.cycles_per_token:,} cycles per token, weights {'resident' if P.resident else 'streamed every token'}, "
        f"{P.instructions_per_token} instructions per token")

    def on_token(n, _t, s):
        metric(token=n, layers=st.n_layers, seconds=round(s["sim_seconds_per_token"], 2), fetched_mb=round(st.store.fetched / 1e6, 1),
               cycles=s["cycles_per_token"], chip_us=round(s["chip_seconds_per_token"] * 1e6, 1))

    out = m.generate(ids, a.max_new, a.temperature, eos=st.tok.eos_token_id, on_token=on_token)
    s = m.token_stats()
    return st.tok.decode(out, skip_special_tokens=True), {"engine": "lpu", "chips": P.chips, "cycles_per_token": P.cycles_per_token,
                                                            "chip_tokens_per_second": round(s["chip_tokens_per_second"], 1),
                                                            "sim_seconds_per_token": round(s["sim_seconds_per_token"], 2)}


def read_text(st: Streamer, a) -> str:
    if a.text:
        return open(a.text, encoding="utf-8", errors="replace").read()
    req = urllib.request.Request(a.hive.rstrip("/") + "/api/lab/stream/sample", headers={"Authorization": f"Bearer {a.token}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read().decode()


def train_lpu(st: Streamer, a) -> None:
    """Train on the virtual LPU: every step is one compiled SGD step on the chips (forward, backward, update); the trained
    layers go back to the Hive like the torch path's, and the Hive merges them."""
    lpu = _lpu(a)
    load_layer, shared = _lpu_loaders(st, lpu)
    text = read_text(st, a)
    ids = st.tok(text, add_special_tokens=False)["input_ids"]
    S = max(8, min(a.seq_len, 128))
    if len(ids) < S + 1:
        raise SystemExit(f"the text has {len(ids)} tokens; at least {S + 1} are needed")
    m = lpu.Model(st.config.to_dict(), load_layer, shared, seq=S, spec=lpu.ChipSpec(word_bytes=4), chips=a.chips, train=True, lr=a.lr)
    P = m.plan
    log(f"Virtual LPU: {P.chips} chip(s), {P.cycles_per_token:,} cycles per step of {S} tokens, weights {'resident' if P.resident else 'streamed every step'}")
    loss = None
    for step in range(a.steps):
        t0 = time.time()
        start = (step * S) % (len(ids) - S - 1)
        loss = m.train_step(ids[start:start + S], ids[start + 1:start + S + 1])
        metric(step=step + 1, loss=round(loss, 4), seconds=round(time.time() - t0, 2), sent_mb=round(st.store.sent / 1e6, 1),
               cycles=P.cycles_per_token, chip_us=round(P.cycles_per_token / m.spec.clock_hz * 1e6, 1))
    m.pull_weights()
    for i in range(st.n_layers):
        orig = st.store.layer(f"layer_{i:04d}")
        prefix = next(k for k in orig if ".layers." in k).split(".layers.")[0] + f".layers.{i}."
        st.store.put_layer(f"layer_{i:04d}", {prefix + k: torch.from_numpy(np.ascontiguousarray(v)).reshape(orig[prefix + k].shape).to(st._stored_dtype)
                                              for k, v in m.weights[i].items()})
    orig = st.store.layer("shared")
    st.store.put_layer("shared", {k: torch.from_numpy(np.ascontiguousarray(m.shared().get(k, v.float().numpy()))).reshape(v.shape).to(st._stored_dtype)
                                  for k, v in orig.items()})
    res = st.store.merge(a.steps, loss)
    s = m.step_stats()
    log("MNX_RESULT " + json.dumps({"steps": a.steps, "loss": loss, "sent_mb": round(st.store.sent / 1e6, 1), "merged": res, "engine": "lpu",
                                    "chips": P.chips, "cycles_per_step": P.cycles_per_token, "tokens_per_second": round(s["tokens_per_second"], 1),
                                    "sim_seconds_per_step": round(s["sim_seconds_per_step"], 2), "losses": [round(x, 4) for x in s["losses"]]}))


def cmd_train(st: Streamer, a) -> None:
    if getattr(a, "engine", "torch") == "lpu":
        import numpy as np  # noqa: F401  (used by train_lpu through the module globals)
        globals()["np"] = np
        return train_lpu(st, a)
    text = read_text(st, a)
    ids = st.tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    seq = min(a.seq_len, int(getattr(st.config, "max_position_embeddings", 2048)))
    if len(ids) < seq + 1:
        sys.exit(f"Need at least {seq + 1} tokens of text; got {len(ids)}")
    windows = max(1, (len(ids) - 1) // seq)
    log(f"Training {st.n_layers} layers, {a.steps} steps of {seq} tokens, lr {a.lr}, {windows} windows of text")
    last = None
    for step in range(1, a.steps + 1):
        t0 = time.time()
        k = (step - 1) % windows
        chunk = ids[k * seq: k * seq + seq + 1].unsqueeze(0).to(st.device)
        last = st.train_step(chunk, a.lr)
        metric(step=step, loss=round(last, 4), seconds=round(time.time() - t0, 1), sent_mb=round(st.store.sent / 1e6, 1))
    merged = st.store.merge(a.steps, last)
    log("MNX_RESULT " + json.dumps({"steps": a.steps, "loss": last, "merged": merged, "sent_mb": round(st.store.sent / 1e6, 1)}))


def cmd_serve(st: Streamer, a) -> None:
    import threading
    lock = threading.Lock()

    class H(BaseHTTPRequestHandler):
        def _send(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self._send(200, {"ok": True, "layers": st.n_layers, "mode": "layer-streaming"})

        def do_POST(self):
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                msgs = req.get("messages") or [{"role": "user", "content": req.get("prompt", "")}]
                prompt = "".join(f"{m['role']}: {m['content']}\n" for m in msgs) + "assistant: " if len(msgs) > 1 else msgs[0]["content"]
                with lock:
                    text = st.generate(prompt, int(req.get("max_tokens", 40)), float(req.get("temperature", 0.7)))
                self._send(200, {"choices": [{"message": {"role": "assistant", "content": text}}]})
            except Exception as exc:
                self._send(400, {"error": str(exc)})

        def log_message(self, *x):
            pass

    log(f"Serving layer-streamed model on port {a.port}")
    ThreadingHTTPServer(("0.0.0.0", a.port), H).serve_forever()


def main() -> None:
    ap = argparse.ArgumentParser(description="Mnx layer-by-layer runner")
    ap.add_argument("--hive", required=True)
    ap.add_argument("--token", default=os.getenv("MNX_TOKEN", ""))
    ap.add_argument("--model", required=True, help="model id in the Hive's Lab (m-…)")
    ap.add_argument("--cache", default=os.path.expanduser("~/.mnx_layers"), help="where fetched layers may be kept")
    ap.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    sub = ap.add_subparsers(dest="mode", required=True)
    r = sub.add_parser("run"); r.add_argument("--prompt", default="Once upon a time"); r.add_argument("--max-new", type=int, default=40)
    r.add_argument("--temperature", type=float, default=0.7)
    r.add_argument("--engine", default="torch", choices=["torch", "lpu"], help="lpu: run the layers on the virtual chip simulator")
    r.add_argument("--chips", type=int, default=1, help="virtual LPU chips (with --engine lpu)")
    t = sub.add_parser("train"); t.add_argument("--text"); t.add_argument("--sample", action="store_true")
    t.add_argument("--steps", type=int, default=10); t.add_argument("--lr", type=float, default=1e-3); t.add_argument("--seq-len", type=int, default=256)
    t.add_argument("--engine", default="torch", choices=["torch", "lpu"], help="lpu: train on the virtual chip simulator (window capped at 128 tokens)")
    t.add_argument("--chips", type=int, default=1)
    s = sub.add_parser("serve"); s.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}.get(a.dtype) or (torch.float16 if device == "cuda" else torch.float32)
    store = Store(a.hive, a.token, a.model, os.path.join(a.cache, a.model))
    log(f"Model {a.model}: {store.manifest['layers']} layers, {store.manifest['bytes'] / 1e6:.1f} MB in total; one layer is "
        f"{store.manifest['largest_layer_bytes'] / 1e6:.1f} MB; running on {device} in {str(dtype).split('.')[-1]}")
    st = Streamer(store, dtype, device)
    {"run": cmd_run, "train": cmd_train, "serve": cmd_serve}[a.mode](st, a)


if __name__ == "__main__":
    main()

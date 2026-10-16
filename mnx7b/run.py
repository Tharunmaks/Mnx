#!/usr/bin/env python3
"""Runs Tharun's own Mnx 7B model (Mnx 1 "Expert Craft": a 126M-parameter
transformer core + 6.9B parameters of hashed n-gram experts) for the Mnx app.

Needs only numpy (works in Termux: pkg install python-numpy). PyTorch is not
needed: core_only.pt is read directly, and the 13.7 GB expert table is
memory-mapped, so each new token reads just a few rows from storage.

It speaks the same small HTTP API as llama-server, so Mnx can use it like any
local model:
    python3 mnx7b/run.py -m models/mnx7b --port 8089
    GET /health, GET /props, POST /v1/chat/completions (stream or not)
Also: python3 mnx7b/run.py -m models/mnx7b --prompt "Hello"   (one reply in the terminal)

Model files (models/mnx7b/): core_only.pt, table.bin, tok16k.json
The maths here matches training/mnx7b train.py (checked by test/mnx7b.test.mjs).
"""
import argparse
import json
import math
import os
import pickle
import sys
import threading
import time
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

# ───────────── reading a PyTorch checkpoint without PyTorch ─────────────
_DTYPES = {"FloatStorage": np.float32, "HalfStorage": np.float16, "BFloat16Storage": "bf16", "DoubleStorage": np.float64,
           "LongStorage": np.int64, "IntStorage": np.int32, "ShortStorage": np.int16, "CharStorage": np.int8,
           "ByteStorage": np.uint8, "BoolStorage": np.bool_}


def _bf16_to_f32(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)


class _Storage:
    def __init__(self, kind):
        self.kind = kind


class _Unpickler(pickle.Unpickler):
    def __init__(self, f, zf, prefix):
        super().__init__(f)
        self.zf, self.prefix, self.cache = zf, prefix, {}

    def find_class(self, module, name):
        if module == "torch._utils" and name in ("_rebuild_tensor_v2", "_rebuild_tensor"):
            return _rebuild_tensor
        if module == "torch._utils" and name == "_rebuild_parameter":
            return lambda data, requires_grad, backward_hooks, *a: data
        if module == "torch" and name in _DTYPES:
            return _Storage(name)
        if module == "torch" and name == "Size":
            return tuple
        if module == "collections" and name == "OrderedDict":
            import collections
            return collections.OrderedDict
        if module in ("builtins", "__builtin__") and name in ("set", "frozenset", "dict", "list", "tuple"):
            import builtins
            return getattr(builtins, name)
        raise pickle.UnpicklingError(f"refusing to load {module}.{name} (only plain tensors are allowed)")

    def persistent_load(self, pid):
        _, storage, key, _location, _numel = pid
        kind = storage.kind if isinstance(storage, _Storage) else getattr(storage, "__name__", str(storage))
        if key not in self.cache:
            raw = self.zf.read(f"{self.prefix}/data/{key}")
            dt = _DTYPES[kind]
            self.cache[key] = _bf16_to_f32(np.frombuffer(raw, np.uint16)) if dt == "bf16" else np.frombuffer(raw, dt)
        return self.cache[key]


def _rebuild_tensor(storage, offset, size, stride, *rest):
    if not size:
        return storage[offset].copy()
    itemsize = storage.itemsize
    return np.lib.stride_tricks.as_strided(storage[offset:], shape=tuple(size), strides=tuple(s * itemsize for s in stride)).copy()


def load_pt(path):
    """Load a torch.save() file of tensors/dicts/numbers as numpy arrays (safe: no code runs)."""
    with zipfile.ZipFile(path) as zf:
        name = next(n for n in zf.namelist() if n.endswith("/data.pkl"))
        with zf.open(name) as f:
            return _Unpickler(f, zf, name[: -len("/data.pkl")]).load()


# ───────────── tokenizer ─────────────
class Tokenizer:
    """tok16k.json via the `tokenizers` library if installed, else a pure-Python
    byte-level BPE that follows the same file (enough for this model)."""

    def __init__(self, path):
        try:
            from tokenizers import Tokenizer as HF
            self.hf = HF.from_file(path)
            self.encode = lambda s: self.hf.encode(s).ids
            self.decode = lambda ids: self.hf.decode(ids, skip_special_tokens=False)
            self.token_to_id = self.hf.token_to_id
            return
        except ImportError:
            pass
        self.hf = None
        spec = json.load(open(path, encoding="utf-8"))
        model = spec["model"]
        if model.get("type") != "BPE":
            raise SystemExit("This tokenizer needs the `tokenizers` package: pip install tokenizers")
        self.vocab = model["vocab"]
        self.inv = {i: t for t, i in self.vocab.items()}
        merges = [tuple(m.split(" ")) if isinstance(m, str) else tuple(m) for m in model["merges"]]
        self.ranks = {m: i for i, m in enumerate(merges)}
        self.special = {t["content"]: t["id"] for t in spec.get("added_tokens", [])}
        for t, i in self.special.items():
            self.inv[i] = t
        bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
        cs = bs[:]
        n = 0
        for b in range(256):
            if b not in bs:
                bs.append(b)
                cs.append(256 + n)
                n += 1
        self.b2u = {b: chr(c) for b, c in zip(bs, cs)}
        self.u2b = {v: k for k, v in self.b2u.items()}
        pre = spec.get("pre_tokenizer") or {}
        self.add_prefix_space = any(p.get("add_prefix_space") for p in ([pre] + pre.get("pretokenizers", [])) if p.get("type") == "ByteLevel")
        import re
        self.split = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?[^\W\d_]+| ?\d+| ?[^\s\w]+|\s+(?!\S)|\s+""")
        self.cache = {}

    def token_to_id(self, t):
        return self.special.get(t, self.vocab.get(t))

    def _bpe(self, word):
        if word in self.cache:
            return self.cache[word]
        parts = list(word)
        while len(parts) > 1:
            best = min(((self.ranks.get((parts[i], parts[i + 1]), 1 << 30), i) for i in range(len(parts) - 1)))
            if best[0] == 1 << 30:
                break
            i = best[1]
            parts[i:i + 2] = [parts[i] + parts[i + 1]]
        ids = [self.vocab[p] for p in parts if p in self.vocab]
        self.cache[word] = ids
        return ids

    def encode(self, text):
        ids = []
        specials = sorted(self.special, key=len, reverse=True)
        i = 0
        chunk = ""
        while i < len(text):
            hit = next((s for s in specials if text.startswith(s, i)), None)
            if hit:
                ids += self._encode_plain(chunk)
                ids.append(self.special[hit])
                chunk = ""
                i += len(hit)
            else:
                chunk += text[i]
                i += 1
        return ids + self._encode_plain(chunk)

    def _encode_plain(self, text):
        if not text:
            return []
        if self.add_prefix_space and not text.startswith(" "):
            text = " " + text
        out = []
        for piece in self.split.findall(text):
            out += self._bpe("".join(self.b2u[b] for b in piece.encode("utf-8")))
        return out

    def decode(self, ids):
        out, buf = [], []
        for i in ids:
            t = self.inv.get(i, "")
            if t in self.special:
                if buf:
                    out.append(bytes(buf).decode("utf-8", "replace"))
                    buf = []
                out.append(t)
            else:
                buf += [self.u2b.get(c, 32) for c in t]
        if buf:
            out.append(bytes(buf).decode("utf-8", "replace"))
        return "".join(out)


# ───────────── the model ─────────────
PR = [1000003, 998244353, 7919]
_SQ2 = 1 / math.sqrt(2)
try:
    from math import erf as _erf
    _verf = np.vectorize(_erf, otypes=[np.float32])
except ImportError:  # pragma: no cover
    _verf = None


def gelu(x):
    # Exact GELU (PyTorch's default): x * Φ(x).
    try:
        from scipy.special import erf  # fast if available
        return x * 0.5 * (1 + erf(x * _SQ2))
    except ImportError:
        return x * 0.5 * (1 + _verf(x * _SQ2))


def silu(x):
    return x / (1 + np.exp(-x))


def rmsnorm(x, w, eps):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * w


class Mnx7B:
    def __init__(self, folder):
        ck = load_pt(os.path.join(folder, "core_only.pt"))
        self.C = C = dict(ck["config"])
        self.E = int(ck["E"])
        self.step = ck.get("step")
        sd = {k: np.ascontiguousarray(v, dtype=np.float32) for k, v in ck["core"].items()}
        self.d, self.L, self.H = C["dim"], C["layers"], C["heads"]
        self.hd = self.d // self.H
        self.T = C["ctx"]
        self.emb = sd["emb.weight"]
        self.embT = np.ascontiguousarray(self.emb.T)  # output layer (tied weights)
        self.scale = 1 / math.sqrt(self.d // self.H)
        self.blocks = []
        for i in range(self.L):
            p = f"blocks.{i}."
            self.blocks.append(dict(n1=sd[p + "n1.weight"], qkv=sd[p + "qkv.weight"].T.copy(), o=sd[p + "o.weight"].T.copy(), n2=sd[p + "n2.weight"],
                                    w13=np.concatenate([sd[p + "w1.weight"], sd[p + "w3.weight"]]).T.copy(), w2=sd[p + "w2.weight"].T.copy()))
        self.nf = sd["nf.weight"]
        self.ein = sd["ein.weight"].T.copy()
        self.eout = sd["eout.weight"].T.copy()
        self.eps = float(np.finfo(np.float32).eps)  # nn.RMSNorm(eps=None) uses the dtype's eps
        f = 1.0 / (10000 ** (np.arange(0, self.hd, 2, dtype=np.float32) / self.hd))
        a = np.outer(np.arange(self.T, dtype=np.float32), f)
        self.cos, self.sin = np.cos(a), np.sin(a)
        PE = 2 * C["ex_dim"] * C["ex_hidden"]
        tpath = os.path.join(folder, "table.bin")
        size = os.path.getsize(tpath)
        if size != self.E * PE * 2:
            raise SystemExit(f"table.bin is {size:,} bytes, expected {self.E * PE * 2:,}. Re-join the parts and check the md5.")
        self.table = np.memmap(tpath, dtype=np.uint16, mode="r", shape=(self.E, PE))
        self.reset()

    def reset(self):
        self.tokens = []
        # Preallocated cache (no copying per token).
        if not hasattr(self, "kc"):
            self.kc = np.zeros((self.L, self.H, self.T, self.hd), np.float32)
            self.vc = np.zeros((self.L, self.H, self.T, self.hd), np.float32)

    def expert_ids(self, toks, start):
        """Hash ids for positions start.. (same int64 wrap-around maths as training)."""
        ix = np.asarray(toks, dtype=np.int64)
        n = len(ix)
        out = []
        with np.errstate(over="ignore"):
            for o in self.C["orders"]:
                h = np.zeros(n - start, np.int64)
                for j in range(o):
                    pos = np.arange(start, n) - j
                    prev = np.where(pos >= 0, ix[np.clip(pos, 0, None)], 0)
                    h = h * np.int64(PR[j % 3]) + prev + 1
                h = (h * np.int64(2654435761) + np.int64(o * 97)) % np.int64(self.E)
                out.append(h)
        return np.stack(out, -1)  # t, k

    def experts(self, z, ids):
        C = self.C
        w = _bf16_to_f32(np.asarray(self.table[ids.reshape(-1)])).reshape(*ids.shape, 2, C["ex_dim"], C["ex_hidden"])
        w1, w2 = w[..., 0, :, :], w[..., 1, :, :]
        h = gelu(np.einsum("td,tkdh->tkh", z, w1))
        return np.einsum("tkh,tkdh->td", h, w2) / len(C["orders"])

    def forward(self, new_tokens):
        """Feed tokens (using the cache), return logits for the last one."""
        start = len(self.tokens)
        self.tokens += list(new_tokens)
        t = len(new_tokens)
        pos = np.arange(start, start + t)
        cos, sin = self.cos[pos][None], self.sin[pos][None]
        x = self.emb[np.asarray(new_tokens)]
        ids = self.expert_ids(self.tokens, start)
        half = self.hd // 2
        for i, b in enumerate(self.blocks):
            if i == self.C["ex_layer"]:
                x = x + self.experts(x @ self.ein, ids) @ self.eout
            qkv = rmsnorm(x, b["n1"], self.eps) @ b["qkv"]
            q, k, v = (qkv[:, j * self.d:(j + 1) * self.d].reshape(t, self.H, self.hd).transpose(1, 0, 2) for j in range(3))

            def rot(z):
                z1, z2 = z[..., :half], z[..., half:]
                return np.concatenate([z1 * cos - z2 * sin, z1 * sin + z2 * cos], -1)

            q, k = rot(q), rot(k)
            n = start + t
            self.kc[i, :, start:n] = k
            self.vc[i, :, start:n] = v
            att = q @ self.kc[i, :, :n].transpose(0, 2, 1) * self.scale  # H, t, n
            if t > 1:
                att = np.where((np.arange(n)[None, :] > pos[:, None])[None], -np.inf, att)
            att = np.exp(att - att.max(-1, keepdims=True))
            att /= att.sum(-1, keepdims=True)
            y = (att @ self.vc[i, :, :n]).transpose(1, 0, 2).reshape(t, self.d)
            x = x + y @ b["o"]
            h = rmsnorm(x, b["n2"], self.eps)
            ab = h @ b["w13"]
            x = x + (silu(ab[:, :ab.shape[1] // 2]) * ab[:, ab.shape[1] // 2:]) @ b["w2"]
        return rmsnorm(x[-1], self.nf, self.eps) @ self.embT

    def full_logits(self, tokens):
        """All positions at once, no cache (for testing)."""
        self.reset()
        out = []
        for tok in tokens:
            out.append(self.forward([tok]))
        self.reset()
        return np.stack(out)


# ───────────── chat ─────────────
def build_prompt(tk, messages, budget):
    """<user> … <mnx> … <end> turns, newest kept when the history is too long."""
    turns = []
    for m in messages:
        c = m.get("content")
        if isinstance(c, list):
            c = "".join(p.get("text", "") for p in c if isinstance(p, dict))
        c = str(c or "").strip()
        if m.get("role") == "system" or not c or c.startswith("<tool_response>"):
            continue
        turns.append((m.get("role"), c))
    while turns and turns[-1][0] != "user":
        turns.pop()
    parts = []
    for role, c in turns:
        parts.append(f"<user> {c}\n<mnx>" if role == "user" else f" {c}\n<end>\n")
    text = "".join(parts)
    ids = tk.encode(text)
    while len(ids) > budget and len(turns) > 1:
        turns = turns[2:] if len(turns) > 2 else turns[-1:]
        parts = [f"<user> {c}\n<mnx>" if r == "user" else f" {c}\n<end>\n" for r, c in turns]
        ids = tk.encode("".join(parts))
    return ids[-budget:]


class Engine:
    def __init__(self, folder, threads=None):
        tok_path = next((p for p in [os.path.join(folder, "tok16k.json"), os.path.join(folder, "tokenizer.json")] if os.path.exists(p)), None)
        if not tok_path:
            raise SystemExit(f"tok16k.json is missing from {folder}. Copy it from the training machine (/workspace/mnx/tok16k.json).")
        self.tk = Tokenizer(tok_path)
        self.model = Mnx7B(folder)
        self.folder = os.path.abspath(folder)
        self.end = self.tk.token_to_id("<end>")
        self.user = self.tk.token_to_id("<user>")
        self.lock = threading.Lock()

    def generate(self, messages, max_tokens=400, temperature=0.7, top_k=40, seed=None, on_token=None, should_stop=None):
        with self.lock:
            m = self.model
            max_tokens = max(1, min(int(max_tokens or 400), m.T - 32))
            prompt = build_prompt(self.tk, messages, m.T - max_tokens)
            rng = np.random.default_rng(seed)
            t0 = time.time()
            # Reuse the cache when the new prompt continues the previous one.
            common = 0
            while common < min(len(m.tokens), len(prompt)) and m.tokens[common] == prompt[common]:
                common += 1
            if common < len(m.tokens) or common == len(prompt):
                m.reset()
                common = 0
            logits = m.forward(prompt[common:])
            t1 = time.time()
            out, text, finish = [], "", "length"
            for _ in range(max_tokens):
                if should_stop and should_stop():
                    finish = "stop"
                    break
                if len(m.tokens) >= m.T:
                    break
                lg = logits.astype(np.float64) / max(temperature, 1e-5)
                if temperature <= 0:
                    nxt = int(np.argmax(lg))
                else:
                    top = np.argpartition(lg, -top_k)[-top_k:]
                    p = np.exp(lg[top] - lg[top].max())
                    nxt = int(top[rng.choice(len(top), p=p / p.sum())])
                if nxt in (self.end, self.user):
                    finish = "stop"
                    break
                out.append(nxt)
                new_text = self.tk.decode(out)
                if on_token and not new_text.endswith("�"):
                    on_token(new_text[len(text):])
                    text = new_text
                logits = m.forward([nxt])
            text_final = self.tk.decode(out)
            if on_token and len(text_final) > len(text):
                on_token(text_final[len(text):])
            t2 = time.time()
            return text_final.strip(), finish, dict(prompt_n=len(prompt) - common, prompt_ms=(t1 - t0) * 1000, predicted_n=len(out), predicted_ms=(t2 - t1) * 1000,
                                                    prompt_per_second=(len(prompt) - common) / max(t1 - t0, 1e-6), predicted_per_second=len(out) / max(t2 - t1, 1e-6))


def serve(engine, host, port):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _json(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                return self._json(200, {"status": "ok"})
            if self.path == "/props":
                return self._json(200, {"model_path": engine.folder, "model": "mnx7b", "n_ctx": engine.model.T, "step": engine.model.step})
            return self._json(404, {"error": "not found"})

        def do_POST(self):
            if self.path not in ("/v1/chat/completions", "/chat/completions"):
                return self._json(404, {"error": "not found"})
            n = int(self.headers.get("content-length") or 0)
            req = json.loads(self.rfile.read(n) or b"{}")
            msgs = req.get("messages") or []
            kw = dict(max_tokens=req.get("max_tokens") or 400, temperature=req.get("temperature", 0.7), top_k=int(req.get("top_k") or 40), seed=req.get("seed"))
            if not req.get("stream"):
                text, finish, timings = engine.generate(msgs, **kw)
                return self._json(200, {"object": "chat.completion", "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": finish}], "timings": timings})
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.send_header("cache-control", "no-cache")
            self.end_headers()
            gone = {"v": False}

            def send(obj):
                try:
                    self.wfile.write(f"data: {json.dumps(obj)}\n\n".encode())
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    gone["v"] = True

            first = {"v": True}

            def tok(piece):
                # The model answers after a space ("<mnx> Hello"); drop it.
                if first["v"]:
                    piece = piece.lstrip()
                    if not piece:
                        return
                    first["v"] = False
                send({"choices": [{"index": 0, "delta": {"content": piece}}]})

            _, finish, timings = engine.generate(msgs, on_token=tok, should_stop=lambda: gone["v"], **kw)
            send({"choices": [{"index": 0, "delta": {}, "finish_reason": finish}], "timings": timings})
            if not gone["v"]:
                try:
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass

    ThreadingHTTPServer((host, port), H).serve_forever()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("-m", "--model", required=True, help="folder with core_only.pt, table.bin and tok16k.json")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8089)
    ap.add_argument("--prompt", help="answer one message in the terminal and exit")
    ap.add_argument("-t", "--threads", type=int)
    a, _unknown = ap.parse_known_args()  # llama-server options (-c, -np, …) are ignored
    if a.threads:
        for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ[v] = str(a.threads)
    t0 = time.time()
    engine = Engine(a.model)
    print(f"Mnx 7B loaded in {time.time() - t0:.1f}s (step {engine.model.step}, {engine.model.E:,} experts)", file=sys.stderr, flush=True)
    if a.prompt:
        text, _, t = engine.generate([{"role": "user", "content": a.prompt}], on_token=lambda s: print(s, end="", flush=True))
        print(f"\n\n[{t['predicted_n']} tokens at {t['predicted_per_second']:.1f}/s]", file=sys.stderr)
        return
    serve(engine, a.host, a.port)


if __name__ == "__main__":
    main()

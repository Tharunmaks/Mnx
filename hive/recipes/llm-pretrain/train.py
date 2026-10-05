"""Create a language model from scratch (or keep training an existing one) at any size up to 500B.

Mnx Lab protocol: params.json + data/ in, output/ out, MNX_METRIC / MNX_RESULT / MNX_STAGE lines.
- Designs a Qwen2-style transformer for the requested parameter count (architect.py),
  or loads an existing model when `init_from` is set (continued training / "add data").
- Trains its own byte-level BPE tokenizer on the data (no download needed), or uses a
  Hugging Face tokenizer when `tokenizer` is a repo id.
- Data: text/markdown/jsonl/csv files in data/, and/or Hugging Face datasets (streamed).
- One GPU, several GPUs on one machine, or several machines (torchrun; DDP for small
  models, FSDP for big ones), or CPU. bf16 on GPU, fp32 on CPU.
- Holds out 2% of the documents, then tests the model on N held-out prompts (next-token
  accuracy and perplexity) and writes sample generations.

Cluster runs: the Lab sets MNX_NNODES, MNX_NODE_RANK, MNX_MASTER_ADDR and MNX_MASTER_PORT on
every machine; this script re-launches itself with torchrun and the machines rendezvous on
the master's port (it must be reachable from the other nodes).
"""

from __future__ import annotations

import csv
import glob
import json
import math
import os
import random
import subprocess
import sys
import time

P = json.load(open("params.json"))
OUT = "output"
os.makedirs(OUT, exist_ok=True)
RANK = int(os.environ.get("RANK", "0"))
WORLD = int(os.environ.get("WORLD_SIZE", "1"))
MAIN = RANK == 0


def log(*a) -> None:
    if MAIN:
        print(*a, flush=True)


def metric(**kw) -> None:
    log("MNX_METRIC " + json.dumps(kw))


def stage(name: str) -> None:
    log(f"MNX_STAGE {name}")


# ---------- multi-GPU / multi-machine: re-launch under torchrun ----------
import torch  # noqa: E402

NNODES = int(os.environ.get("MNX_NNODES", "1") or 1)
if "RANK" not in os.environ and not P.get("single_gpu"):
    nproc = torch.cuda.device_count() if torch.cuda.is_available() else 1
    if NNODES > 1 or nproc > 1:
        args = [sys.executable, "-m", "torch.distributed.run", f"--nnodes={NNODES}", f"--nproc_per_node={nproc}"]
        if NNODES > 1:
            args += [f"--node_rank={os.environ.get('MNX_NODE_RANK', '0')}", "--rdzv_backend=c10d",
                     f"--rdzv_endpoint={os.environ.get('MNX_MASTER_ADDR', '127.0.0.1')}:{os.environ.get('MNX_MASTER_PORT', '29400')}",
                     "--rdzv_id=mnx", "--max_restarts=0"]
            print(f"Node {os.environ.get('MNX_NODE_RANK', '0')} of {NNODES}: joining the cluster through "
                  f"{os.environ.get('MNX_MASTER_ADDR')}:{os.environ.get('MNX_MASTER_PORT', '29400')} with {nproc} process(es)", flush=True)
        else:
            args += ["--standalone"]
            print(f"{nproc} GPUs found: launching with torchrun", flush=True)
        sys.exit(subprocess.call(args + sys.argv))

if WORLD > 1:
    import torch.distributed as dist
    dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))

LOCAL = int(os.environ.get("LOCAL_RANK", "0"))
DEVICE = torch.device("cuda", LOCAL) if torch.cuda.is_available() else torch.device("cpu")
DTYPE = torch.bfloat16 if DEVICE.type == "cuda" else torch.float32
torch.manual_seed(42)
random.seed(42)
if WORLD > 1:
    log(f"Distributed: {WORLD} processes on {NNODES} machine(s), backend {'nccl' if DEVICE.type == 'cuda' else 'gloo'}")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import architect  # noqa: E402  (copied next to this script by the Lab)

# ---------- data ----------
stage("data")


def read_text_files() -> list[str]:
    docs: list[str] = []
    col = P.get("text_column") or "text"
    for path in sorted(glob.glob("data/*")):
        name = path.lower()
        try:
            if name.endswith(".jsonl"):
                for line in open(path, encoding="utf-8", errors="replace"):
                    if not line.strip():
                        continue
                    o = json.loads(line)
                    if "messages" in o:
                        docs.append("\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in o["messages"]))
                    else:
                        v = o.get(col) or o.get("text") or o.get("content") or o.get("output") or ""
                        if v:
                            docs.append(str(v))
            elif name.endswith(".csv"):
                with open(path, newline="", encoding="utf-8", errors="replace") as fh:
                    r = csv.DictReader(fh)
                    key = col if col in (r.fieldnames or []) else (r.fieldnames or [None])[0]
                    docs += [row[key] for row in r if key and row.get(key)]
            elif name.endswith((".txt", ".md", ".py", ".js", ".html", ".json", ".rst")):
                text = open(path, encoding="utf-8", errors="replace").read()
                # split big files on blank lines so held-out docs are meaningful
                parts = [p for p in text.split("\n\n") if p.strip()]
                docs += parts if len(parts) > 20 else [text]
        except Exception as exc:
            log(f"Skipping {path}: {exc}")
    return docs


def dataset_sources() -> list[dict]:
    sources = []
    if (P.get("hf_dataset") or "").strip():
        sources.append({"name": P["hf_dataset"].strip(), "config": (P.get("hf_config") or "").strip() or None,
                        "column": (P.get("text_column") or "").strip(), "max_examples": int(P.get("max_examples") or 20000)})
    extra = (P.get("hf_datasets") or "").strip()
    if extra:
        try:
            for s in json.loads(extra):
                if isinstance(s, dict) and s.get("name"):
                    sources.append({"name": s["name"], "config": s.get("config") or None, "data_dir": s.get("data_dir") or None,
                                    "column": s.get("column") or "", "max_examples": int(s.get("max_examples") or P.get("max_examples") or 20000),
                                    "split": s.get("split") or "train"})
        except (ValueError, TypeError) as exc:
            sys.exit(f"hf_datasets isn't a valid JSON list: {exc}")
    return sources


def read_hf_dataset(src: dict) -> list[str]:
    from datasets import load_dataset
    name, cfg, n_max = src["name"], src.get("config"), src["max_examples"]
    where = f"{name}{' (' + cfg + ')' if cfg else ''}{' ' + src['data_dir'] if src.get('data_dir') else ''}"
    log(f"Streaming dataset {where}, up to {n_max:,} rows …")
    ds = load_dataset(name, cfg, split=src.get("split") or P.get("hf_split") or "train", streaming=True,
                      data_dir=src.get("data_dir"), token=os.getenv("HF_TOKEN") or None)
    col = src.get("column") or ""
    docs = []
    for i, row in enumerate(ds):
        if i >= n_max:
            break
        if not col:
            col = next((k for k in ("text", "content", "code", "story", "output", "document", "messages") if k in row), None) or \
                  next((k for k, v in row.items() if isinstance(v, str)), "text")
            log(f"Using column '{col}'")
        v = row.get(col)
        if isinstance(row.get("messages"), list):
            v = "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in row["messages"] if isinstance(m, dict))
        if isinstance(v, str) and v.strip():
            docs.append(v)
        if i and i % 5000 == 0:
            log(f"  {i:,} rows …")
    log(f"  {len(docs):,} documents from {where}")
    return docs


docs = read_text_files()
for src in dataset_sources():
    docs += read_hf_dataset(src)
if not docs:
    sys.exit("No training text found: upload .txt/.md/.jsonl/.csv files or set a Hugging Face dataset")
random.shuffle(docs)
chars = sum(len(d) for d in docs)
log(f"{len(docs):,} documents, {chars / 1e6:.1f}M characters")

# ---------- architecture + tokenizer ----------
init_from = (P.get("init_from") or "").strip()
if init_from:
    from transformers import AutoConfig, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(init_from)
    cfg = AutoConfig.from_pretrained(init_from)
    arch = {"params": None, "hidden": cfg.hidden_size, "layers": cfg.num_hidden_layers, "heads": cfg.num_attention_heads,
            "kv_heads": getattr(cfg, "num_key_value_heads", cfg.num_attention_heads), "intermediate": cfg.intermediate_size,
            "vocab": cfg.vocab_size, "seq_len": int(P.get("seq_len") or 0) or min(1024, getattr(cfg, "max_position_embeddings", 1024)),
            "tied": getattr(cfg, "tie_word_embeddings", False)}
    log(f"Continuing from {init_from}")
else:
    target = float(P.get("custom_params") or 0) or architect.parse_size(str(P.get("size") or "50M")) or 5e7
    target = min(target, architect.MAX_PARAMS)
    tok_id = (P.get("tokenizer") or "own").strip()
    vocab = None
    if tok_id.lower() not in ("own", "auto", ""):
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(tok_id)
        vocab = len(tok)
    arch = architect.design(target, vocab)
    if int(P.get("seq_len") or 0):
        arch["seq_len"] = int(P["seq_len"])
    if tok_id.lower() in ("own", "auto", ""):
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
        from transformers import PreTrainedTokenizerFast
        vsize = min(arch["vocab"], max(1024, chars // 20))  # tiny data can't fill a big vocabulary
        log(f"Training a {vsize:,}-token BPE tokenizer on the data …")
        bpe = Tokenizer(models.BPE())
        bpe.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        bpe.decoder = decoders.ByteLevel()
        bpe.train_from_iterator(docs, trainers.BpeTrainer(vocab_size=vsize, special_tokens=["<|endoftext|>", "<|pad|>"],
                                                          initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False))
        tok = PreTrainedTokenizerFast(tokenizer_object=bpe, eos_token="<|endoftext|>", bos_token="<|endoftext|>", pad_token="<|pad|>")
        tok.chat_template = ("{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}\n{% endfor %}"
                             "{% if add_generation_prompt %}assistant:{% endif %}")
        if len(tok) != arch["vocab"]:
            arch = architect.design(target, len(tok))
            arch["seq_len"] = int(P.get("seq_len") or 0) or arch["seq_len"]
    log(f"Design: {arch['params_text']} parameters · {arch['layers']} layers · width {arch['hidden']} · "
        f"{arch['heads']} heads · vocab {arch['vocab']:,} · context {arch['seq_len']}")

if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
SEQ = int(arch["seq_len"])

# ---------- tokenize ----------
stage("tokenizing")
log("Tokenizing …")
ids: list[int] = []
eos = tok.eos_token_id if tok.eos_token_id is not None else 0
n_hold = max(1, len(docs) // 50)
hold_ids: list[int] = []
for i, d in enumerate(docs):
    enc = tok(d, add_special_tokens=False)["input_ids"]
    (hold_ids if i < n_hold else ids).extend(enc + [eos])
import numpy as np  # noqa: E402

train_arr = np.array(ids, dtype=np.int64)
hold_arr = np.array(hold_ids, dtype=np.int64)
del ids, hold_ids
n_train = len(train_arr)
log(f"{n_train:,} training tokens, {len(hold_arr):,} held out")
if n_train < SEQ + 1:
    sys.exit(f"Too little text: {n_train} tokens, need at least {SEQ + 1}. Add more data or a smaller context.")

# ---------- budget ----------
n_params_guess = arch.get("params") or 0
budget = int(P.get("tokens") or 0) or int(architect.TOKENS_PER_PARAM * (n_params_guess or 1e7))
max_epochs = float(P.get("epochs_max") or 3)
budget = int(min(budget, max_epochs * n_train))
batch_tokens = int(P.get("batch_tokens") or 0)
if not batch_tokens:
    n_ref = n_params_guess or 1e8
    batch_tokens = 16384 if n_ref < 2e7 else 65536 if n_ref < 3e8 else 262144 if n_ref < 3e9 else 1048576
batch_tokens = min(batch_tokens, max(SEQ, n_train // 4))
steps_total = max(10, budget // batch_tokens)
budget = steps_total * batch_tokens
log(f"Budget: {budget:,} tokens in {steps_total:,} steps of {batch_tokens:,} tokens "
    f"({budget / n_train:.1f} passes over the data)")

# ---------- model ----------
from transformers import AutoModelForCausalLM, Qwen2Config  # noqa: E402

hf_cfg = None
if init_from:
    model = AutoModelForCausalLM.from_pretrained(init_from, torch_dtype=torch.float32)
else:
    hf_cfg = architect.to_hf_config(arch, {"eos": eos, "bos": tok.bos_token_id, "pad": tok.pad_token_id})
    model = AutoModelForCausalLM.from_config(Qwen2Config(**{k: v for k, v in hf_cfg.items() if k != "architectures"}))
n_params = sum(p.numel() for p in model.parameters())
arch["params"] = n_params
arch["params_text"] = architect.fmt_params(n_params)
log(f"Model has {n_params:,} parameters ({arch['params_text']})")
if n_params >= 4e8 or P.get("grad_checkpoint"):
    model.gradient_checkpointing_enable()
model.to(DEVICE)

use_fsdp = WORLD > 1 and n_params >= 8e8 and DEVICE.type == "cuda"
if WORLD > 1:
    if use_fsdp:
        from functools import partial
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
        from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
        model = FSDP(model, auto_wrap_policy=partial(transformer_auto_wrap_policy, transformer_layer_cls={Qwen2DecoderLayer}),
                     mixed_precision=MixedPrecision(param_dtype=DTYPE, reduce_dtype=torch.float32, buffer_dtype=DTYPE),
                     sharding_strategy=ShardingStrategy.FULL_SHARD, device_id=DEVICE, use_orig_params=True)
        log(f"Using FSDP across {WORLD} GPUs")
    else:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[DEVICE.index] if DEVICE.type == "cuda" else None)
        log(f"Using DDP across {WORLD} processes")

# micro-batch: sequences per forward pass on each device
mb = int(P.get("micro_batch") or 0) or (16 if n_params < 5e7 else 8 if n_params < 5e8 else 4 if n_params < 3e9 else 1)
mb = max(1, min(mb, batch_tokens // SEQ // WORLD or 1))
accum = max(1, batch_tokens // (mb * SEQ * WORLD))
lr = float(P.get("learning_rate") or 0) or (3e-3 if n_params < 2e7 else 1e-3 if n_params < 2e8 else 6e-4 if n_params < 2e9 else 3e-4 if n_params < 2e10 else 1.5e-4)
if init_from:
    lr = float(P.get("learning_rate") or 0) or lr / 5
opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
warm = max(1, min(steps_total // 20, 500))


def lr_at(step: int) -> float:
    if step < warm:
        return lr * (step + 1) / warm
    t = (step - warm) / max(1, steps_total - warm)
    return lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t)))


rng = np.random.default_rng(42 + RANK)


def batch(arr: np.ndarray, n: int, starts=None):
    if starts is None:
        starts = rng.integers(0, len(arr) - SEQ - 1, size=n)
    x = torch.stack([torch.from_numpy(arr[s:s + SEQ]) for s in starts]).to(DEVICE)
    y = torch.stack([torch.from_numpy(arr[s + 1:s + SEQ + 1]) for s in starts]).to(DEVICE)
    return x, y


def forward(x, y):
    with torch.autocast(device_type=DEVICE.type, dtype=DTYPE, enabled=DEVICE.type == "cuda"):
        return model(input_ids=x, labels=y).loss


@torch.no_grad()
def evaluate(n_batches: int = 8) -> float:
    model.eval()
    losses = []
    for _ in range(n_batches):
        x, y = batch(hold_arr if len(hold_arr) > SEQ + 2 else train_arr, min(mb, 8))
        losses.append(forward(x, y).item())
    model.train()
    return sum(losses) / len(losses)


# ---------- train ----------
stage("training")
model.train()
t0 = time.time()
tokens_seen = 0
log_every = max(1, min(50, steps_total // 40))
eval_every = max(log_every * 4, steps_total // 10)
for step in range(steps_total):
    for g in opt.param_groups:
        g["lr"] = lr_at(step)
    total = 0.0
    for _ in range(accum):
        x, y = batch(train_arr, mb)
        try:
            loss = forward(x, y) / accum
            loss.backward()
        except torch.cuda.OutOfMemoryError:
            sys.exit("Out of GPU memory: pick a smaller model size, a shorter context, or more GPUs")
        total += loss.item()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    opt.zero_grad(set_to_none=True)
    tokens_seen += batch_tokens
    if step % log_every == 0 or step == steps_total - 1:
        el = time.time() - t0
        rate = tokens_seen / max(el, 1e-6)
        eta = (steps_total - step - 1) * batch_tokens / max(rate, 1)
        m = {"step": step + 1, "tokens": tokens_seen, "loss": round(total, 4), "lr": round(lr_at(step), 7),
             "tok_s": int(rate), "eta_min": round(eta / 60, 1)}
        if (step + 1) % eval_every == 0 or step == steps_total - 1:
            m["eval_loss"] = round(evaluate(), 4)
        metric(**m)
train_seconds = time.time() - t0

# ---------- save ----------
stage("saving")
log("Saving …")
if use_fsdp:
    from torch.distributed.fsdp import FullStateDictConfig, StateDictType
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, FullStateDictConfig(offload_to_cpu=True, rank0_only=True)):
        state = model.state_dict()
    if MAIN:
        base = AutoModelForCausalLM.from_pretrained(init_from, torch_dtype=torch.float32) if init_from else \
            AutoModelForCausalLM.from_config(Qwen2Config(**{k: v for k, v in hf_cfg.items() if k != "architectures"}))
        base.load_state_dict(state)
        base.save_pretrained(f"{OUT}/model", safe_serialization=True)
elif MAIN:
    (model.module if hasattr(model, "module") else model).save_pretrained(f"{OUT}/model", safe_serialization=True)
if MAIN:
    tok.save_pretrained(f"{OUT}/model")

# ---------- test with held-out prompts + samples ----------
result = {"params": n_params, "params_text": arch["params_text"], "tokens": tokens_seen, "steps": steps_total,
          "minutes": round(train_seconds / 60, 1), "from_scratch": not init_from}
if MAIN:
    stage("testing")
    n_test = int(P.get("test_prompts") or 0)
    pool = hold_arr if len(hold_arr) > SEQ + 2 else train_arr
    t1 = time.time()
    if n_test > 0:
        log(f"Testing with {n_test:,} held-out prompts …")
        model.eval()
        correct = total_tok = 0
        loss_sum = 0.0
        done = 0
        test_rng = np.random.default_rng(7)
        tb = min(mb, 8)
        with torch.no_grad():
            while done < n_test:
                n = min(tb, n_test - done)
                starts = test_rng.integers(0, len(pool) - SEQ - 1, size=n)
                x, y = batch(pool, n, starts)
                with torch.autocast(device_type=DEVICE.type, dtype=DTYPE, enabled=DEVICE.type == "cuda"):
                    out = model(input_ids=x)
                logits = out.logits.float()
                loss_sum += torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1)).item() * n
                correct += (logits.argmax(-1) == y).sum().item()
                total_tok += y.numel()
                done += n
                if done % max(tb * 25, 1) == 0 or done == n_test:
                    log(f"  tested {done:,}/{n_test:,} prompts")
        test_loss = loss_sum / max(n_test, 1)
        result.update({"test_prompts": n_test, "test_accuracy": round(100 * correct / max(total_tok, 1), 2),
                       "perplexity": round(math.exp(min(test_loss, 20)), 2), "eval_loss": round(test_loss, 4),
                       "test_minutes": round((time.time() - t1) / 60, 1)})
        log(f"Tested with {n_test:,} prompts: next-token accuracy {result['test_accuracy']}%, perplexity {result['perplexity']}")
    else:
        final_loss = evaluate(16)
        result["eval_loss"] = round(final_loss, 4)
        result["perplexity"] = round(math.exp(min(final_loss, 20)), 2)
    samples = []
    prompts = [p for p in str(P.get("sample_prompts") or "Once upon a time|The|def ").split("|") if p.strip()][:4]
    try:
        gen_model = AutoModelForCausalLM.from_pretrained(f"{OUT}/model", torch_dtype=torch.float32)
        gen_dev = DEVICE if n_params < 3e9 else torch.device("cpu")
        gen_model.to(gen_dev).eval()
        for ptxt in prompts:
            inp = tok(ptxt, return_tensors="pt", add_special_tokens=False).to(gen_dev)
            with torch.no_grad():
                out = gen_model.generate(**inp, max_new_tokens=60, do_sample=True, temperature=0.8, top_p=0.95,
                                         pad_token_id=tok.pad_token_id)
            samples.append({"prompt": ptxt, "text": tok.decode(out[0], skip_special_tokens=True)})
        del gen_model
    except Exception as exc:
        log(f"(Couldn't generate samples: {exc})")
    json.dump({"result": result, "samples": samples, "arch": arch, "tokens": tokens_seen},
              open(f"{OUT}/eval.json", "w"), indent=2)
    json.dump({"base_model": init_from or "from scratch", "merged": True, "from_scratch": not init_from, "arch": arch},
              open(f"{OUT}/mnx_model.json", "w"))
    result["samples"] = [{"prompt": s["prompt"], "text": s["text"][:160]} for s in samples[:3]]
    log("MNX_RESULT " + json.dumps(result))
if WORLD > 1:
    dist.barrier()
    dist.destroy_process_group()

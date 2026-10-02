"""LoRA fine-tune of a causal language model on chat / prompt-completion / text JSONL.

Mnx Lab protocol: params.json + data/ in, output/ out, MNX_METRIC / MNX_RESULT lines.
Works on CPU (slow, fine for 0.5B and small data); uses a CUDA GPU when present.
Set HF_TOKEN in the Lab's secrets for private or gated models.
"""
import glob
import json
import math
import os
import random
import sys
import time

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

P = json.load(open("params.json"))
os.makedirs("output", exist_ok=True)
base = P.get("base_model") or "Qwen/Qwen2.5-0.5B-Instruct"
max_len = int(P.get("max_length", 1024))
torch.manual_seed(42)
random.seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else (torch.float16 if device == "cuda" else torch.float32)
print(f"Device: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else os.cpu_count()} {'GPU' if device == 'cuda' else 'CPU threads'}), dtype {dtype}", flush=True)
print(f"Loading {base} …", flush=True)
tok = AutoTokenizer.from_pretrained(base)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=dtype).to(device)
model.gradient_checkpointing_enable()
model.enable_input_require_grads()


def encode(ex):
    """Return (input_ids, labels) with prompt tokens masked out (-100)."""
    if "conversations" in ex and "messages" not in ex:  # ShareGPT: [{"from": "human", "value": …}]
        roles = {"human": "user", "user": "user", "gpt": "assistant", "assistant": "assistant", "system": "system"}
        ex = {"messages": [{"role": roles.get(t.get("from"), "user"), "content": t.get("value", "")} for t in ex["conversations"]]}
    if "messages" in ex:
        msgs = ex["messages"]
        if tok.chat_template:
            full = tok.apply_chat_template(msgs, tokenize=False)
            prompt = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True) if msgs[-1]["role"] == "assistant" else ""
        else:
            full = "".join(f"{m['role']}: {m['content']}\n" for m in msgs)
            prompt = "".join(f"{m['role']}: {m['content']}\n" for m in msgs[:-1]) + "assistant: "
    elif ("instruction" in ex and ("output" in ex or "response" in ex)) or "prompt" in ex:
        if "instruction" in ex:  # Alpaca style; its "prompt" column (if any) already contains the answer
            q = ex["instruction"] + ("\n" + ex["input"] if ex.get("input") else "")
            a = ex.get("output") or ex.get("response") or ""
        else:
            q, a = ex["prompt"], ex.get("completion") or ex.get("response") or ex.get("output") or ""
        msgs = [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
        return encode({"messages": msgs})
    elif "text" in ex:
        full, prompt = ex["text"], ""
    else:
        raise ValueError(f"Unknown example format: {list(ex)[:5]}")
    ids = tok(full, add_special_tokens=False)["input_ids"][:max_len]
    if tok.eos_token_id is not None and (not ids or ids[-1] != tok.eos_token_id) and len(ids) < max_len:
        ids.append(tok.eos_token_id)
    n_prompt = len(tok(prompt, add_special_tokens=False)["input_ids"]) if prompt else 0
    labels = [-100] * min(n_prompt, len(ids)) + ids[n_prompt:]
    return ids, labels


examples = []
for f in sorted(glob.glob("data/*.jsonl")):
    for i, line in enumerate(open(f, encoding="utf-8"), 1):
        if line.strip():
            try:
                examples.append(encode(json.loads(line)))
            except Exception as exc:
                sys.exit(f"{os.path.basename(f)} line {i}: {exc}")
if not examples and P.get("hf_dataset"):
    from datasets import load_dataset
    name = P["hf_dataset"].strip()
    print(f"Downloading dataset {name} …", flush=True)
    ds = load_dataset(name, split=P.get("hf_split") or "train", token=os.getenv("HF_TOKEN") or None)
    total = len(ds)
    n = min(total, int(P.get("max_examples", 2000)))
    ds = ds.shuffle(seed=42).select(range(n))
    print(f"Using {n} of {total} examples; columns: {', '.join(ds.column_names)}", flush=True)
    for i, row in enumerate(ds):
        try:
            examples.append(encode(dict(row)))
        except Exception as exc:
            sys.exit(f"Dataset row {i}: {exc} (columns: {', '.join(ds.column_names)})")
examples = [e for e in examples if any(l != -100 for l in e[1])]
if not examples:
    sys.exit("No usable examples: upload a .jsonl file or set a Hugging Face dataset")
print(f"{len(examples)} examples, longest {max(len(e[0]) for e in examples)} tokens", flush=True)

model = get_peft_model(model, LoraConfig(
    r=int(P.get("lora_r", 16)), lora_alpha=2 * int(P.get("lora_r", 16)), lora_dropout=0.05,
    target_modules="all-linear", task_type="CAUSAL_LM"))
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Training {trainable / 1e6:.1f}M LoRA parameters", flush=True)

bs, accum, epochs = int(P.get("batch_size", 2)), int(P.get("grad_accum", 4)), int(P.get("epochs", 3))
steps_total = max(1, math.ceil(len(examples) / bs / accum) * epochs)
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=float(P.get("learning_rate", 2e-4)))
sched = get_linear_schedule_with_warmup(opt, max(1, steps_total // 20), steps_total)


def batches():
    order = list(range(len(examples)))
    random.shuffle(order)
    for i in range(0, len(order), bs):
        chunk = [examples[j] for j in order[i:i + bs]]
        width = max(len(ids) for ids, _ in chunk)
        ids = torch.full((len(chunk), width), tok.pad_token_id)
        lab = torch.full((len(chunk), width), -100)
        att = torch.zeros((len(chunk), width), dtype=torch.long)
        for k, (a, b) in enumerate(chunk):
            ids[k, :len(a)] = torch.tensor(a)
            lab[k, :len(b)] = torch.tensor(b)
            att[k, :len(a)] = 1
        yield ids.to(device), lab.to(device), att.to(device)


model.train()
step, t0, running = 0, time.time(), []
for epoch in range(1, epochs + 1):
    for n, (ids, lab, att) in enumerate(batches(), 1):
        loss = model(input_ids=ids, attention_mask=att, labels=lab).loss / accum
        loss.backward()
        running.append(loss.item() * accum)
        if n % accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            step += 1
            print("MNX_METRIC " + json.dumps({"step": step, "epoch": epoch, "loss": round(sum(running) / len(running), 4),
                                              "lr": sched.get_last_lr()[0], "sec": round(time.time() - t0)}), flush=True)
            running = []
    if running:  # leftover partial accumulation at the end of the epoch
        opt.step(); sched.step(); opt.zero_grad(); step += 1
        print("MNX_METRIC " + json.dumps({"step": step, "epoch": epoch, "loss": round(sum(running) / len(running), 4)}), flush=True)
        running = []

print("Saving …", flush=True)
merged = bool(P.get("merge", True))
if merged:
    model = model.merge_and_unload()
    model.save_pretrained("output/model", safe_serialization=True)
else:
    model.save_pretrained("output/adapter")
tok.save_pretrained("output/model" if merged else "output/adapter")
json.dump({"base_model": base, "merged": merged}, open("output/mnx_model.json", "w"))
print("MNX_RESULT " + json.dumps({"base_model": base, "examples": len(examples), "steps": step,
                                  "merged": merged, "minutes": round((time.time() - t0) / 60, 1)}), flush=True)

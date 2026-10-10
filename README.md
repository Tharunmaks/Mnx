# MNX 2.0 Max Coder

**MNX 2.0 Max Coder** is a decoder-only transformer language model for source
code, implemented from scratch in PyTorch. Every part of the pipeline lives in
this repository and depends on nothing but `torch` and `numpy`:

- a byte-level **BPE tokenizer** trained from raw files, with fill-in-the-middle
  (FIM) special tokens;
- a modern **transformer architecture** (RMSNorm, RoPE, grouped-query
  attention, SwiGLU, tied embeddings) with a KV-cache for fast decoding;
- a **data pipeline** that packs source files into training windows and applies
  the PSM fill-in-the-middle transformation;
- a **training loop** (AdamW, warmup + cosine schedule, gradient clipping and
  accumulation, validation, CSV logging, checkpoint resume);
- an **inference CLI** for code completion and fill-in-the-middle with
  greedy / temperature / top-k / top-p sampling.

No pre-trained weights or third-party model classes are used anywhere.

> **Honest note.** The bundled demo trains a ~1M-parameter model for a few
> hundred steps on a laptop CPU using 51 tiny Python snippets. It learns to
> reproduce and recombine that corpus; it is a toy. The `small`, `base` and
> `max` presets describe real model sizes (up to a 7B-class flagship), but
> pretraining them requires a GPU cluster and a large deduplicated code corpus.
> The distributed trainer for that (`mnx.pretrain`, see
> [Training the 7B model](#training-the-7b-model)) is included and tested on
> CPU, but a 7B run needs thousands of GPU-hours.

---

## Architecture

`MNXCoder` (in `mnx/model.py`) is a pre-norm decoder-only transformer:

```
tokens ─► Embedding ─► [ RMSNorm ─► GQA attention (RoPE, causal SDPA) ─► + ]
                       [ RMSNorm ─► SwiGLU MLP                       ─► + ] × n_layers
                   ─► RMSNorm ─► LM head (tied with embedding) ─► logits
```

| Component | Implementation |
|-----------|----------------|
| Normalisation | `RMSNorm`, pre-norm on both sub-blocks and before the LM head |
| Positions | Rotary position embeddings (RoPE, half-split rotation, θ = 10 000) |
| Attention | Grouped-query attention (`n_heads` query heads share `n_kv_heads` K/V heads) using `torch.nn.functional.scaled_dot_product_attention` with a causal mask |
| Feed-forward | SwiGLU: `down(silu(gate(x)) * up(x))` |
| Embeddings | Input and output embeddings tied |
| Init | N(0, 0.02); residual output projections scaled by 1/√(2·n_layers) |
| Decoding | KV cache; greedy, temperature, top-k and nucleus (top-p) sampling; `<\|eos\|>` stopping; sliding-window re-prefill past `max_seq_len` |

### Presets

Defined in `mnx/configs.py`. Only `tiny` is intended to be trained on a CPU;
the others are configuration presets. Parameter counts are exact (tied weights
counted once), computed with `MNXCoder.num_parameters()`.

| Preset | Name | d_model | Layers | Heads (Q / KV) | d_ff | Context | Vocab | Params | Non-embedding |
|--------|------|--------:|-------:|---------------:|-----:|--------:|------:|-------:|--------------:|
| `tiny`  | `mnx-2.0-max-coder-tiny`  | 128  | 4  | 4 / 2  | 448   | 256    | 1 024  | **1.02M**   | 0.89M  |
| `small` | `mnx-2.0-max-coder-small` | 768  | 12 | 12 / 4 | 2 048  | 4 096  | 32 768 | **100.7M**  | 75.5M  |
| `base`  | `mnx-2.0-max-coder-base`  | 2048 | 24 | 16 / 8 | 5 632  | 8 192  | 32 768 | **1.20B**   | 1.13B  |
| `max`   | `mnx-2.0-max-coder-max`   | 4096 | 32 | 32 / 8 | 14 336 | 16 384 | 32 768 | **7.11B**   | 6.98B  |

```python
from mnx import MNXCoder, get_config

model = MNXCoder(get_config("small"))
print(model.num_parameters())  # 100682496
```

---

## Quickstart

### 1. Install

```bash
# CPU-only PyTorch (use the matching CUDA index URL for a GPU build)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install -e .          # optional: installs the mnx-train / mnx-generate / mnx-tokenizer commands
```

### 2. Train the tokenizer

```bash
python -m mnx.tokenizer train --data data/sample --vocab-size 1024 --out runs/tiny/tokenizer.json
python -m mnx.tokenizer encode --tokenizer runs/tiny/tokenizer.json --text "def add(a, b):"
```

The tokenizer is byte-level BPE: ids 0–255 are raw bytes, the next six ids are
the special tokens `<|bos|>`, `<|eos|>`, `<|pad|>`, `<|fim_prefix|>`,
`<|fim_middle|>`, `<|fim_suffix|>`, and the rest are learned merges. Any string
round-trips losslessly.

### 3. Train the tiny demo model (CPU, ~15 s)

```bash
python -m mnx.train --config tiny --data data/sample --out runs/tiny \
    --tokenizer runs/tiny/tokenizer.json --steps 300 --batch-size 8
```

If `--tokenizer` is omitted a tokenizer is trained on `--data` automatically.
Typical output on a 4-core CPU:

```
corpus: 51 docs (46 train / 5 val), 7228 train tokens, seq_len=256, vocab=1024
model mnx-2.0-max-coder-tiny: 1.02M parameters
step      1/300 | lr 1.00e-04 | loss 6.9719 |    0.1s
step     50/300 | lr 2.97e-03 | loss 2.8520 | val 4.0697 |    2.2s
step    100/300 | lr 2.59e-03 | loss 1.5141 | val 3.9342 |    4.4s
step    200/300 | lr 1.13e-03 | loss 0.2244 | val 4.4771 |    8.7s
step    300/300 | lr 3.00e-04 | loss 0.1479 | val 4.7230 |   13.0s
done. checkpoint: runs/tiny/ckpt.pt  best val loss: 3.9342
```

Training loss falls from ≈ ln(1024) ≈ 6.9 to ≈ 0.15 while the held-out loss
bottoms out around 3.9 and then rises: with only ~7 K training tokens the model
memorises the corpus, exactly as expected for a toy run. `runs/tiny/` contains
`ckpt.pt`, `tokenizer.json`, `config.json` and `log.csv`
(`step,lr,train_loss,val_loss,elapsed_s`). Add `--resume` to continue from the
checkpoint.

Useful flags: `--seq-len`, `--grad-accum`, `--lr`, `--warmup`, `--fim-rate`
(fraction of documents rewritten as FIM, default 0.3), `--val-fraction`,
`--set key=value ...` to override any `MNXConfig` field, `--device cuda`.

### 4. Generate code

```bash
python -m mnx.generate --ckpt runs/tiny --prompt "def add(a, b):" --max-new-tokens 48 --temperature 0
```

```
--- MNX 2.0 Max Coder [mnx-2.0-max-coder-tiny, step 300] ---
def add(a, b):
    """Return the sum of a and b."""
    return a + b


def subtract(a, b):
    """Return a minus b."""
    return a - b
```

Sampling options: `--temperature` (0 = greedy), `--top-k`, `--top-p`, `--seed`,
`--max-new-tokens`, `--raw` (print only the completion), `--prompt -` to read
the prompt from stdin.

---

## Fill-in-the-middle (FIM)

During training a fraction of documents (`--fim-rate`) is rewritten in the
PSM layout, so the model learns to produce the span between a prefix and a
suffix:

```
<|bos|><|fim_prefix|> prefix <|fim_suffix|> suffix <|fim_middle|> middle <|eos|>
```

At inference time pass `--fim` with a prefix and suffix (`\n`/`\t` escapes are
interpreted); the model emits the middle until `<|eos|>`:

```bash
python -m mnx.generate --ckpt runs/tiny --fim \
    --prefix 'def is_prime(n):\n    if n < 2:\n        ' \
    --suffix '\n    if n % 2 == 0:\n        return n == 2\n' \
    --max-new-tokens 16 --temperature 0
```

Programmatic use:

```python
from mnx.generate import complete, fill_in_middle
from mnx.train import load_checkpoint

model, tok, meta = load_checkpoint("runs/tiny")
print(complete(model, tok, "def fib(n):", max_new_tokens=40, temperature=0.0))
print(fill_in_middle(model, tok, prefix="def square(x):\n    ", suffix="\n", max_new_tokens=8))
```

---

## Project layout

```
mnx/
  __init__.py     package exports (MNXCoder, MNXConfig, BPETokenizer, PRESETS, get_config)
  tokenizer.py    byte-level BPE tokenizer: train / encode / decode / save / load + CLI
  model.py        MNXConfig + MNXCoder (RMSNorm, RoPE, GQA, SwiGLU, KV cache, generate)
  configs.py      named presets: tiny / small / base / max
  data.py         corpus loading, document packing, FIM transform, CodeDataset
  train.py        training loop, LR schedule, checkpoints + CLI (python -m mnx.train)
  generate.py     completion and FIM inference + CLI (python -m mnx.generate)
  prepare.py      parallel corpus -> uint16 token shards (python -m mnx.prepare)
  pretrain.py     FSDP2 multi-GPU pretraining, sharded checkpoints, export (python -m mnx.pretrain)
  budget.py       compute / time / cost / memory estimator (python -m mnx.budget)
scripts/
  launch_7b.sh    multi-node torchrun launcher for the 7B preset
data/sample/      51 short self-authored Python snippets used by the demo and tests
tests/            pytest suite (tokenizer, model, data, training, CLIs)
pyproject.toml    package metadata (name: mnx)
requirements.txt  pinned-ish dependencies
```

## Tests

```bash
pytest -q
```

The suite covers tokenizer round trips and persistence, model output shapes,
causality (changing a future token leaves earlier logits unchanged), KV-cache
equivalence with the full forward pass, `generate()` length / eos handling /
sliding window, RoPE relativity, the FIM transform, dataset windows, a 5-step
training smoke test, checkpoint resume and both CLIs, plus the scale-up path:
shard preparation (serial/parallel equivalence, JSONL input), meta-device
init of the 7B preset, activation checkpointing, single-process and 2-rank
FSDP pretraining with resume, and export back to `ckpt.pt`. It runs in a few seconds
on CPU.

## Training the 7B model

The `max` preset (7.11B parameters) is trained with `mnx.pretrain`, a
multi-GPU trainer built on PyTorch FSDP2:

- parameters, gradients and AdamW state sharded across all GPUs (one FSDP unit
  per transformer block), with meta-device init so no rank ever holds the full
  fp32 model;
- bf16 compute with fp32 master weights and fp32 gradient reduction;
- activation checkpointing (`--grad-checkpointing`) and per-block
  `torch.compile` (`--compile`);
- streaming `uint16` token shards read with `numpy.memmap`;
- sharded checkpoints (`torch.distributed.checkpoint`) with automatic resume
  from `--out`, keeping the last `--keep-checkpoints`;
- tokens/s and MFU logging to stdout and `log.csv`.

### 1. Check the budget first

```bash
python -m mnx.budget --config max --gpus 64 --seq-len 4096
```

```
model            mnx-2.0-max-coder-max  (7.11B params)
tokens           142.3B  (20 tokens/param)
compute          6.99e+21 FLOPs
GPU-hours        4,908 H100-hours at 40% MFU
wall-clock       76.7 h on 64 GPUs (3.2 days)
est. cost        $12,269
```

These are estimates (6·N FLOPs per token plus attention, an assumed 40% MFU,
indicative cloud prices). 20 tokens/parameter is the compute-optimal minimum.
Strong open 7B models are trained on 2T to 15T+ tokens, which is 15x to 100x
this budget (`--tokens 2e12`).

### 2. Tokenizer and data

```bash
# 32k-entry BPE tokenizer on a representative sample (the trainer is pure
# Python, so keep the sample to a few hundred MB)
python -m mnx.tokenizer train --data /data/code-sample --vocab-size 32768 --out tok/tokenizer.json

# Tokenise the full corpus (directories of source files and/or .jsonl files
# with a "content" or "text" field) into ~100M-token shards
python -m mnx.prepare --input /data/code /data/the-stack.jsonl \
    --tokenizer tok/tokenizer.json --out /data/mnx-shards --workers 64 --fim-rate 0.5
```

The corpus should be deduplicated and licence-filtered beforehand. A document
lands in the validation split based on a hash of its contents
(`--val-fraction`, default 0.5%).

### 3. Pretrain

```bash
# one node, 8 GPUs
torchrun --nproc_per_node 8 -m mnx.pretrain --config max --data /data/mnx-shards \
    --out runs/mnx-2.0-max --seq-len 4096 --micro-batch 2 --grad-accum 32 \
    --steps 34000 --lr 3e-4 --min-lr 3e-5 --warmup 2000 --grad-checkpointing --compile

# multi-node: run on every node (computes grad-accum for ~4M tokens/step)
NNODES=8 NODE_RANK=0 MASTER_ADDR=node0 DATA=/data/mnx-shards ./scripts/launch_7b.sh
```

Rerunning the same command resumes from the latest checkpoint in `--out`.
On a single process (no `torchrun`) the same script trains without sharding
under `torch.autocast`, which is what the CPU tests use.

### 4. Export and use

```bash
python -m mnx.pretrain export --run runs/mnx-2.0-max --out runs/mnx-2.0-max-final  # bf16 ckpt.pt (~14 GB)
python -m mnx.generate --ckpt runs/mnx-2.0-max-final --device cuda --prompt "def quicksort(xs):"
```

A pretrained base model completes code; it does not follow chat instructions.
That takes a further supervised fine-tuning stage on instruction data, which
this repository does not include yet.

## License

MIT.

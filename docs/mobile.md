# MNX1 on a phone

Phones are good at running small models and bad at training them: phone
CPUs, GPUs and NPUs are limited by heat and battery. So MNX follows the usual
workflow: **train on a PC, quantize, deploy to the phone.**

There are two parts to put on a phone.

## 1. The MNX1 agents (works today, no build step)

The router, the coding agent, math, logic, dates, units and retrieval are
pure Python with no dependencies, so they run anywhere Python runs. On
Android, install [Termux](https://termux.dev), then:

```bash
pkg install python git
git clone https://github.com/Tharunmaks/Mnx && cd Mnx
python -m mnx.agent            # interactive MNX1, learning into ~/.mnx
```

Answers take milliseconds, except program search, which can take a few
seconds on a phone CPU.

## 2. The neural coder (ExecuTorch)

Train on a PC (see the README), then export:

```bash
pip install executorch
python -m mnx.export_mobile --ckpt runs/cpu --out mobile/ --check
```

This writes:

| File | What it is |
|---|---|
| `mobile/mnx.pte` | the model as an ExecuTorch program, lowered to the **XNNPACK** backend, with **int8** weights and dynamic int8 activations |
| `mobile/tokenizer.json` | the BPE tokenizer (byte-level, so any text encodes) |
| `mobile/mnx.json` | sizes and the BOS/EOS ids the app needs |

`--check` loads the `.pte` with the ExecuTorch runtime on the PC and checks
that it predicts the same next token as the PyTorch model.

The program maps token ids `(1, T)`, with `2 ≤ T ≤ 256`, to next-token
logits `(1, vocab)`. The app runs the loop: encode the prompt with BOS in
front, call `forward`, take the argmax (or sample), append, and stop at EOS.

On Android, add ExecuTorch's library (`org.pytorch:executorch-android`) and
load the program with `Module.load(path)`; on iOS use the ExecuTorch
framework. The tokenizer and the sampling loop must be written in the app's
language; this repository doesn't include app code yet.

### Why XNNPACK and not the NPU

XNNPACK runs on every Android and iOS CPU. The vendor NPU backends
(Qualcomm QNN for Snapdragon, MediaTek NeuroPilot for Dimensity) can be much
faster for supported models, but they need each vendor's SDK and per-chip
builds. At 10M parameters MNX's model is already fast on a phone CPU, so
the NPU isn't needed yet. To try one, swap the `XnnpackPartitioner` in
`mnx/export_mobile.py` for that vendor's partitioner.

### Training on the phone

ExecuTorch has an experimental on-device training path, but it is slow and
practical only for very small models or adapters. MNX1's own learning
(corrections, feedback, saved code) needs no gradient training, so it already
works on the phone through Termux.

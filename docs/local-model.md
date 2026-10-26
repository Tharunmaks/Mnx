# Raising coverage with a local code model (optional, no PC needed)

MNX1 answers on its own with no model, no network and no API key, and that
part never changes. But hand-written recipes plus program search only reach
problems whose shape MNX1 already knows: on the half of HumanEval that was
never looked at while writing recipes, it solves **2 of 82**. Its strength is
precision, not coverage.

If you want coverage too, you can point MNX1 at a small code model running on
**your own device** — including an Android phone, with no PC involved. MNX1
treats whatever the model says as a *proposal*: it runs every candidate in a
sandbox, throws away the ones that fail, and makes the survivors agree before
it shows you anything. The model widens the search; MNX1 stays the judge.

This is off unless you pass `--model`.

## What to expect

| Model (4-bit) | Size on disk | HumanEval pass@1, model alone |
|---|---|---|
| Qwen2.5-Coder-1.5B-Instruct | ~1.1 GB | 70.7 |
| Qwen2.5-Coder-3B-Instruct | ~2.0 GB | 84.1 |
| Qwen2.5-Coder-7B-Instruct | ~4.7 GB | 88.4 |

Those are the authors' numbers for the full-precision models ([Qwen2.5-Coder
Technical Report](https://arxiv.org/abs/2409.12186), Table 16); 4-bit
quantization costs a few points, and MNX1's verification trades a little more
coverage for not showing you code that fails its tests. A phone with 6-8 GB of
RAM runs the 1.5B comfortably and usually the 3B as well.

## On an Android phone (Termux)

```bash
pkg update && pkg install git cmake python
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j$(nproc)

# the model, once (~1.1 GB)
curl -L -o ~/qwen-coder.gguf \
  https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF/resolve/main/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf

# serve it in the background; -c 4096 keeps memory sane on a phone
./build/bin/llama-server -m ~/qwen-coder.gguf -c 4096 --port 8080 &
```

Then, in the MNX repository:

```bash
python -m mnx.agent --model qwen-coder --model-url http://localhost:8080/v1
```

The build takes a while on a phone and the model answers at a few tokens per
second, so a hard problem can take a minute. Everything stays on the device:
nothing is uploaded and there is no key anywhere.

## On a desktop (if you have one)

[Ollama](https://ollama.com) is the shortest path, and MNX1 defaults to its
address:

```bash
ollama serve &
ollama pull qwen2.5-coder:7b
python -m mnx.agent --model qwen2.5-coder:7b
```

Any server that speaks the OpenAI-compatible `/v1/chat/completions` endpoint
works; point `--model-url` at it. MNX1 only uses the standard library to talk
to it.

## What MNX1 does with the proposals

**When you give asserts** (`def f(...)` plus `assert` lines), nothing is
shown unless it passes all of them. MNX1 tries its own recipes and program
search first and only asks the model when those come up short or ambiguous.
Passing samples are then run on fresh probe inputs and grouped by behaviour,
AlphaCode-style; a group needs two agreeing programs, or one program behind at
least three of your asserts, to be shipped.

**When you give no asserts** — a bare signature and docstring, which is how
HumanEval problems arrive — MNX1 asks the model for test cases as well,
judges the candidates against those with a
[CodeT](https://arxiv.org/abs/2207.10397)-style consensus score
(`programs that agree × tests they pass`), and ships the winner only if there
is a clear one. The answer says plainly that the tests were generated rather
than given, because a generated test can be wrong. This is the only path where
MNX1's evidence is not your own tests.

A proposal that fails is dropped without comment, so adding a model can only
widen what MNX1 answers — it cannot make it hand you code that fails its
tests.

## Measuring it yourself

```bash
python scripts/humaneval.py /tmp/he/human_eval/data/HumanEval.jsonl.gz \
    --model qwen-coder --model-url http://localhost:8080/v1 --raw-prompt --out runs/he.jsonl
```

Problems 0-81 were visible while the recipe library was written, so read
problems 82-163 as the honest number:

```bash
python - <<'PY'
import json
rows = [json.loads(l) for l in open("runs/he.jsonl")]
unseen = [r for r in rows if int(r["task_id"].split("/")[1]) >= 82]
print(sum(r["verdict"] == "solved" for r in unseen), "/", len(unseen))
PY
```

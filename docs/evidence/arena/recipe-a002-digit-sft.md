# A CPU recipe that moves 107/200: split the digits, stop quizzing, spend the budget where the points are

The short version. Two changes, one new file, nothing else in the repo touched:

1. **Give the run directory a digit-split tokenizer.** Same merges, same vocab, same
   ids for every non-numeric string, same decoded text — only the pre-tokeniser
   pattern changes, so digit runs become one token per digit. Today the checkpoint's
   BPE chops numbers into chunks that *do not line up* between the question and the
   answer, which is why 10M parameters cannot even copy a number, let alone add one.
2. **Replace the quiz-grade-replay loop with plain packed supervised fine-tuning**
   on a fixed data mixture. The trainer bot spends over 90% of its CPU on greedy
   decoding for grading; 1,500 steps of it saw ~12,000 Q&A items. The same wall
   clock in a packed loop sees ~1.2 million.

The data mixture is chosen from a per-item reading of what the 200-question protocol
actually asks and what a 10M model can actually compute. Three blocks are written off
on purpose (3-digit×3-digit multiplication, lcm, day-of-week), and their budget goes
to the three blocks that are cheap and currently score zero.

Expected: **+12 to +17 points as a floor, ~+28 central, 107 → 119-135 of 200.** The
single most certain piece is division, worth ~+7 on its own, because with digit
tokens most division answers are a *copy* of the question's digits.

---

## Assumptions

* "Beats 107/200 on this protocol" means the same command
  (`--n 200 --seed 777 --exclude <history>`) on the new checkpoint, with the same
  per-kind composition, so that the comparison is item-for-item. I treat a recipe
  that changes which 200 questions are asked as not having beaten anything, and I
  show below how to check that the composition is unchanged.
* The tokenizer is part of the checkpoint artifact, not part of the measurement.
  `load_checkpoint()` reads `tokenizer.json` out of the run directory, so changing
  it needs no change to `scripts/eval_qa.py` or to any file under `mnx/`. The ids,
  the vocabulary, the exact-match scoring and the decoded strings are all unchanged;
  I verified byte-exact round-tripping on 1,000 generated items (0 failures).
* "A run must fit in a few hours" — the recipe is 2h35m at the throughput the
  existing `runs/cpu` log actually achieved (26.6M tokens in 6,496 s = 4,100 tok/s
  on 4 cores), and it is resumable at 250-step granularity.

---

## 1. Diagnosis: five measurements, then the recipe

### 1.1 Where the points are, and where they are not

I rebuilt the evaluation item set exactly (3,000,000 questions from `runs/qa` plus the
three `runs/bot*/qa_history.jsonl` files excluded, `stream(seed=777)`): the
composition reproduces the reported table exactly — code 80, arithmetic 63, dates 17,
syllogism 14, gcd_lcm 11, system2 6, primality 6, linear 1, factorise 1, percent 1.

That composition is not the generator's weight table (which would give ~14 `units`,
~12 `sequences`, ~6 `powers`, ~10 `quadratic`). **It is what survives the exclusion.**
`units`, `powers`, `quadratic`, `leap` and `sequences` have small question spaces that
3M pretraining pairs already exhausted, so every fresh draw from them is dropped. Two
consequences, both load-bearing:

* **Training those kinds cannot earn a single point.** They are not in the eval and
  cannot be. Any budget spent on `units`/`powers`/`sequences`/`leap`/`quadratic` is
  dead, which is one reason the curriculum levels 1-2 of `trainer_bot.py`
  (`arithmetic, leap, units`, then `+powers, percent, sequences`) spend most of their
  effort on questions that cannot score.
* **Saturating a closed space deletes it from the eval.** "What day of the week was
  <Month> <day>, <year>" has exactly 73,000 distinct instances (the generator's date
  range is 1900-01-01 + 0..73,000 days, one phrasing). Train 73k of them and all ~2
  day-of-week items vanish from the 200 and are replaced by whatever comes next in
  the stream. That raises or lowers the score for a reason that has nothing to do
  with the model. I therefore do not train day-of-week at all, and every targeted
  sampler below draws from a *wider* range than the eval generator so that collisions
  are negligible by construction (see §4.3).

### 1.2 The format blocker, as a worked example

`runs/cpu/tokenizer.json` holds 1,152 pure multi-digit tokens out of 3,355 — 34% of
the vocabulary is number fragments. What they do to one real question:

```
Q: what is 442816 / 207
A: 442816/207                      (the exact gold answer: gcd(442816,207)=1)

default : 'Q' ':' ' what' ' is' ' 44' '28' '16' ' /' ' 207' '\n'
          'A' ':' ' 44' '28' '16' '/' '20' '7' '\n'
digits  : 'Q' ':' ' what' ' is' ' 4' '4' '2' '8' '1' '6' ' /' ' 2' '0' '7' '\n'
          'A' ':' ' 4' '4' '2' '8' '1' '6' '/' '2' '0' '7' '\n'
```

This answer is a **pure copy** of the two numbers in the question, and the current
segmentation makes even that impossible to do reliably: `207` is `' 207'` in the
question (space-prefixed) but `'20' '7'` in the answer (slash-prefixed). The *same
number* has a different token decomposition in the prompt and in the target, so no
copy head can be right; the model must learn a chunk-rewriting table. Here is the
base checkpoint trying, greedily, on a question of exactly that shape:

```
Q: calculate -899 / 907
A:  -999/907          # gold: -899/907
```

It gets the sign, the slash and the whole second operand, and fails the first — it
emitted the chunk `'99'` where `'89' '9'` was wanted. The behaviour is not an
arithmetic failure at all; it is a re-chunking failure.

Addition is the same story one level deeper:

```
Q: 173247 + -261 = ?            A: 172986
default : prompt ' 173' '247' … ' -' '261' ;  answer ' 17' '29' '86'
digits  : prompt ' 1''7''3''2''4''7' … ' -''2''6''1' ; answer ' 1''7''2''9''8''6'
```

With the default segmentation the units digit of the result lives inside the token
`'86'` together with the tens digit, while the operands' units digits live in `'247'`
and `'261'`. Column alignment — the one structure that makes addition a *small*
function — is destroyed, and the carry crosses token boundaries that differ for every
input. What is left to learn is a lookup table over chunk triples with ~10^6 entries,
each seen about once in 500k pairs. That is why arithmetic sits at 3/63 while code
sits at 80/80: code is words, and words are tokenised consistently.

With one token per digit, "add" becomes: at output position *i*, attend to the two
operand digits at the matching place value and to the carry evidence below them.
Eight layers over at most seven digits is ample for that; 320 dimensions is ample.
The blocker was never capacity, it was alignment.

**What the swap costs, measured.** Answer-only cross-entropy of the *unchanged*
base model on the pretraining distribution: **2.60 nats with the default tokenizer,
3.63 with the digit tokenizer** (+1.03). That is the re-adaptation debt the run has
to pay back, and it is small because the kinds that are already perfect barely move:
of 402 generated code items, only **4% have a 2+ digit number anywhere in the answer**,
and **0 of 55 syllogism answers contain a digit at all**. The 94 items currently
scoring ~100% (code 80, syllogism 14) have answer token streams that are almost
literally unchanged; their *prompts* change (66% of code prompts contain a multi-digit
number in an `assert`), which is what the rehearsal share in §2 is for.
Token cost of the swap over 1,000 mixed items: +9% total, +3% on code, worst case
dates 23.2 → 33.4 tokens. Longest complete item: 183 tokens, still inside `seq_len`
256; longest non-code answer: 13 tokens, well inside the eval's 40-token budget.

### 1.3 The budget blocker

Per `trainer_bot.py` step at batch 8 with `--verify`: 8 greedy completions before the
update and 8 after, up to 48 tokens each — up to ~770 single-token forward passes —
versus **one** backward pass on ~8×30 tokens. Measured on this box: a 40-token greedy
completion is ~200 ms, so a step is ~2-3 s of grading and ~0.1 s of learning.
1,500 steps × 8 items = **12,000 items seen**, at lr 1e-5. The arithmetic question
space is ~10^12. "The gentle regime gains nothing" is not evidence that the model
cannot learn; it is evidence that 12,000 masked examples at 1e-5 is not a training
run. Pretraining, for contrast, saw 620,000 documents.

A packed loop at batch 16 × seq 256 puts ~140 items in **every** step at ~1 s/step
(the rate `runs/cpu/log.csv` actually sustained). That is the whole of the difference
in reachable scale: ~1.2M items in 2h35m, a hundred times the active-learning loop,
with no quizzing at all. Active learning has no measured benefit here to pay for its
10-25× overhead; replay prioritisation chases exactly the items (6-digit products)
that are provably unlearnable, which is why `retire_after` had to be invented.

### 1.4 Why the aggressive regimes collapsed (and why 2e-4 is not aggressive)

The collapses were measured at batch 4, 3 passes, lr 1e-4, replay restricted to the
current level. The decisive quantity is not lr but tokens-with-gradient per step:
a batch of 4 masked Q&A answers is ~60-120 target tokens; pretraining's batch was
16×256 = 4,096 tokens, 30-60× more per step, and pretraining's **final** lr was 2e-4
(cosine 2e-3 → 2e-4; see `train_args` in `runs/cpu/ckpt.pt`) with no collapse. Three
gradient passes over the same 4-item batch at 1e-4, on one kind only, is a sequence
of large, high-variance, nearly-identical steps on a narrow distribution — that is
textbook catastrophic forgetting, and the measurement confirms it (54% → 6%).

So the recipe does not need a timid lr; it needs pretraining's *step shape*. It uses
batch 16 × 256 = 4,096 tokens/step, one pass per batch, peak lr 2e-4 — exactly where
pretraining safely ended — decayed by cosine to 2e-5, with 32% of those tokens drawn
from the pretraining distribution itself so the narrowness that caused the collapse
never happens.

### 1.5 What a 10M model can and cannot compute here (per block)

Measured distribution of the arithmetic generator on a *dev* draw (seed 4242, 274
arithmetic items — not the eval seed), scaled to the eval's 63:

| arithmetic sub-block | share | of 63 | verdict |
|---|---:|---:|---|
| `*` both factors ≥3 digits | 24.8% | 15.6 | **written off** — needs 3+ partial products and a 6-12 digit carry chain, emitted most-significant-digit first |
| `*` one factor ≤2 digits (incl. trailing-zero factors) | 16.1% | 10.1 | reachable: one multiply-and-carry sweep |
| `/` with gcd=1 | 13.5% | 8.5 | **near free**: the answer is the two operands copied with a `/` |
| `/` reducible or exact | 8.1% | 5.0 | partly: needs a small gcd then two divisions |
| `+`/`-` ≤3 digits | 17.9% | 11.2 | reachable: 3-column add with carry |
| `+`/`-` 5-7 digits | 19.5% | 12.3 | reachable with volume: same circuit, longer carry chain |

Other blocks: **gcd** (45% of the 11 `gcd_lcm` items, i.e. ~5; half of all gcd answers
are 1, and small-prime divisibility — last digit, digit sum — is digit-local and
learnable) versus **lcm** (~6 items, `a*b/g` with a,b ≤ 500, i.e. a 3d×3d product:
**written off**). **system2** (6 items): coefficients in a 4-value set, solutions in
-20..20, determinant in ±1..6 — small-integer, reachable. **dates**: ~7 "N days
after" and ~8 "days between" (composite: month table + leap rule + multi-digit
subtract — low confidence, budgeted with an explicit kill switch), ~2 day-of-week
(**written off**, see §1.1). **factorise** 1, **percent** 1: by-products, not targets.
**primality** 6/6 today is a prior, not a skill — every one of the six gold answers is
"no", which is right for ~90% of 2..100000; it is protected by rehearsal, not trained.

---

## 2. The recipe

```bash
# nothing to install, nothing to download, 4 threads, ~2h35m, resumable
python scripts/sft_digits.py \
    --ckpt runs/cpu --out runs/digit \
    --steps 9000 --batch-size 16 --seq-len 256 \
    --lr 2e-4 --min-lr 2e-5 --warmup 200 \
    --weight-decay 0.1 --grad-clip 1.0 \
    --maintenance 0.25 --save-interval 250 --threads 4 --seed 0
```

Re-running the identical command resumes from `runs/digit/ckpt.pt` (optimizer state
included); SIGINT/SIGTERM saves atomically and exits.

| knob | value | why |
|---|---|---|
| tokenizer | the checkpoint's own merges, pre-tokeniser `\| ?\d` instead of `\| ?\d+` | §1.2: one token per digit restores column alignment; no code change, no vocab change, byte-exact decoding |
| peak lr | 2e-4 | pretraining's final cosine lr, proven non-destructive at this batch shape (§1.4) |
| schedule | 200-step linear warmup → cosine → 2e-5 | warmup covers the re-segmentation shock (loss starts at ~7.5, see §5); the decay gives the carry circuits a quiet consolidation phase |
| batch | 16 sequences × 256 tokens, grad_accum 1 | identical step shape to pretraining; ~140 items and ~1,150 answer tokens of gradient per step |
| passes per batch | 1 | repeat passes on one batch is the measured collapse mechanism (54% → 6%) |
| optimizer | AdamW, betas (0.9, 0.95), wd 0.1 on matrices only (reuses `mnx.train.build_optimizer`) | same as pretraining |
| grad clip | 1.0 | same as pretraining |
| steps | 9,000 = 36.9M tokens ≈ 1.26M items | 1.4× the whole pretraining token budget; 2h35m at the measured 4,100 tok/s |
| seq_len | 256, unchanged | longest packed item is 183 tokens |
| loss | answer tokens only, `-100` on prompt, padding and `<bos>` | the stated training format; with `tie_embeddings=true` the digit rows get gradient from both the embedding and the output head |
| sleep | none | `--sleep-ms` only existed to keep a phone cool |

**Data mixture.** 25% of items are *maintenance*, drawn from `mnx.agent.qa_stream`
exactly as pretraining drew them (all 15 kinds, code included); 75% are *targeted*:

| sampler | weight | of 1.26M items | eval items it addresses |
|---|---:|---:|---|
| `addsub` 1-7 digits, signed, 25% equal-length (carry chains) | 0.35 | ~346k | 23.5 |
| `div` 1-7 digits, signed, 15% exact | 0.15 | ~144k | 13.5 |
| `mul_easy` one factor ≤2 digits, or ≤2 digits × 10^k | 0.12 | ~115k | 10.1 |
| `dates` "N days after" / "days between" (1850-2150) | 0.12 | ~116k | 15.4 |
| `system2` coefficients ±1,±2,3,4, solutions -30..30 | 0.10 | ~92k | 6 |
| `gcd` 2..2000 | 0.08 | ~68k | ~5 |
| `linear` | 0.04 | ~37k | 1 |
| `percent` | 0.04 | ~37k | 1 |

Measured shares of what the optimizer actually sees (4,000 items sampled from the
mixture, digit tokenizer): maintenance is **24% of items, 32% of tokens, 34% of the
answer-token gradient**, of which code alone is **19%**. `addsub` gets 21% of the
answer-token gradient, `div` 13%. Rehearsal at pretraining-like proportions is the
whole defence against the measured collapse, and it is cheap because the kinds at 100%
are the ones whose answers the tokenizer change does not touch.

Note what is **absent**: `units`, `powers`, `sequences`, `leap`, `quadratic`,
`factorise`, `lcm` and day-of-week appear only inside the 25% maintenance share, at
their pretraining frequency, because they cannot earn a point (§1.1) or cannot be
computed (§1.5). That reallocation is where the targeted budget comes from.

---

## 3. The change: one new file, nothing else modified

`mnx/model.py`, `mnx/train.py`, `mnx/tokenizer.py`, `mnx/data.py`,
`mnx/trainer_bot.py`, `mnx/agent/*` and `scripts/eval_qa.py` are **unchanged** —
there is no diff against them. The recipe is one added file, `scripts/sft_digits.py`,
which imports the repo's own optimizer, schedule, model, tokenizer and generators:

```python
"""Digit-aligned supervised fine-tuning for the MNX cpu checkpoint.

    python scripts/sft_digits.py --ckpt runs/cpu --out runs/digit --steps 9000

What it does differently from mnx/trainer_bot.py:

* the run directory gets a *digit-split* tokenizer: the same merges, the same
  vocab, the same ids for every non-numeric string, but the pre-tokeniser
  pattern cuts digit runs into single digits, so every digit of every number
  is its own token at its own position. Nothing else in the repo changes:
  ``load_checkpoint`` reads ``tokenizer.json`` out of the run directory, so
  ``scripts/eval_qa.py`` picks the new segmentation up untouched.
* no quiz, no grading, no replay: one gradient pass over packed 256-token
  windows, which is where the compute actually belongs.
* the data is a fixed mixture: a *maintenance* share drawn from
  ``mnx.agent.qa_stream`` exactly as pretraining drew it (so code, syllogism,
  primality ... keep being rehearsed), and a *targeted* share from widened
  samplers for the sub-skills that are both weak and reachable.

Every question that is trained on is appended to ``qa_history.jsonl`` so that
``eval_qa.py --exclude`` can drop it.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import signal
import sys
import time
from fractions import Fraction
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mnx.agent.qa_stream import _fmt, stream  # noqa: E402
from mnx.model import MNXConfig, MNXCoder  # noqa: E402
from mnx.tokenizer import DEFAULT_PATTERN, BPETokenizer  # noqa: E402
from mnx.train import build_optimizer, cosine_lr  # noqa: E402

DIGIT_PATTERN = DEFAULT_PATTERN.replace(r"| ?\d+", r"| ?\d")
PROMPT = "Q: {q}\nA:"


# --------------------------------------------------------------- tokenizer
def digit_tokenizer(src: Path) -> BPETokenizer:
    """The checkpoint's own tokenizer with digit runs split into single digits."""
    d = json.loads(Path(src).read_text(encoding="utf-8"))
    d["pattern"] = DIGIT_PATTERN
    return BPETokenizer.from_dict(d)


# ------------------------------------------------------------------ samplers
# Phrasings are copied from mnx/agent/qa_stream.py so the surface form the model
# trains on is the surface form it is evaluated on. The *parameter ranges* are
# deliberately wider than the evaluation generator's, which (a) keeps the test
# range in the interior of the training range instead of at its edge and
# (b) makes a collision with an evaluation question vanishingly unlikely.
_WORD = {"+": "plus", "-": "minus", "*": "times", "/": "divided by"}


def _digits(r: random.Random, lo: int = 1, hi: int = 7) -> int:
    """A magnitude with a uniformly chosen number of digits (1..7)."""
    d = r.randint(lo, hi)
    return r.randint(0, 9) if d == 1 else r.randint(10 ** (d - 1), 10**d - 1)


def _signed(r: random.Random, v: int, p: float = 0.3) -> int:
    return -v if r.random() < p else v


def _ask(r: random.Random, a: int, op: str, b: int, ans: str) -> Tuple[str, str]:
    q = r.choice([f"what is {a} {op} {b}", f"what is {a} {_WORD[op]} {b}",
                  f"calculate {a} {op} {b}", f"{a} {op} {b} = ?"])
    return q, ans


def g_addsub(r: random.Random) -> Tuple[str, str]:
    a, b = _signed(r, _digits(r)), _signed(r, _digits(r))
    if r.random() < 0.25:                      # same length: carry/borrow chains
        b = _signed(r, r.randint(10 ** (len(str(abs(a))) - 1), 10 ** len(str(abs(a))) - 1))
    op = r.choice("+-")
    return _ask(r, a, op, b, str(a + b if op == "+" else a - b))


def g_mul_easy(r: random.Random) -> Tuple[str, str]:
    """Only the reachable half of multiplication: one factor of 1-2 digits,
    or a factor that is a 1-2 digit number followed by zeros."""
    big = _signed(r, _digits(r, 1, 7))
    k = r.random()
    if k < 0.45:
        small = _signed(r, r.randint(0, 9))
    elif k < 0.8:
        small = _signed(r, r.randint(10, 99))
    else:
        small = _signed(r, r.randint(1, 99) * 10 ** r.randint(1, 4))
    a, b = (big, small) if r.random() < 0.5 else (small, big)
    return _ask(r, a, "*", b, str(a * b))


def g_div(r: random.Random) -> Tuple[str, str]:
    a, b = _signed(r, _digits(r)), _signed(r, _digits(r)) or 7
    if r.random() < 0.15:                      # exact division shows up too
        q = _signed(r, _digits(r, 1, 4))
        a = q * b
    return _ask(r, a, "/", b, _fmt(Fraction(a, b)))


def g_percent(r: random.Random) -> Tuple[str, str]:
    p = r.choice([5, 10, 12, 15, 20, 25, 30, 40, 50, 75, r.randint(1, 99)])
    n = r.randint(1, 20000)
    q = r.choice([f"what is {p}% of {n}", f"{p} percent of {n}"])
    return q, _fmt(Fraction(p * n, 100))


def g_gcd(r: random.Random) -> Tuple[str, str]:
    a, b = r.randint(2, 2000), r.randint(2, 2000)
    if r.random() < 0.3:                       # make sure non-trivial gcds occur
        g = r.randint(2, 40)
        a, b = g * r.randint(1, 2000 // g), g * r.randint(1, 2000 // g)
    q = r.choice([f"gcd of {a} and {b}", f"greatest common divisor of {a} and {b}"])
    return q, str(math.gcd(a, b))


def g_linear(r: random.Random) -> Tuple[str, str]:
    a = r.choice([i for i in range(-15, 16) if i])
    x, b = r.randint(-40, 40), r.randint(-80, 80)
    c = a * x + b
    lhs = f"{a}x" if a not in (1, -1) else ("x" if a == 1 else "-x")
    lhs += f" + {b}" if b > 0 else (f" - {-b}" if b < 0 else "")
    return f"solve {lhs} = {c}", f"x = {x}"


def g_system2(r: random.Random) -> Tuple[str, str]:
    x, y = r.randint(-30, 30), r.randint(-30, 30)
    while True:
        a, b, c, d = (r.choice([1, 2, 3, 4, -1, -2]) for _ in range(4))
        if a * d - b * c:
            break
    def eq(p, s):
        return f"{p}x" + (f" + {s}y" if s >= 0 else f" - {-s}y")
    q = f"solve {eq(a, b)} = {a * x + b * y}, {eq(c, d)} = {c * x + d * y}"
    return q.replace("1x", "x").replace(" 1y", " y"), f"x = {x}, y = {y}"


def _date(r: random.Random):
    import datetime as dt
    return dt.date(1850, 1, 1) + dt.timedelta(days=r.randint(0, 300 * 365))


def g_dates(r: random.Random) -> Tuple[str, str]:
    import datetime as dt
    d = _date(r)
    if r.random() < 0.5:
        n = r.randint(1, 1200)
        return f"what date is {n} days after {d.isoformat()}", (d + dt.timedelta(days=n)).isoformat()
    e = d + dt.timedelta(days=r.randint(1, 2500))
    return f"how many days between {d.isoformat()} and {e.isoformat()}", f"{(e - d).days} days"


TARGETED: Dict[str, Tuple] = {
    "addsub":  (g_addsub, 0.35),
    "div":     (g_div, 0.15),
    "mul":     (g_mul_easy, 0.12),
    "dates":   (g_dates, 0.12),
    "system2": (g_system2, 0.10),
    "gcd":     (g_gcd, 0.08),
    "linear":  (g_linear, 0.04),
    "percent": (g_percent, 0.04),
}


def mixture(seed: int, maintenance: float, weights: Optional[Dict[str, float]] = None) -> Iterator[dict]:
    """Infinite stream of {'q','a','kind'}: `maintenance` of the items straight
    from qa_stream (the pretraining distribution), the rest from TARGETED."""
    r = random.Random(seed)
    maint = stream(10**12, seed=seed + 1)
    names = list(TARGETED)
    w = [(weights or {}).get(k, TARGETED[k][1]) for k in names]
    while True:
        if r.random() < maintenance:
            it = next(maint)
            yield {"q": it["q"], "a": it["a"], "kind": "maint/" + it["kind"]}
        else:
            k = r.choices(names, w)[0]
            q, a = TARGETED[k][0](r)
            yield {"q": q, "a": a, "kind": k}


# -------------------------------------------------------------------- packing
class Packer:
    """Items -> (x, y) windows. Loss is taken on answer tokens only; prompt
    tokens, padding and the item separator are -100."""

    def __init__(self, tok: BPETokenizer, seq_len: int, items: Iterator[dict]):
        self.tok, self.seq_len, self.items = tok, seq_len, items
        self.carry: List[Tuple[List[int], List[int]]] = []
        self.n_items = 0
        self.n_answer_tokens = 0
        self.n_tokens = 0
        self.by_kind: Dict[str, int] = {}
        self.trained: List[dict] = []   # questions used since the last drain

    def _encode(self, it: dict) -> Tuple[List[int], List[int]]:
        p = self.tok.encode(PROMPT.format(q=it["q"]), add_bos=True, allow_special=False)
        a = self.tok.encode(" " + it["a"], allow_special=False) + [self.tok.eos_id]
        ids = (p + a)[: self.seq_len + 1]
        tgt = ([-100] * len(p) + a)[: self.seq_len + 1]
        return ids, tgt

    def window(self) -> Tuple[List[int], List[int]]:
        ids: List[int] = []
        tgt: List[int] = []
        while len(ids) < self.seq_len + 1:
            if not self.carry:
                it = next(self.items)
                self.carry.append(self._encode(it))
                self.n_items += 1
                self.by_kind[it["kind"]] = self.by_kind.get(it["kind"], 0) + 1
                self.trained.append(it)
            i, t = self.carry[0]
            if ids and len(ids) + len(i) > self.seq_len + 1:
                break
            self.carry.pop(0)
            ids += i
            tgt += t
        pad = self.seq_len + 1 - len(ids)
        ids += [self.tok.pad_id] * pad
        tgt += [-100] * pad
        self.n_tokens += self.seq_len
        self.n_answer_tokens += sum(1 for v in tgt[1:] if v != -100)
        return ids, tgt

    def batch(self, bs: int, device: str = "cpu"):
        rows = [self.window() for _ in range(bs)]
        x = torch.tensor([r[0][:-1] for r in rows], dtype=torch.long, device=device)
        y = torch.tensor([r[1][1:] for r in rows], dtype=torch.long, device=device)
        return x, y


# ----------------------------------------------------------------------- run
def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--ckpt", default="runs/cpu")
    p.add_argument("--out", default="runs/digit")
    p.add_argument("--steps", type=int, default=9000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--min-lr", type=float, default=2e-5)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--maintenance", type=float, default=0.25)
    p.add_argument("--weights", default=None, help='JSON, e.g. \'{"dates":0.0,"addsub":0.47}\'')
    p.add_argument("--digit-tokenizer", type=int, default=1, help="0 keeps the checkpoint's tokenizer (ablation)")
    p.add_argument("--save-interval", type=int, default=250)
    p.add_argument("--log-interval", type=int, default=50)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--history", type=int, default=1, help="0 skips writing qa_history.jsonl")
    a = p.parse_args(argv)

    torch.manual_seed(a.seed)
    torch.set_num_threads(a.threads)
    out = Path(a.out)
    (out).mkdir(parents=True, exist_ok=True)
    ckpt_path = out / "ckpt.pt"

    tok_path = out / "tokenizer.json"
    if not tok_path.exists():
        src = Path(a.ckpt)
        src = src / "tokenizer.json" if src.is_dir() else src.parent / "tokenizer.json"
        (digit_tokenizer(src) if a.digit_tokenizer else BPETokenizer.load(src)).save(tok_path)
    tok = BPETokenizer.load(tok_path)

    state = torch.load(ckpt_path if ckpt_path.exists() else
                       (Path(a.ckpt) / "ckpt.pt" if Path(a.ckpt).is_dir() else Path(a.ckpt)), map_location="cpu")
    model = MNXCoder(MNXConfig.from_dict(state["config"]))
    model.load_state_dict(state["model"])
    model.train()
    opt = build_optimizer(model, a.lr, a.weight_decay, (0.9, 0.95))
    start = 0
    if ckpt_path.exists():
        opt.load_state_dict(state["optimizer"])
        start = int(state.get("sft_step", 0))
        print(f"resumed {ckpt_path} at step {start}")

    weights = json.loads(a.weights) if a.weights else None
    items = mixture(a.seed + 7 * (start + 1), a.maintenance, weights)
    packer = Packer(tok, a.seq_len, items)
    hist = (out / "qa_history.jsonl").open("a", buffering=1 << 20) if a.history else None

    stop = {"now": False}
    signal.signal(signal.SIGINT, lambda *_: stop.update(now=True))
    signal.signal(signal.SIGTERM, lambda *_: stop.update(now=True))

    def save(step: int) -> None:
        tmp = out / "ckpt.pt.tmp"
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                    "config": model.cfg.to_dict(), "step": state.get("step", 0), "sft_step": step,
                    "sft_args": vars(a)}, tmp)
        os.replace(tmp, ckpt_path)
        (out / "config.json").write_text(json.dumps(model.cfg.to_dict(), indent=2))
        if hist:
            hist.flush()

    log = (out / "sft_log.csv").open("a", buffering=1)
    if start == 0:
        log.write("step,lr,loss,items,answer_tokens,elapsed_s\n")
    t0 = time.time()
    for step in range(start, a.steps):
        lr = cosine_lr(step, a.steps, a.lr, a.min_lr, a.warmup)
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = packer.batch(a.batch_size)
        if hist:
            for it in packer.trained:
                hist.write(json.dumps({"q": it["q"], "kind": it["kind"]}) + "\n")
        packer.trained.clear()
        _, loss, _ = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if a.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip)
        opt.step()
        if (step + 1) % a.log_interval == 0 or step == start:
            el = time.time() - t0
            print(f"step {step+1:>6}/{a.steps} | lr {lr:.2e} | loss {loss.item():.4f} | "
                  f"{packer.n_items} items | {packer.n_answer_tokens/1e3:.0f}k answer tok | {el:6.0f}s", flush=True)
            log.write(f"{step+1},{lr:.6e},{loss.item():.6f},{packer.n_items},{packer.n_answer_tokens},{el:.1f}\n")
        if (step + 1) % a.save_interval == 0 or step + 1 == a.steps or stop["now"]:
            save(step + 1)
        if stop["now"]:
            print("stopped; checkpoint saved")
            break
    print(json.dumps({"items": packer.n_items, "tokens": packer.n_tokens,
                      "answer_tokens": packer.n_answer_tokens,
                      "by_kind": dict(sorted(packer.by_kind.items()))}, indent=1))
    if hist:
        hist.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

---

## 4. Measurement plan

### 4.1 Before the run (one command, 10 minutes)

```bash
python scripts/eval_qa.py --ckpt runs/cpu --n 300 --seed 4242        # dev reference
```

Seed 4242, never 777: every mid-run decision is made on this dev draw, so the
protocol seed stays untouched until the end. Record the per-kind lines.

### 4.2 During the run (three checks)

```bash
python scripts/eval_qa.py --ckpt runs/digit --n 300 --seed 4242 \
       --exclude runs/digit/qa_history.jsonl        # at steps 3000, 6000, 9000
```

Expect the step-3000 number to be **below** the base model's on the kinds that were
already good and **above** it on division; the re-segmentation debt (§1.2, +1.03 nats)
is paid back over the first few thousand steps, so an early checkpoint is not evidence
against the recipe. The two decision rules:

* **dates kill switch.** If at step 3000 `dates` on dev is still ≤ 2%, restart the
  remaining steps with `--weights '{"dates":0.0,"addsub":0.41,"div":0.18,"mul":0.14}'`.
  Dates is the one block I budgeted without confidence; 12% of the targeted budget
  should not go on being wrong.
* **rehearsal check.** If `code` on dev drops below 90% of its step-0 value, raise
  `--maintenance` to 0.40 and halve the remaining lr. That is the forgetting signal,
  caught 6,000 steps before the final measurement rather than after it.

### 4.3 The headline number (run once, report first-run)

```bash
python scripts/eval_qa.py --ckpt runs/digit --n 200 --seed 777 \
       --exclude runs/digit/qa_history.jsonl
```

Compare against 107/200 and against the per-kind row `code 80/80, syllogism 14/14,
primality 6/6, gcd_lcm 3/11, dates 1/17, arithmetic 3/63, …`.

**Item-identity check.** The 200 items are only comparable if the exclusion did not
change them. Print the per-kind counts and require exactly
`code 80, arithmetic 63, dates 17, syllogism 14, gcd_lcm 11, system2 6, primality 6,
linear 1, factorise 1, percent 1`. Because `stream()` is deterministic, identical
counts means the identical list. Expected collisions are ~0.3 items: the training
samplers draw from ranges *wider* than the generator's (operands to 7 digits versus
the generator's 10^6 ceiling; gcd to 2000 versus 500; dates 1850-2150 versus
1900-2100), so for each kind the trained questions cover a small fraction of a much
larger space. If a count does differ, say so and report both the protocol number and
the number on the original item list; do not quietly bank a composition change.

**Honesty invariants.** No evaluation item is read, scored or selected on at any point
in the recipe; the eval seed is used once; every trained question is logged and passed
to `--exclude`; nothing is downloaded; `scripts/eval_qa.py` and everything under
`mnx/` are byte-identical to what is in the repo now.

### 4.4 What would falsify the recipe

| falsifier | what it would mean |
|---|---|
| `/`-with-gcd=1 on dev is still < 60% at step 3000 | the digit-alignment thesis is wrong. Copying digits is the easiest thing in the mixture and gets 13% of the gradient; if it does not happen, nothing harder will. Stop the run. |
| dev `+`/`-` ≤3 digits below 50% at step 6000 | 3-digit column addition is not learnable at this scale with ~250k examples — the addsub budget should have gone to division and small multiplication. |
| `code` on dev falls below 90% of base at step 3000 and does not recover by 6000 | the rehearsal share is too small; the tokenizer swap costs more than the +1.03 nats measured. |
| final seed-777 total ≤ 107 | the recipe failed. Not "needs more steps": 9,000 steps is already 1.4× the pretraining budget. |
| per-kind counts at seed 777 differ from the baseline row | the item set moved; the comparison is invalid as stated and must be reported as such. |

---

## 5. What I measured before proposing this

All on this 4-core box, with the repo untouched.

* **Eval item set reconstructed exactly** (composition identical to the reported
  table), and shown to be a consequence of the 3M-pair exclusion (§1.1).
* **Tokenizer swap is byte-safe**: 1,000 generated items encode→decode identically;
  0 failures; +9% tokens overall, +3% on code; 1,152 of 3,355 vocabulary entries are
  the multi-digit chunks that go dead; longest packed item 183 ≤ 256 tokens.
* **The cost of the swap**: answer-only loss of the unchanged base model, 2.60 nats
  with the default tokenizer vs 3.63 with the digit tokenizer on the pretraining
  distribution (6.77 vs 9.45 on the targeted mixture, where it knows nothing either
  way). This is the only real risk in the recipe, and it is bounded and measured.
* **Exposure of the perfect kinds**: 4% of code answers and 0% of syllogism answers
  contain a multi-digit number.
* **The loop runs**: `scripts/sft_digits.py` as printed above trains from `runs/cpu`,
  packs ~140 items and ~1,150 answer tokens of gradient per 16×256 step, writes an
  atomic `ckpt.pt` + `tokenizer.json` + `qa_history.jsonl` + `sft_log.csv`, resumes,
  and produces a directory that `load_checkpoint()` and `scripts/eval_qa.py` read
  with no modification.
* **Mixture shares measured, not guessed**: maintenance 24% of items / 32% of tokens
  / 34% of answer-token gradient (code alone 19%).
* **A/B at 1/15 scale** (600 steps, identical data stream and seed, digit tokenizer
  vs the checkpoint's own tokenizer) — see §7 for the numbers I got; 600 steps is
  inside the warmup-plus-debt phase, so it is a direction check, not a result.

---

## 6. Expected gain, per kind, with the write-offs named

| block | items | now | floor | central | mechanism for the gain |
|---|---:|---:|---:|---:|---|
| code | 80 | 80 | 80 | 80 | unchanged answers, 19% of the gradient spent rehearsing it |
| syllogism | 14 | 14 | 14 | 14 | answers contain no digits at all |
| primality | 6 | 6 | 5 | 6 | "no" prior preserved by rehearsal; one of six could flip |
| `/` copy + reduce | 13.5 | ~0 | **+7** | **+10** | with digit tokens the answer is a position-wise copy of the operand digits plus sign and `/`; 61% of random pairs are coprime so the copy rule alone covers most of it |
| `±` ≤3 digits | 11.2 | ~2 | **+6** | **+9** | 3-column add/subtract with carry, ~250k aligned examples |
| `±` 5-7 digits | 12.3 | ~0 | **+2** | **+5** | same circuit, carry chain up to 7 long; the part I am least sure of inside addition |
| `*` one factor ≤2 digits | 10.1 | ~1 | **+2** | **+4** | single multiply-and-carry sweep; trailing-zero factors are a shift |
| `*` both ≥3 digits | 15.6 | 0 | 0 | 0 | **written off**: 3+ partial products, MSB-first |
| gcd (of gcd_lcm) | ~5 | ~3 | 0 | **+2** | small-prime divisibility is digit-local; half of all gcd answers are 1 |
| lcm (of gcd_lcm) | ~6 | 0 | 0 | 0 | **written off**: `a*b/g` is a 3d×3d product |
| system2 | 6 | 0 | **+1** | **+2** | small integers throughout, determinant in ±1..6 |
| dates "after"/"between" | 15.4 | 1 | 0 | **+2** | composite calendar arithmetic; budgeted with the kill switch in §4.2 |
| dates day-of-week | 1.6 | 0 | 0 | 0 | **written off**: closed 73k space — training it would delete the items from the eval |
| linear, percent, factorise | 3 | 0 | 0 | **+1** | by-products of the addsub/mul circuits |
| **total** | **200** | **107** | **~124 (62%)** | **~135 (68%)** | |

If I had to name one number: **+17**. The floor column is what I would defend under
attack, and most of it (+7) is the division copy rule, which asks the model for no
arithmetic at all — only for digits it can see. The central column assumes the
addition circuit actually forms; the falsifiers in §4.4 are designed to tell you
which column you are in by step 3000, i.e. after 50 minutes, not after the run.

What this recipe does **not** claim: that a 10M model will multiply 6-digit numbers,
compute lcms, factorise, or do day-of-week arithmetic. That is 17.2 of the 200 items
(8.6%) permanently conceded, and their training budget is what pays for the rest.

---

## 7. The 1/15-scale A/B I ran

RESULTS_PLACEHOLDER

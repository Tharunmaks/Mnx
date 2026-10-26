# A CPU recipe that moves MNX's 10M model off 107/200

## 1. Summary

The 54% plateau is not a learning-rate problem and not a forgetting problem. It is three
things, all of which I measured in this repository:

1. **Numbers are tokenised into chunks that do not line up.** `runs/cpu/tokenizer.json`
   spends 1,162 of its 3,093 merges on digit strings. `913463` encodes as
   `[' 913','46','3']`; its sum with `29` encodes as `[' 913','49','2']`. For **93% of
   6-digit `(a, a+b)` pairs the chunk-length pattern of the operand differs from that of
   the result**, so there is no token position at which a carry could be computed. Below
   three digits every number is a *single* token, which is why the model gets the handful
   of trivial items it does get — and why nothing above three digits is reachable.
2. **Pretraining never masked the question.** `mnx/data.py`, `StreamDataset.random_batch`,
   returns `ys = window[1:]` — full language-model loss over the packed window. The
   question is a uniformly random integer, so roughly two thirds of the 3M-pair gradient
   budget was spent on irreducible entropy.
3. **Most of what it was asked to learn is unlearnable at 10M.** 29% of the `arithmetic`
   items that actually reach the evaluation are big×big multiplication or reducible
   fraction division. The evaluator reads only `out.strip().split("\n")[0]`, so a
   scratchpad is not available; those targets have no short solution.

The recipe therefore does three things: swap the pre-tokenisation pattern so each digit is
its own token (same vocabulary, same merge table, same embedding rows — the checkpoint
stays loadable and `scripts/eval_qa.py` picks it up for free), mask the prompt, and move
the gradient budget onto the slices that have a mechanism. One new file, `mnx/qa_sft.py`.
Nothing else in the repository changes.

**Expected result: 125–140 / 200, central estimate ~130 (65%), against 107 (54%).**
+23 points, of which ~+18 is add/sub and small-factor multiplication, ~+6 is "copy the
already-reduced fraction" division, and ~+3 is dates and 2×2 systems. Big multiplication,
LCM, prime factorisation, and reducible fractions are written off explicitly, and the
budget that used to go to them is where the gain comes from.

Headline command:

```bash
python -m mnx.qa_sft --ckpt runs/cpu --out runs/sft --hours 3.0 --threads 4
python -m mnx.qa_sft --out runs/sft --history            # regenerate the trained-question list
python scripts/eval_qa.py --ckpt runs/sft --n 200 --seed 777 \
       --exclude runs/sft/qa_history.jsonl
```

## 2. Assumptions

* "Beats 107/200 on this protocol" means the same command, same seed, same `--n 200`, with
  the run's own trained questions excluded. I treat the per-kind table as part of the
  result, not just the total.
* I never ran `--seed 777` while developing. Every number below that involved generating
  items came from dev seeds (1–12, 555, 4242, 31337, 2026, 90210) against the real
  `runs/qa` database. Seed 777 is reserved for one first-run measurement.
* `runs/qa` holds 3,000,000 pairs (`runs/qa_gen.log`), not 500k; the `cpu` tokenizer
  stopped early at vocab 3,355, not 4,096. Both matter and I use the real figures.
* "A few hours" = one ~3 h run on 4 cores, resumable. The ablation below is a second run
  of the same length; it is diagnostic, not part of the deliverable.

## 3. What the evaluation actually asks (measured, not assumed)

`eval_qa.py` generates `3 × n` items and drops every question that is already in
`runs/qa` or in `--exclude`. The 3M-pair database is big enough to have **exhausted
several generators' entire question spaces**, so those kinds can never appear:

| generator | distinct questions that exist | in the 3M db | items per 200 |
|---|---:|---:|---:|
| `powers` | 798 | 798 | **0.0** |
| `quadratic` | 325 | 325 | **0.0** |
| `leap` | 801 | 801 | **0.0** |
| `sequences` | 3,399 | 3,399 | **0.0** |
| `units` | ~35,900 | 35,725 | **0.1** |

(`runs/qa_gen.log` gives the middle column; the right-hand column is the mean over 12 dev
seeds and is confirmed by the seed-777 table you already have, where all five are absent.)
**Any budget spent on those five kinds is wasted by construction.** The baseline recipe's
curriculum spends its first two levels on `leap`, `units`, `powers` and `sequences`.

Mean composition of the 200 scored items over 12 dev seeds (277 of 600 generated items
survive the filter, so there are ~77 items of headroom above the 200 cap):

| kind | items / 200 | now | what is reachable |
|---|---:|---:|---|
| `code` | 80.1 | 100% | nothing to gain, everything to lose |
| `arithmetic` | 57.8 | 5% | 40.8 reachable, 17.0 not |
| `dates` | 19.2 | 6% | all three forms, with low confidence on two |
| `gcd_lcm` | 13.3 | 27% | ~4.0 (the `gcd = 1` prior); Euclid is out of reach |
| `syllogism` | 10.1 | 100% | nothing to gain |
| `primality` | 6.1 | 100% | nothing to gain (majority class "no" scores ~90%) |
| `system2` | 6.0 | 0% | 6.0 |
| `factorise` | 3.2 | 0% | ~0.3 (the prime-`n` case) — written off |
| `linear` | 2.6 | 0% | 2.6 |
| `percent` | 1.6 | 0% | 1.6 |

Two of the "already perfect" kinds are perfect for reasons worth naming, because it
changes how you defend them. `primality` is a majority-class guess: 90% of
`randint(2,100000)` is composite, and 6/6 is what "always no" gets. `syllogism`'s two
hard templates (15,660 and 29,232 distinct questions) are exhausted by the database, so
every surviving syllogism is the single copy-the-noun template `All a are b. All b are c.
X is an a. Is X a c?` → `Yes, X is a c.` Neither is a skill that can be improved; both
are cheap to protect.

### The arithmetic breakdown that decides the budget

Classifying every `arithmetic` item that reaches the scored 200, over 12 dev seeds:

| slice | items / 200 | verdict |
|---|---:|---|
| add/sub, 3-digit operands | 10.4 | **learnable** |
| add/sub, 5–6-digit operands | 11.6 | **learnable** |
| add/sub, 1–2-digit operands | 0.8 | already mostly right |
| `a × b`, one factor ≤ 20 | 10.3 | **learnable** (short carry chain) |
| `a × b`, both factors > 20 | 14.0 | **write off** |
| `a / b`, answer a fraction already in lowest terms | 6.6 | **learnable: the answer is a copy** |
| `a / b`, answer a reducible fraction | 2.9 | write off (needs `gcd`) |
| `a / b`, answer a terminating decimal | 1.1 | **learnable** (shift-divide) |

Reachable: **40.8 of 57.8**. The 1–2-digit cases are nearly gone because their question
space (≈13,000 for both operands ≤ 20) is exhausted — the evaluation's arithmetic is
*specifically* the part the current tokenizer cannot represent.

The division line is the one most people throw away. `_fmt(Fraction(a,b))` prints
`"{num}/{den}"` whenever the quotient is not a multiple of 0.001, and `Fraction` only
normalises the sign. Two random integers are coprime with probability 6/π² = 0.608, so for
**6.6 items per 200 the correct answer is literally the two operands copied across with
the sign moved to the numerator**. That is a copy task with a sign rule, not arithmetic.
I am equally explicit that this buys nothing on the 2.9 reducible items: a model that
always copies is wrong there, which is where it already is.

### Dates: I had this wrong at first and the correction matters

0.4 of the `dates` generator's mass is day-of-week, which looked like the obvious prize.
It is not: enumerating the day-of-week question space against `runs/qa` shows **87.5% of
it is already in the database**, so day-of-week is worth only ~1.9 of the 19.2 dates
items. The remaining ~8.5 are `how many days between d1 and d2` → `N days` and ~8.8 are
`what date is N days after d` → ISO date. Both are harder than day-of-week and both get
drilled, with the honest caveat in §7 that they are the lowest-confidence part of the mix.

## 4. Why the two measured collapses happened, and why this run does not repeat them

The 6% and 4% runs shared three properties: three gradient passes over the *same* batch,
a batch of four, and batch contents selected *because the model failed them*. Failure
selection on this distribution concentrates the gradient on big multiplication and
reducible fractions — targets with no short solution — and the only way to reduce that
loss is to distort representations that other kinds share. That is the mechanism: not the
learning rate, but repeated high-variance updates toward an unfittable subpopulation.

So the fix is not "a smaller learning rate" (the measured gentle run proves a small enough
rate changes nothing). The fix is:

* **one pass per batch, every question drawn once** — no failure-driven feedback loop at all;
* **~4,000 tokens per update** (≈130 items) instead of 4 items, which is a 30× reduction
  in gradient variance;
* **a stationary mixture that includes the kinds being protected** — code at ~40% of the
  token budget, syllogism, primality and `gcd_lcm` as maintenance streams;
* **unfittable targets removed from the mixture**, so no amount of training pressure can
  push the model to distort shared features in pursuit of them.

On the learning rate specifically: `runs/cpu/log.csv` ends at `2.000001e-04`, the floor of
the pretraining cosine. 2e-4 is therefore the largest rate at which this model is
*demonstrably* stable on this data, and it is 20× the gentle run's 1e-5 that provably
moved nothing. That is the anchor, not a guess.

## 5. The representation change, and what it costs

The pre-tokenisation regex lives in the serialised tokenizer (`"pattern"` in
`tokenizer.json`), so changing it needs **no edit to `mnx/tokenizer.py`** and the new
pattern travels with the run directory that `load_checkpoint` reads:

```
old   r"'(?:[sdmt]|ll|ve|re)| ?[A-Za-z_][A-Za-z0-9_]*| ?\d+| ?[^\sA-Za-z0-9_]+|\s+(?!\S)|\s+"
new   r"'(?:[sdmt]|ll|ve|re)| ?[A-Za-z_][A-Za-z0-9_]*| ?\d | ?[^\sA-Za-z0-9_]+|\s+(?!\S)|\s+"
                                                        ^ the whole change: \d+ -> \d
```

BPE merges only apply *inside* a pre-token, so one digit per pre-token makes every
digit-spanning merge unreachable without deleting it. Vocabulary size, merge ranks,
special-token ids and therefore every row of the embedding matrix are untouched — I assert
this in code. Numbers now use the 20 tokens `' 0'…' 9'` and `'0'…'9'`, which are merges
#1–#11 and #14–… , i.e. the most frequent in the table, so their embeddings are already
well trained.

What it costs, measured over 4,000 dev-seed Q&A documents:

| kind | tokens/doc now | with digit split | p95 | max | docs tokenised identically |
|---|---:|---:|---:|---:|---:|
| `code` | 78.6 | 80.8 | 133 | **183** (< 256) | 46% |
| `syllogism` | 31.9 | 31.9 | 35 | 37 | **100%** |
| `arithmetic` | 16.0 | 21.9 | 31 | 35 | 1% |
| `dates` | 21.1 | 28.7 | 36 | 36 | 0% |

Syllogism is byte-for-byte unchanged, so its 14/14 cannot move. Code grows 2.8% and stays
well inside `max_seq_len = 256`.

And the de-risking measurement that matters most — swap the tokenizer on `runs/cpu` with
**zero training** and score 25 dev drills per kind:

| | code | syllogism | primality | gcd_lcm | division | add/sub |
|---|---:|---:|---:|---:|---:|---:|
| original pattern | 88% | 100% | 88% | 36% | 28% | 4% |
| digit-split pattern | **88%** | **100%** | **92%** | 28% | 4% | 0% |

The re-segmentation costs nothing on the two kinds worth 90 of the 200 points even before
any repair training. What it costs is the near-zero numeric performance, which is exactly
what the fine-tune then rebuilds from a representation that can support it.

## 6. The recipe

### Data

Item weights (a code item is ~4× longer, so code at 0.15 of the items is ~40% of the
*token* budget — the share of the score it defends):

| stream | weight | items/200 it targets |
|---|---:|---|
| `addsub` — generator wordings, operand-width curriculum | 0.30 | 22.8 |
| `mul_small` — one factor in [−20, 20] | 0.12 | 10.3 (+ it supplies the multiply `percent` needs) |
| `division` — the generator's own branch, undistorted | 0.08 | 7.7 |
| `between` — `how many days between` | 0.06 | 8.5 |
| `after` — `what date is N days after` | 0.05 | 8.8 |
| `weekday` | 0.03 | 1.9 |
| `system2` | 0.04 | 6.0 |
| `linear` | 0.03 | 2.6 |
| `percent` — restricted to reachable `p` values | 0.02 | 1.6 |
| `gcd_lcm` | 0.04 | maintenance |
| `primality` | 0.03 | maintenance |
| `syllogism` | 0.05 | maintenance |
| `code` | 0.15 | maintenance (80 items) |

Not in the mixture, each with a reason: `units`, `sequences`, `powers`, `quadratic`,
`leap` (0.1 items per 200 between them — question spaces exhausted); `factorise` and LCM
(trial division / multiply-then-divide, no mechanism at 10M); big×big multiplication
(14.0 items) and reducible-fraction division (2.9 items) — no scratchpad is available
because the evaluator reads only the first output line.

Curricula, both on the same progress axis (fraction of the planned item count):

| progress | add/sub, mul, div operand width | date gap |
|---|---|---|
| 0.00–0.12 | ≤ 2 digits | ≤ 28 days (never crosses a month) |
| 0.12–0.30 | ≤ 4 digits | ≤ 120 days |
| 0.30–0.55 | ≤ 7 digits | ≤ 700 days |
| 0.55–1.00 | **the generator's own `_num` mixture** | ≤ 2000 days (the generator's own) |

The last stage matters: `_num` is the exact operand distribution the evaluation uses
(bimodal, ~37% 3-digit and ~36% 6–7-digit), so the run *ends* with no distribution shift.
The early stages are a curriculum device and cost nothing in exclusions, because
small-operand questions are already in the 3M database.

### Loop

| | value | why |
|---|---|---|
| init | `runs/cpu` | as given |
| loss | cross-entropy on answer tokens + EOS, prompt masked with `-100` | mechanism (2): the question is irreducible entropy |
| batching | length-bucketed from a pool of 8,192 items; `token_budget` 2,048, `grad_accum` 2 | ~4,096 tokens ≈ 130 items per update; 30× less gradient variance than batch 4 |
| passes per batch | **1** | the collapse mechanism; nothing is ever seen twice |
| optimiser | AdamW, β = (0.9, 0.95), weight decay 0.1 on matrices only (`build_optimizer`) | unchanged from pretraining |
| lr | peak **2e-4**, linear warmup 300 steps, cosine to **1e-5** | 2e-4 is the floor of the pretraining cosine — the largest demonstrably stable rate |
| grad clip | 1.0 | unchanged |
| steps | ~9,200 (set from a 200-step rate measurement; ~3 h at the 3.5 k tok/s I measured on an idle 4-core box) | |
| sequence length | 256, unchanged; longest drill is 183 tokens | |
| EMA | decay 0.999, written to `ckpt_ema.pt` | free stabiliser; **the raw/EMA choice is made on the dev-seed validation before seed 777 is touched** |
| threads | 4 | |
| checkpointing | atomic `ckpt.pt` + optimiser state + `sft_state.json` every 500 steps and on SIGINT/SIGTERM | resumable |

Resources: 10.4M fp32 parameters = 42 MB, AdamW state 84 MB, the EMA copy 42 MB, and a
4,096-token forward/backward whose largest tensor is the 4,096 x 3,355 logit block — under
2 GB peak against the 15 GB available, and no GPU, no network and no dependency beyond
PyTorch CPU and the standard library. The item generator runs at ~11,000 items/s, i.e.
~2% of one core against the ~150 items/s the optimiser consumes, so data generation is
never the bottleneck.

The data stream is a pure function of `(seed, item index)`, so `--resume` continues at the
exact item it stopped at and `--history` *regenerates* the trained-question list instead of
appending it during training (appending double-counts across a resume — a real bug in the
obvious implementation).

### Keeping the measurement honest: the exclusion footprint

`eval_qa.py` drops every question the run trained on. That has a consequence nobody should
exploit: **excluding questions of a kind you fail raises your score**, because the 200 slots
are back-filled from the surviving stream, which is 40% `code` at 100%. Flooding
`qa_history.jsonl` with `factorise`, LCM and big-multiplication questions would raise the
total without teaching the model anything. That is gaming the measurement and I am not
proposing it; I name it so that the number below can be checked for it.

The defence is to bound the footprint and report it. For every kind whose question space
is small enough to matter, the recipe draws a **bounded set of distinct questions and
samples it with replacement**:

| kind | distinct drills | reachable pool | fraction removed |
|---|---:|---:|---:|
| `weekday` (1900–2095) | 3,000 | ~9,100 | 33% of 1.9 items ≈ 0.6 |
| `primality` | 20,000 | ~66,000 | ~30% of a kind that is already at its ceiling |
| `gcd_lcm` | 20,000 | ~800,000 | 2.5% |
| `system2` | 20,000 | ~250,000 | 8% |
| `linear` | 1,500 | ~16,000 | 9% |
| `percent` | 8,000 | ~339,000 | 2% |

`addsub`, `mul_small`, `division`, `between`, `after` and `code` are uncapped because
their spaces are 10⁸–10¹³. 70% of the day-of-week drills use years 2096–2495, which the
evaluation generator never emits and which are *the same function* — the Gregorian
calendar has a 400-year period — so they buy sample count at zero cost in eval items, and
because they span 400 years the compact solution is the arithmetic rule rather than a
year→code table. Total expected extra exclusions: **≈12 of the 600 generated items**,
against ~77 items of headroom, so the 200 slots still fill and the per-kind denominators
move by at most one item. §7 makes that a pass/fail check rather than a promise.

## 7. Measurement plan, with the results that would falsify it

Before training:

```bash
# B0 - the baseline you already have, re-confirmed on this machine
python scripts/eval_qa.py --ckpt runs/cpu --n 200 --seed 777          # expect 107/200
# B1 - control: the tokenizer swap alone, no training.  MUST be much worse than B0.
#      (copy runs/cpu to runs/cpu_digit and rewrite its tokenizer.json "pattern")
python scripts/eval_qa.py --ckpt runs/cpu_digit --n 200 --seed 777
```

B1 exists so that nobody can claim the gain came from the tokenizer rather than the
training. Predicted: a large drop, concentrated in the numeric kinds.

During the run, on dev drills (`--val-seed 90210`, never the eval seed), 32 items per
kind, every 2,000 steps. Four gates:

| gate | step | condition | action if it fails |
|---|---|---|---|
| G1 code | 2,000 | `code` ≥ 0.9 × its step-0 value | raise `code` to 0.25, restart; the re-segmentation is hurting code |
| G2 add/sub | 4,000 | ≤ 4-digit `addsub` ≥ 40% | **the core claim is false.** Fall back: keep the tokenizer, drop ≥ 5-digit add/sub, spend the weight on `division` + `mul_small` + 3-digit add/sub. Expect +10 instead of +23 |
| G3 dates | 6,000 | `between` ≥ 15% | move `between` + `after` weight (0.11) to `addsub` and `mul_small`; expect dates to stay at 1 |
| G4 maintenance | every | `syllogism` = 100% and `primality` ≥ 85% | roll back to the last checkpoint and halve the lr |

After the run, **once**:

```bash
python -m mnx.qa_sft --out runs/sft --history
python scripts/eval_qa.py --ckpt runs/sft --n 200 --seed 777 --exclude runs/sft/qa_history.jsonl
python scripts/eval_qa.py --ckpt runs/sft --n 600 --seed 777 --exclude runs/sft/qa_history.jsonl
```

Compare per kind, not just the total. The `--n 600` run is there because 200 items give a
±3.5-point standard error on the total and ±6 items on `arithmetic`; a +23 claim needs the
larger sample to be anything other than suggestive.

**What refutes the recipe**

* Total ≤ 107/200. The recipe is refuted outright.
* `arithmetic` < 12/63. The digit-alignment mechanism is wrong, or 3 h is not enough
  budget for left-to-right carries; either way the central claim fails.
* `code` < 76/80 or `syllogism` < 13/14. The re-segmentation is not as cheap as the
  zero-training measurement in §5 says; the recipe is net-harmful and should not ship.
* The `code`, `syllogism` or `primality` denominators at seed 777 move by more than two
  items from the published table. The comparison is then contaminated by the exclusion
  footprint and only the `--n 600` numbers mean anything.
* **The ablation**: the identical run with `--no-digit` (same data, same schedule, same
  budget — one flag) reaches within 5 points of the digit run on ≤ 4-digit `addsub`. Then
  the tokenizer is not the mechanism and the explanation in §3 is wrong, even if the score
  went up.

I ran a 2,400-step paired version of that ablation on this machine (1-digit-to-4-digit
add/sub at 82% of items, code and syllogism as maintenance, identical seeds and schedule,
the only difference being the `"pattern"` field). Two honest notes about it. First, an
earlier version at ≤ 2 digits showed **no** advantage for digit splitting (20% vs 25% at
step 300) — correctly so, because 1–2-digit numbers are single tokens under *both* schemes,
and I had chosen the one regime where the mechanism cannot apply. That is why the gates
above test ≤ 4 digits. Second, the machine was heavily contended, so the absolute rate
(~550 tok/s per job instead of 3.5 k) and hence the absolute accuracies are much lower
than a clean 3 h run would give; only the A-vs-B comparison is meaningful.

## 8. Expected gain, per kind

Denominators as in the seed-777 table you have. "Reachable" is the measured count from §3.

| kind | now | reachable | expected | where it comes from |
|---|---:|---:|---:|---|
| `code` | 80/80 | — | **78–80** | protected at ~40% of the token budget; measured to survive re-segmentation untrained |
| `syllogism` | 14/14 | — | **14** | tokenisation provably identical |
| `primality` | 6/6 | — | **5–6** | majority-class prior, maintained |
| `arithmetic` | 3/63 | 40.8 | **16–28** | add/sub 22.8 at an expected 55–75%; small-factor mul 10.3 at 30–50%; coprime division 6.6 at 60–85%; decimal division 1.1 |
| `dates` | 1/17 | ~19 | **2–5** | weekday ~1.9 (most of the space is already trained); `between` and `after` at 10–25% — the least confident line here |
| `gcd_lcm` | 3/11 | ~4 | **3–4** | prior maintained; no mechanism for Euclid, so no gain claimed |
| `system2` | 0/6 | 6.0 | **1–3** | per-coefficient-pattern 2×2 rational inverse: 255 non-singular patterns, answers in a 41×41 grid, determinants in −9…9 |
| `linear` | 0/1 | 2.6 | **0–1** | `x = (c−b)/a`: a 3-digit subtraction then an exact division by \|a\| ≤ 12 with the answer in −30…30 |
| `percent` | 0/1 | 1.6 | **0–1** | `p·n` is a small-factor multiply, then a two-place decimal shift |
| `factorise` | 0/1 | ~0.3 | **0** | written off |
| **total** | **107** | | **125–140** | central **~130 / 200 (65%)** |

Mechanisms, stated so they can be argued with rather than believed:

* **Add/sub left-to-right.** Carry propagation is an associative prefix scan over
  per-position generate/propagate/kill signals, so a transformer can resolve a 7-position
  carry chain in ⌈log₂ 7⌉ ≈ 3 attention layers; 8 layers is enough. The usual fix —
  emitting the answer least-significant digit first — is **not available**, because the
  evaluator compares the first output line to the exact gold string. Left-to-right
  therefore costs more samples than reversed output, which is why add/sub gets the largest
  single weight and a width curriculum rather than a flat mixture. Digit-level tokens are
  what make the prefix scan expressible at all: with the current chunking there is no
  position whose value is "the k-th digit of the operand".
* **Small-factor multiplication.** `s × b` with \|s\| ≤ 20 is a per-digit multiply plus a
  carry of bounded size — the same prefix-scan structure as addition, which is why it
  shares the gain and why big×big (a quadratic number of partial products) does not.
* **Coprime division.** The target is the two operands copied with the sign normalised
  onto the numerator. Copying is what this model is already demonstrably good at
  (`code` 80/80, `syllogism` 14/14 are both copy tasks). The model will not learn to
  *decide* coprimality, so it will copy when it should reduce; that costs nothing, since
  those 2.9 items are already wrong.
* **Dates.** `how many days between` decomposes into two day-of-year lookups (a 12-entry
  cumulative table), a 7-entry `365 × Δy` table, a leap correction and one 4-digit signed
  subtraction — i.e. it sits *on top of* the add/sub circuit this recipe builds, which is
  why it is drilled after the width curriculum has opened up and why gate G3 exists.
  `what date is N days after` needs the same machinery inverted with variable month
  lengths and must get all ten characters right; it is the weakest claim in the table.
* **`system2`.** Coefficients come from `{1, 2, 3, −1}`, so there are only 255
  non-singular 2×2 patterns; for each, `(x, y)` is a fixed rational linear map of two
  3-digit right-hand sides. That is a per-pattern linear readout, well within 10M
  parameters — the reason 128,250 pretraining examples produced 0/6 is mechanisms (1)
  and (2), not capacity.

A recipe that claimed 30 points here would have to claim big multiplication or fraction
reduction, and neither has a mechanism that fits in a 256-token single-line answer.

## 9. If I had a second run

In priority order: (a) raise `addsub` to 0.45 and drop `after` entirely if G3 fails,
(b) add a second ~3 h continuation from the §8 checkpoint with the mixture re-weighted
by measured per-slice accuracy (which is what the trainer bot's active learning was
*trying* to do, but at the batch level where it destroys the model instead of at the
mixture level where it is safe), (c) retrain the tokenizer from scratch with digits
excluded from merging and re-pretrain — the clean version of §5, which this recipe
approximates without paying for a new pretraining run.

## 10. The change

One new file, `mnx/qa_sft.py`. **No existing file changes** — `scripts/eval_qa.py`,
`mnx/model.py`, `mnx/train.py`, `mnx/tokenizer.py`, `mnx/trainer_bot.py` and
`mnx/agent/qa_stream.py` are untouched, because the pre-tokenisation pattern is a
serialised field of `tokenizer.json` and `load_checkpoint` reads the tokenizer from the
run directory. `mnx/trainer_bot.py` stays as it is; this is a sibling entry point, not a
replacement.

Run sheet:

```bash
# 0. controls, before anything is trained
python scripts/eval_qa.py --ckpt runs/cpu --n 200 --seed 777
cp -r runs/cpu runs/cpu_digit && python - <<'EOF'
import json, pathlib
from mnx.tokenizer import DEFAULT_PATTERN
p = pathlib.Path("runs/cpu_digit/tokenizer.json"); d = json.loads(p.read_text())
d["pattern"] = DEFAULT_PATTERN.replace(r"| ?\d+", r"| ?\d"); p.write_text(json.dumps(d, indent=1))
EOF
python scripts/eval_qa.py --ckpt runs/cpu_digit --n 200 --seed 777      # must be much worse

# 1. the run (~3 h, 4 cores, resumable with --resume after any interruption)
python -m mnx.qa_sft --ckpt runs/cpu --out runs/sft --hours 3.0 --threads 4

# 2. the ablation that tests the explanation, not the score
python -m mnx.qa_sft --ckpt runs/cpu --out runs/sft_nodigit --hours 3.0 --threads 4 --no-digit

# 3. the measurement, once
python -m mnx.qa_sft --out runs/sft --history
python scripts/eval_qa.py --ckpt runs/sft --n 200 --seed 777 --exclude runs/sft/qa_history.jsonl
python scripts/eval_qa.py --ckpt runs/sft --n 600 --seed 777 --exclude runs/sft/qa_history.jsonl
```

Save the file below as `mnx/qa_sft.py`; it is a new file and nothing else moves.

I ran this file end to end on `runs/cpu`: it trains, saves atomically, resumes at the
exact item index, regenerates a duplicate-free `qa_history.jsonl` with the out-of-range
day-of-week drills correctly excluded, and writes a `tokenizer.json` whose pattern is the
one in §5.

```python
"""Targeted supervised fine-tuning of the MNX 10M CPU checkpoint on Q&A.

    python -m mnx.qa_sft --ckpt runs/cpu --out runs/sft --hours 3.0
    python -m mnx.qa_sft --out runs/sft --resume
    python -m mnx.qa_sft --out runs/sft --history          # trained-question list
    python -m mnx.qa_sft --ckpt runs/cpu --out runs/sft_nodigit --hours 3.0 --no-digit   # ablation

Three differences from ``mnx.trainer_bot``, each measured rather than assumed.

1. **Digit-level pre-tokenisation.**  The run's ``tokenizer.json`` is written
   with the pre-token pattern's ``| ?\\d+`` branch replaced by ``| ?\\d``.  The
   merge table, the special tokens and therefore every embedding row are
   byte-identical, so the pretrained checkpoint stays loadable and
   ``scripts/eval_qa.py`` picks the new segmentation up for free (it loads the
   tokenizer from the checkpoint directory).  Under the old pattern
   ``913463 + 29 = 913492`` segments as
   ``[' 913','46','3']  [' 29']  ->  [' 913','49','2']`` - the operand chunks
   and the result chunks do not line up, so there is no position at which a
   carry can be computed.  Digit-level segmentation makes that alignment exist.

2. **A fittable mixture.**  60% of the ``arithmetic`` generator's mass
   (big x big multiplication, non-coprime fraction division) cannot be fitted
   by a 10M model without a scratchpad, and the evaluator reads only the first
   output line, so a scratchpad is not available.  Those targets are dropped
   from the gradient budget; the budget moves to add/sub, small-factor
   multiplication, coprime division (where the target *is* a copy of the two
   operands) and day-of-week.

3. **Plain single-pass SFT.**  One gradient pass per batch, ~4000 tokens per
   update, answer tokens only in the loss, every question drawn once.  The
   measured collapses came from repeated passes on failure-selected batches of
   four; that mechanism is simply absent here.

The whole data stream is a pure function of ``(seed, item index)``, so the run
resumes exactly and the trained-question list can be regenerated afterwards for
``eval_qa.py --exclude`` instead of being stored during training.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import signal
import sys
import time
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import torch

from mnx.agent import qa_stream as QS
from mnx.agent.expert_stream import _held_out_names
from mnx.tokenizer import DEFAULT_PATTERN, BPETokenizer
from mnx.train import CKPT_NAME, CONFIG_NAME, TOKENIZER_NAME, build_optimizer, cosine_lr, load_checkpoint

PROMPT = "Q: {q}\nA:"
DIGIT_PATTERN = DEFAULT_PATTERN.replace(r"| ?\d+", r"| ?\d")
STATE_NAME = "sft_state.json"
EMA_NAME = "ckpt_ema.pt"


def make_digit_tokenizer(src: BPETokenizer) -> BPETokenizer:
    """Same vocabulary and merges; digits split one per pre-token."""
    d = src.to_dict()
    d["pattern"] = DIGIT_PATTERN
    tok = BPETokenizer.from_dict(d)
    assert tok.vocab_size == src.vocab_size, "digit split must not change the vocabulary"
    return tok


# ------------------------------------------------------------------- drills
# The question wordings are exactly the generator's own, so a drill is
# indistinguishable in form from an evaluation item.
ARITH_FORMS = ("what is {a} {op} {b}", "what is {a} {word} {b}", "calculate {a} {op} {b}", "{a} {op} {b} = ?")
WORDS = {"+": "plus", "-": "minus", "*": "times", "/": "divided by"}
MONTH_LEN = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)


def _ask(r: random.Random, a: int, b: int, op: str) -> str:
    return r.choice(ARITH_FORMS).format(a=a, b=b, op=op, word=WORDS[op])


def _nd(r: random.Random, dmax: int) -> int:
    """An integer whose digit length is uniform on 1..dmax, negative 25% of the time."""
    k = r.randint(1, dmax)
    v = r.randint(0, 9) if k == 1 else r.randint(10 ** (k - 1), 10 ** k - 1)
    return -v if r.random() < 0.25 else v


def _operand(r: random.Random, dmax: Optional[int]) -> int:
    return QS._num(r) if dmax is None else _nd(r, dmax)


def addsub(r: random.Random, dmax: Optional[int]) -> dict:
    a, b = _operand(r, dmax), _operand(r, dmax)
    op = r.choice("+-")
    return {"q": _ask(r, a, b, op), "a": str(a + b if op == "+" else a - b), "kind": "arithmetic"}


def mul_small(r: random.Random, dmax: Optional[int]) -> dict:
    """Multiplication with one factor in [-20, 20]: 45% of the generator's
    multiplication mass, and the only part whose carry chain is short enough."""
    s = r.randint(0, 20)
    if r.random() < 0.25:
        s = -s
    big = _operand(r, dmax)
    a, b = (s, big) if r.random() < 0.5 else (big, s)
    return {"q": _ask(r, a, b, "*"), "a": str(a * b), "kind": "arithmetic"}


def division(r: random.Random, dmax: Optional[int]) -> dict:
    """The generator's own division branch, undistorted.  86% of its answers are
    a fraction and 61% of those are already in lowest terms, i.e. the target is
    a sign-normalised copy of the two operands."""
    a = _operand(r, dmax)
    b = _operand(r, dmax) or 7
    return {"q": _ask(r, a, b, "/"), "a": QS._fmt(Fraction(a, b)), "kind": "arithmetic"}


def _iso_date(r: random.Random) -> dt.date:
    return dt.date(1900, 1, 1) + dt.timedelta(days=r.randint(0, 200 * 365))


def days_between(r: random.Random, gapmax: int) -> dict:
    """``how many days between d1 and d2`` -> ``N days``.  The generator's own
    form; ``gapmax`` is the curriculum knob (a gap under 28 needs no month
    carry, a gap over 365 needs a year carry)."""
    d = _iso_date(r)
    e = d + dt.timedelta(days=r.randint(1, gapmax))
    return {"q": f"how many days between {d.isoformat()} and {e.isoformat()}",
            "a": f"{(e - d).days} days", "kind": "dates"}


def date_after(r: random.Random, gapmax: int) -> dict:
    """``what date is N days after d`` -> ISO date."""
    d = _iso_date(r)
    n = r.randint(1, min(1000, gapmax))
    return {"q": f"what date is {n} days after {d.isoformat()}",
            "a": (d + dt.timedelta(days=n)).isoformat(), "kind": "dates"}


def percent_reachable(r: random.Random, _dmax: Optional[int] = None) -> dict:
    """``p% of n``, restricted to the only p values the evaluator can still ask.

    The generator picks p from ``[5,10,12,15,20,25,30,40,50,75, randint(1,99)]``;
    the ten literal values x 2000 n x 2 wordings = 40,000 questions and the
    3M-pair database holds 57,178 percent questions, so every literal-p question
    is already in it and can never appear in the evaluation.  Only the
    ``randint(1,99)`` tail is reachable."""
    p = r.choice([v for v in range(1, 100) if v not in (5, 10, 12, 15, 20, 25, 30, 40, 50, 75)])
    n = r.randint(1, 2000)
    q = r.choice([f"what is {p}% of {n}", f"{p} percent of {n}"])
    return {"q": q, "a": QS._fmt(Fraction(p * n, 100)), "kind": "percent"}


def weekday(r: random.Random, in_range: bool) -> dict:
    """Day-of-week.  ``in_range`` draws the 1900-2095 years the evaluator asks
    about; otherwise 2096-2495, which is the *same* function (the Gregorian
    calendar has a 400-year period) with different digits, so it buys sample
    count without removing dates from the evaluator's pool."""
    y = r.randint(1900, 2095) if in_range else r.randint(2096, 2495)
    m = r.randint(1, 12)
    last = 29 if (m == 2 and y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)) else MONTH_LEN[m - 1]
    d = r.randint(1, last)
    return {"q": f"what day of the week was {QS.MONTHS[m - 1]} {d}, {y}",
            "a": QS.WEEKDAYS[dt.date(y, m, d).weekday()], "kind": "dates", "in_eval_pool": in_range}


def _plain(kind: str) -> Callable[[random.Random, Optional[int]], dict]:
    gen = QS.GENERATORS[kind][1]

    def f(r: random.Random, _dmax: Optional[int] = None) -> dict:
        q, a = gen(r)
        return {"q": q, "a": a, "kind": kind}
    return f


def _code(banned) -> Callable[[random.Random, Optional[int]], dict]:
    def f(r: random.Random, _dmax: Optional[int] = None) -> dict:
        q, a = QS.code_task(r, banned)
        return {"q": q, "a": a, "kind": "code"}
    return f


# Item weights.  A code item is ~4x longer than an arithmetic one, so code at
# 0.15 of the items is ~40% of the token budget - the share of the score it
# defends (80 of 200).
MIX: Dict[str, float] = {
    "addsub": 0.30,          # 22.8 of the 57.8 arithmetic items that reach the eval
    "mul_small": 0.12,       # 10.3; also supplies the multiply that percent needs
    "division": 0.08,        #  7.7 reachable (6.6 coprime copy + 1.1 clean decimal)
    "between": 0.06,         #  8.5 of the 19.2 dates items
    "after": 0.05,           #  8.8 dates items; lowest-confidence slice in the mix
    "weekday": 0.03,         #  1.9 dates items (87% of that space is already trained)
    "system2": 0.04,         #  6.0
    "linear": 0.03,          #  2.6
    "percent": 0.02,         #  1.6
    "gcd_lcm": 0.04,         # maintenance: hold the "answer 1" prior worth 3-4
    "primality": 0.03,       # maintenance: hold the majority-class prior worth 6
    "syllogism": 0.05,       # maintenance: 10-14 items, tokenisation unchanged
    "code": 0.15,            # maintenance: 80 items, ~40% of the token budget
}
# Not in the mix, with a reason: units, sequences, powers, quadratic and leap
# together contribute 0.1 items per 200 because the 3M-pair database already
# holds their entire question space (35,725 / 3,399 / 798 / 325 / 801 distinct
# questions), so eval_qa drops every one of them.  factorise (3.2 items) and
# lcm need trial division / a multiply-divide pair that a 10M model has no
# mechanism for, and big x big multiplication (14.0 items) and reducible
# fraction division (2.9) need a scratchpad the output format forbids.

# Kinds whose question space is small enough that unbounded drilling would start
# removing evaluation items: draw a bounded set of distinct questions and sample
# it with replacement.  value = number of distinct questions allowed.
CAPS: Dict[str, int] = {
    "weekday_in_range": 3_000,    # only ~9,100 in-range dates are still reachable
    "system2": 20_000,            # of ~250,000 reachable
    "linear": 1_500,              # of ~16,000 reachable - nearly exhausted already
    "percent": 8_000,             # of ~339,000 reachable
    "gcd_lcm": 20_000,            # of ~800,000 reachable
    "primality": 20_000,          # of ~66,000 reachable
}
# Fraction of day-of-week drills taken from the 1900-2095 years the evaluator
# uses; the rest come from 2096-2495, which is the same function (400-year
# Gregorian period) and is never asked about, so it costs no eval items.
WEEKDAY_IN_RANGE = 0.3

# Operand-width curriculum for the three arithmetic drills, as a fraction of the
# planned item count.  ``None`` = the generator's own ``_num`` mixture, so the
# run *ends* on the evaluation distribution.
STAGES: Tuple[Tuple[float, Optional[int]], ...] = ((0.12, 2), (0.30, 4), (0.55, 7), (2.0, None))
# Date-gap curriculum, same progress axis: a gap below 28 days never crosses a
# month boundary, a gap below 365 never crosses a year.
GAP_STAGES: Tuple[Tuple[float, int], ...] = ((0.12, 28), (0.30, 120), (0.55, 700), (2.0, 2000))


def stage_gap(progress: float) -> int:
    for upto, gap in GAP_STAGES:
        if progress < upto:
            return gap
    return 2000


def stage_dmax(progress: float, override: Optional[int] = None) -> Optional[int]:
    if override:
        return override
    for upto, dmax in STAGES:
        if progress < upto:
            return dmax
    return None


class Mixture:
    """``item(i, progress)`` is a pure function of (seed, i), which is what makes
    the run resumable and the trained-question list regenerable."""

    def __init__(self, seed: int = 0, dmax_override: Optional[int] = None) -> None:
        self.seed = seed
        self.dmax_override = dmax_override
        banned = _held_out_names()
        self.fns: Dict[str, Callable] = {
            "addsub": addsub, "mul_small": mul_small, "division": division,
            "between": lambda r, d=None: days_between(r, d if isinstance(d, int) else 2000),
            "after": lambda r, d=None: date_after(r, d if isinstance(d, int) else 1000),
            "weekday": lambda r, d=None: weekday(r, True),
            "percent": percent_reachable, "code": _code(banned)}
        for k in ("system2", "linear", "gcd_lcm", "primality", "syllogism"):
            self.fns[k] = _plain(k)
        self.keys = [k for k in MIX if MIX[k] > 0]
        self.weights = [MIX[k] for k in self.keys]

    def item(self, i: int, progress: float) -> dict:
        r = random.Random(f"{self.seed}:{i}")
        name = r.choices(self.keys, self.weights)[0]
        cap = CAPS.get(name)
        if name == "weekday":
            if r.random() < WEEKDAY_IN_RANGE:
                rr = random.Random("%s:wd:%d" % (self.seed, r.randrange(CAPS["weekday_in_range"])))
                return weekday(rr, True)
            return weekday(r, False)
        if name in ("between", "after"):
            return self.fns[name](r, stage_gap(progress))
        if cap is not None:                       # bounded distinct set, sampled with replacement
            r = random.Random(f"{self.seed}:{name}:{r.randrange(cap)}")
        return self.fns[name](r, stage_dmax(progress, self.dmax_override))


# ------------------------------------------------------------------ batching
def encode(tok: BPETokenizer, it: dict, limit: int) -> Optional[Tuple[List[int], List[int]]]:
    prompt = tok.encode(PROMPT.format(q=it["q"]), add_bos=True, allow_special=False)
    answer = tok.encode(" " + it["a"], allow_special=False) + [tok.eos_id]
    ids = (prompt + answer)[:limit]
    tgt = ([-100] * len(prompt) + answer)[:limit]
    return (ids, tgt) if len(ids) >= 2 and len(prompt) + len(answer) <= limit else None


def pack(rows: Sequence[Tuple[List[int], List[int]]], pad: int):
    m = max(len(r[0]) for r in rows)
    x = torch.full((len(rows), m), pad, dtype=torch.long)
    y = torch.full((len(rows), m), -100, dtype=torch.long)
    for i, (ids, tgt) in enumerate(rows):
        x[i, : len(ids)] = torch.tensor(ids)
        y[i, : len(tgt)] = torch.tensor(tgt)
    return x[:, :-1].contiguous(), y[:, 1:].contiguous()


def batch_stream(mix: Mixture, tok: BPETokenizer, limit: int, token_budget: int, pool_items: int,
                 start_item: int, plan_items: Callable[[], int], stop: Callable[[], bool]
                 ) -> Iterator[Tuple[torch.Tensor, torch.Tensor, int]]:
    """Length-bucketed batches of roughly constant token count.  Yields
    ``(x, y, items_consumed_so_far)``."""
    rng = random.Random(0xB0A7)
    i = start_item
    while not stop():
        pool = []
        for _ in range(pool_items):
            enc = encode(tok, mix.item(i, min(0.999, i / max(1, plan_items()))), limit)
            i += 1
            if enc is not None:
                pool.append(enc)
        pool.sort(key=lambda e: len(e[0]))
        groups, j = [], 0
        while j < len(pool):
            n = max(1, token_budget // max(1, len(pool[j][0])))
            groups.append(pool[j: j + n])
            j += n
        rng.shuffle(groups)
        for g in groups:
            x, y = pack(g, tok.pad_id)
            yield x, y, i


# ---------------------------------------------------------------- validation
VAL_KINDS = ("addsub", "mul_small", "division", "between", "after", "weekday",
             "system2", "linear", "percent", "gcd_lcm", "primality", "syllogism", "code")


def make_val(seed: int, per_kind: int, dmax: Optional[int] = None) -> Dict[str, List[dict]]:
    """Held-out validation drills.  ``seed`` must never be the evaluation seed."""
    mix = Mixture(seed)
    out: Dict[str, List[dict]] = {}
    for k in VAL_KINDS:
        r = random.Random(f"val:{seed}:{k}")
        f = mix.fns[k]
        arg = dmax if k not in ("between", "after") else (2000 if k == "between" else 1000)
        out[k] = [f(r, arg) for _ in range(per_kind)]
    return out


@torch.no_grad()
def validate(model, tok, val: Dict[str, List[dict]]) -> Dict[str, float]:
    from mnx.agent.skills.neural import run_tests
    from mnx.generate import complete
    was = model.training
    model.eval()
    res: Dict[str, float] = {}
    for kind, items in val.items():
        ok = 0
        for it in items:
            out = complete(model, tok, PROMPT.format(q=it["q"]),
                           max_new_tokens=120 if kind == "code" else 40, temperature=0.0)
            if kind == "code":
                tests = [l for l in it["q"].splitlines() if l.startswith("assert ")]
                ok += bool(tests) and run_tests(out.strip() + "\n", tests)[0]
            else:
                ok += out.strip().split("\n")[0].strip() == it["a"].strip()
        res[kind] = ok / max(1, len(items))
    if was:
        model.train()
    return res


# --------------------------------------------------------------------- train
@dataclass
class Args:
    ckpt: str = "runs/cpu"
    out: str = "runs/sft"
    hours: float = 3.0
    steps: int = 0
    lr: float = 2.0e-4
    min_lr: float = 1.0e-5
    warmup: int = 300
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    token_budget: int = 2048
    grad_accum: int = 2
    pool: int = 8192
    ema_decay: float = 0.999
    seed: int = 0
    threads: int = 4
    val_every: int = 2000
    val_n: int = 32
    val_seed: int = 90210
    save_every: int = 500
    log_every: int = 50
    resume: bool = False
    no_digit: bool = False
    dmax_override: int = 0
    history: bool = False
    history_out: str = ""


def _atomic(path: Path, blob) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(blob, tmp)
    tmp.replace(path)


def write_history(out: Path, dest: Optional[Path] = None) -> int:
    """Regenerate every question the run trained on, for ``eval_qa --exclude``."""
    st = json.loads((out / STATE_NAME).read_text())
    mix = Mixture(st["seed"], st.get("dmax_override") or None)
    n, plan = int(st["items"]), int(st["plan_items"])
    dest = dest or (out / "qa_history.jsonl")
    with dest.open("w", encoding="utf-8") as f:
        for i in range(n):
            it = mix.item(i, min(0.999, i / max(1, plan)))
            if it.get("in_eval_pool") is False:
                continue                      # out-of-range years: not in the evaluator's pool
            f.write(json.dumps({"q": it["q"]}) + "\n")
    return n


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for f, v in asdict(Args()).items():
        flag = "--" + f.replace("_", "-")
        if isinstance(v, bool):
            p.add_argument(flag, action="store_true")
        else:
            p.add_argument(flag, type=type(v), default=v)
    a = Args(**vars(p.parse_args(argv)))
    out = Path(a.out)

    if a.history:
        n = write_history(out, Path(a.history_out) if a.history_out else None)
        print(f"regenerated {n} trained questions")
        return 0

    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    out.mkdir(parents=True, exist_ok=True)
    resuming = a.resume and (out / CKPT_NAME).exists() and (out / STATE_NAME).exists()
    src = out if resuming else Path(a.ckpt)
    model, src_tok, _ = load_checkpoint(src)
    tok = src_tok if (resuming or a.no_digit) else make_digit_tokenizer(src_tok)
    tok.save(out / TOKENIZER_NAME)
    (out / CONFIG_NAME).write_text(json.dumps(model.cfg.to_dict(), indent=2))
    assert tok.vocab_size == model.cfg.vocab_size
    model.train()
    opt = build_optimizer(model, a.lr, a.weight_decay, (a.beta1, a.beta2))

    step, item0, plan = 0, 0, 0
    if resuming:
        blob = torch.load(out / CKPT_NAME, map_location="cpu")
        opt.load_state_dict(blob["optimizer"])
        st = json.loads((out / STATE_NAME).read_text())
        step, item0, plan = int(st["step"]), int(st["items"]), int(st["plan_items"])
        del blob
        print(f"resumed {out} at step {step}, item {item0}, plan {plan}")

    ema = {k: v.detach().clone() for k, v in model.state_dict().items()} if a.ema_decay else None
    stop_flag = {"v": False}
    signal.signal(signal.SIGINT, lambda *_: stop_flag.__setitem__("v", True))
    signal.signal(signal.SIGTERM, lambda *_: stop_flag.__setitem__("v", True))
    t0 = time.time()
    t_end = t0 + a.hours * 3600 if a.hours else None
    total_steps = {"v": a.steps or 20000}
    plan_items = {"v": plan or 1_400_000}
    items = item0

    def stop() -> bool:
        return stop_flag["v"] or (a.steps and step >= a.steps) or (t_end is not None and time.time() > t_end)

    def state() -> dict:
        return {"step": step, "items": items, "plan_items": plan_items["v"], "seed": a.seed,
                "total_steps": total_steps["v"], "dmax_override": a.dmax_override,
                "digit_tokenizer": not a.no_digit}

    def save() -> None:
        _atomic(out / CKPT_NAME, {"model": model.state_dict(), "optimizer": opt.state_dict(),
                                  "step": step, "config": model.cfg.to_dict(), "best_val": 0.0})
        if ema is not None:
            _atomic(out / EMA_NAME, {"model": ema, "step": step,
                                     "config": model.cfg.to_dict(), "best_val": 0.0})
        tmp = out / (STATE_NAME + ".tmp")
        tmp.write_text(json.dumps(state(), indent=1))
        tmp.replace(out / STATE_NAME)

    val = make_val(a.val_seed, a.val_n, a.dmax_override or None)
    log = (out / "log.csv").open("a" if resuming else "w", encoding="utf-8")
    if not resuming:
        log.write("step,lr,loss,items,tokens_per_s\n")
    mix = Mixture(a.seed, a.dmax_override or None)
    stream = batch_stream(mix, tok, model.cfg.max_seq_len, a.token_budget, a.pool,
                          item0, lambda: plan_items["v"], stop)
    opt.zero_grad(set_to_none=True)
    micro, run_loss, seen_tok = 0, 0.0, 0
    for x, y, items_now in stream:
        _, loss, _ = model(x, y)
        (loss / a.grad_accum).backward()
        run_loss += float(loss.item()) / a.grad_accum
        seen_tok += x.numel()
        micro += 1
        if micro < a.grad_accum:
            continue
        torch.nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip)
        for g in opt.param_groups:
            g["lr"] = cosine_lr(step, total_steps["v"], a.lr, a.min_lr, a.warmup)
        opt.step()
        opt.zero_grad(set_to_none=True)
        micro, step, items = 0, step + 1, items_now
        if ema is not None:
            d = min(a.ema_decay, (step + 1) / (step + 10))      # warm up the average
            with torch.no_grad():
                for k, v in model.state_dict().items():
                    ema[k].mul_(d).add_(v.detach(), alpha=1 - d)
        if step == 200 and not a.steps and not resuming:
            # size the schedule and the curriculum from the measured rate, once
            rate_steps = 200 / (time.time() - t0)
            rate_items = (items - item0) / (time.time() - t0)
            total_steps["v"] = int(max(400, rate_steps * (t_end - t0 - 120)))
            plan_items["v"] = int(max(10000, rate_items * (t_end - t0 - 120)))
            print(f"[plan] {rate_steps:.2f} steps/s -> {total_steps['v']} steps, "
                  f"{plan_items['v']} items", flush=True)
        if step % a.log_every == 0:
            tps = seen_tok / (time.time() - t0)
            msg = (f"step {step:>6} lr {opt.param_groups[0]['lr']:.2e} loss "
                   f"{run_loss / a.log_every * a.grad_accum:.4f} items {items} "
                   f"{tps:.0f} tok/s dmax {stage_dmax(items / max(1, plan_items['v']), a.dmax_override or None)}")
            print(msg, flush=True)
            log.write(f"{step},{opt.param_groups[0]['lr']:.6e},"
                      f"{run_loss / a.log_every * a.grad_accum:.5f},{items},{tps:.0f}\n")
            log.flush()
            run_loss = 0.0
        if step % a.save_every == 0:
            save()
        if a.val_every and step % a.val_every == 0:
            v = validate(model, tok, val)
            print("[val %d] " % step + "  ".join(f"{k}={s:.0%}" for k, s in v.items()), flush=True)
            log.write(f"# val {step} {json.dumps(v)}\n")
            log.flush()
        if stop():
            break
    save()
    n = write_history(out)
    v = validate(model, tok, val)
    print(f"[final step {step}, {items} items, {n} questions in history] "
          + "  ".join(f"{k}={s:.0%}" for k, s in v.items()), flush=True)
    log.write(f"# val {step} {json.dumps(v)}\n")
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

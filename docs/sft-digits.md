# One digit per token: why the arithmetic would not come

Six training levers moved MNX's unseen HumanEval score by nothing, and 6,735
trainer-bot steps moved its arithmetic from 3/63 to 4/63 at best and to 0 at
worst. The reason turned out to be upstream of every one of them.

## The finding

`runs/cpu/tokenizer.json` holds roughly 1,150 pure multi-digit tokens out of
about 3,355 merges — a third of the merge table is runs of digits. So

```
Q: what is 1234 plus 5678
```

does not reach the model as four digits, a word, and four digits. It reaches it
as a couple of opaque symbols whose relationship to 1, 2, 3 and 4 the model has
to infer, with no column alignment between the operands and no column alignment
with the answer it has to emit. Carrying is close to unlearnable in that
representation, and no learning rate fixes a representation.

Three arena competitors found this separately, from different strategy cards,
each measuring it in this repository rather than asserting it. They agreed on two
further points: the trainer bot spends over 90% of its CPU greedily quizzing
itself instead of learning, and the measured 54% → 6% collapses are explained by
the learning rate *per supervised token* — three gradient passes over one batch
of four items is 7.6× pretraining's effective rate — not by the nominal 1e-4.
Their write-ups are kept under `.arena/`.

## The change

`scripts/sft_digits.py`, and nothing else. `mnx/model.py`, `mnx/train.py`,
`mnx/tokenizer.py`, `mnx/data.py`, `mnx/trainer_bot.py`, `mnx/agent/` and
`scripts/eval_qa.py` are untouched.

**The tokenizer.** The run directory gets the checkpoint's own tokenizer with one
field changed: the pre-tokenizer pattern cuts digit runs into single digits
(`| ?\d` instead of `| ?\d+`). Same vocabulary, same merges, same ids for every
non-numeric string, and decoding stays byte-exact — `'2026-10-26'` goes from 6
tokens to 10 and comes back as itself. `load_checkpoint` reads `tokenizer.json`
out of the run directory, so the scorer picks the new segmentation up with no
change to the scorer.

**No quizzing.** One gradient pass over packed 256-token windows, loss on answer
tokens only, prompt and padding at `-100`. That is where the compute belongs.

**A fixed mixture.** 25% of items are maintenance, drawn from
`mnx.agent.qa_stream` exactly as pretraining drew them, all fifteen kinds, code
included. The other 75% come from wider samplers for the sub-skills that are both
weak and reachable: signed add and subtract to seven digits with deliberate carry
chains, division, multiplication where one factor is small, date arithmetic, two
by two systems, gcd, linear equations, percentages. What is *absent* from the
targeted share matters as much: units, powers, sequences, leap years, quadratics
and factorising appear only at their pretraining frequency, because they either
cannot earn a point on this protocol or cannot be computed at 10M parameters.

The maintenance share is not politeness. 100 of the baseline's 107 points are code
(80), syllogism (14) and primality (6). The downside of a bad recipe is −100 and
the upside is about +15, and that asymmetry decides every trade-off in the
recipe.

## Running it

```bash
python scripts/sft_digits.py --ckpt runs/cpu --out runs/digit \
    --steps 9000 --batch-size 16 --seq-len 256 \
    --lr 2e-4 --min-lr 2e-5 --warmup 200 --maintenance 0.25 \
    --save-interval 250 --threads 4 --seed 0
```

Re-running the identical command resumes from `runs/digit/ckpt.pt` with the
optimizer state; SIGINT and SIGTERM save atomically and exit. Measured throughput
on four CPU threads with nothing else running is about 1.4 s/step, so 9,000 steps
is about three and a half hours, not the 2h35m the recipe estimated.

Every question trained on is appended to `runs/digit/qa_history.jsonl`, and the
scorer is given that file, so nothing the run saw can be scored:

```bash
python scripts/eval_qa.py --ckpt runs/digit --n 200 --seed 777 \
       --exclude runs/digit/qa_history.jsonl
```

## What would falsify it

Written down before the run, from the competitor's own list:

| falsifier | what it would mean |
|---|---|
| division with gcd 1 still under 60% on dev at step 3,000 | the digit-alignment thesis is wrong: copying digits is the easiest thing in the mixture, and if that does not happen nothing harder will |
| add and subtract up to three digits under 50% on dev at step 6,000 | column addition is not learnable at this scale on this budget |
| `code` on dev below 90% of its base value at 3,000 and not recovered by 6,000 | the rehearsal share is too small, caught 6,000 steps before the final measurement rather than after it |
| final seed-777 total at or below 107/200 | the recipe failed — and 9,000 steps is already 1.4× the whole pretraining token budget, so "needs more steps" is not the answer |
| per-kind counts at seed 777 differ from the baseline row | the exclusion moved the item set, and the comparison has to be reported as such rather than banked |

The baseline to beat was re-measured twice on the day of the run and came out
identical both times: **107/200**, with code 80/80, syllogism 14/14, primality
6/6, gcd_lcm 3/11, arithmetic 3/63, dates 1/17, system2 0/6, and zero for
factorise, linear and percent.

# The 24/7 CPU Q&A trainer bot

`mnx/trainer_bot.py` is Tharun's "24/7 Autonomous CPU Q&A AI Trainer Bot"
design, implemented against MNX's own transformer and its verified Q&A
generators. Instead of reading a fixed corpus front to back, it keeps quizzing
the model, grades the answers, and spends its gradient steps where the model is
still wrong.

```bash
python -m mnx.trainer_bot --ckpt runs/cpu --workdir runs/bot          # runs until stopped
python -m mnx.trainer_bot --workdir runs/bot --status                 # what it has done
kill -INT <pid>                                                       # saves, then exits
```

## The loop (one step)

1. **Batch**: `--replay-ratio` of the batch comes from the replay buffer
   (hardest first), the rest is fresh from the generator hub at the current
   curriculum level.
2. **Pre-update quiz**: the model answers `Q: ...\nA:` greedily, before any
   weight changes.
3. **Grade**: `grade_prediction` scores each answer on `[0, 1]` as token F1
   (65%) plus exact numeric match (35%). An answer with the wrong number scores
   zero on that part, so "close" is not rewarded.
4. **Adaptive epochs**: a batch averaging below `--pass-mark` (0.65) gets
   `--active-epochs` updates; a batch above it gets one maintenance update.
5. **Verify**: the same batch is re-quizzed to measure `score_before ->
   score_after` (`--no-verify` skips this and is about a third faster).
6. **Replay priority**: `max(0.05, loss * mistake_boost + novelty_boost)`,
   where `mistake_boost` is 1.5 below a score of 0.6 and `novelty_boost` is
   `1 / (1 + 0.15 * times_seen)`, so a question the model keeps failing keeps
   coming back and a question it has seen twenty times fades.
7. **Curriculum**: every ten steps the level moves on the 50-step rolling
   score — up at 0.75, down at 0.45.

## Levels

| Level | Q&A it draws from |
|---|---|
| 1 | arithmetic, leap years, unit conversion |
| 2 | + powers, percentages, gcd/lcm, primality, sequences |
| 3 | + linear equations, factorising, syllogisms |
| 4 | + quadratics, 2x2 systems, date arithmetic |
| 5 | + writing code to pass asserts |

Held-out function names are excluded by `qa_stream`, so the bot never trains
on an evaluation task.

## Files it writes

| Path | What it holds |
|---|---|
| `checkpoints/latest_model.pt` | model + optimizer, written atomically every `--save-interval` steps and on SIGINT |
| `checkpoints/bot_state.json` | step, level, rolling scores, the whole replay buffer |
| `training_metrics.jsonl` | one row per step |
| `qa_history.jsonl` | every question, the gold answer, what the model said, the score |

Restarting with the same `--workdir` resumes from both files, so a 24/7 run
survives being killed, and `--cpu-threads` / `--sleep-ms` keep the machine (or
a phone in Termux) usable while it runs.

## Extra questions from disk

`--data-path ./my_qa.jsonl` (or a folder) is a hot folder: drop
`{"q": ..., "a": ...}` lines in while the bot runs and it picks them up on the
next step without restarting.

## Two rules that came out of actually running it

**Replay must cross the curriculum.** The first run filtered replay by the
current level, which looks reasonable and is ruinous: an hour of level-1
training (arithmetic, leap years, units) took the model from **54% to 6%** on
the unseen Q&A set, because nothing reminded it of the code and logic it
already answered perfectly (code 100% -> 2%, syllogisms 100% -> 0%; the two
score sheets are in [evidence/trainer-bot-forgetting.md](evidence/trainer-bot-forgetting.md)). Fresh
questions come from the current level; replay comes from every kind. That is
what "prevent catastrophic forgetting" has to mean here. `--warm-start` seeds
the buffer with questions from every kind before step 1, so maintenance covers
the model's existing skills from the start.

**Hopeless questions have to be retired.** Prioritized replay keeps
resurfacing what the model fails, and exact multi-digit arithmetic is not
learnable at 10M parameters, so the budget drains into impossible items and
the rolling score sits flat. `--retire-after` (default 8) drops a question
after that many failures.

## What it does and does not improve

It improves what it measures: answering the generated Q&A. It does **not**
teach the 10M model to program — HumanEval needs knowledge this curriculum
does not contain, and a 10M-parameter model trained on a CPU is far below the
size where coding ability appears. The honest coding numbers remain the ones in
the README: MNX1's verified coder, and the optional
[local model](local-model.md) when wider coverage is wanted.

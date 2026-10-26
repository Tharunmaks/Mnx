# What three competitors found, separately

These are the arena write-ups as their authors left them, unedited. Each
competitor got the identical task — "find a CPU training recipe that actually
improves MNX's 10M model" — and a different strategy card, and worked without
seeing the others.

- `recipe-a001-representation.md` — leads on the representation change and the
  per-kind headroom accounting; estimates 125–140/200.
- `recipe-a002-digit-sft.md` — the most prescriptive: one new file, a full knob
  table with a reason per knob, a measurement plan and five falsifiers. This is
  the one that was implemented, as `scripts/sft_digits.py`.
- `recipe-a004-digit-aligned-sft.md` — the most careful accounting: an abort
  gate, the learning-rate-per-supervised-token table that explains both earlier
  collapses, and re-scoring the base checkpoint under the identical `--exclude`
  so the comparison stays valid; estimates ~120/200.

They converge, independently, on the same two causes: a tokenizer that merges
runs of digits, and a trainer bot that spends its CPU quizzing rather than
learning. `docs/sft-digits.md` is the short version and what was done about it.

A fourth competitor was still working when the account's session limit stopped
it, and the bracket was never judged — the limit had already wiped out two
sixteen-competitor runs overnight. The recipe was chosen by reading all three.

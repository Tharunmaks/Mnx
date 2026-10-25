# The agent registry: 10,309 small agents

MNX1's specialists are the agents that reason: the coder, the mathematician,
the reasoner, the researcher. The registry is the other half of the picture —
a very large number of tiny agents, each doing exactly one thing, each able to
prove it does it.

```bash
python -m mnx.agent.registry stats
python -m mnx.agent.registry search "convert celsius to fahrenheit"
python -m mnx.agent.registry run unit.celsius_to_fahrenheit 100     # 212.0
python -m mnx.agent.registry verify --sample 300
python -m mnx.agent.registry build --scale 7                        # rebuild from scratch
```

MNX1 uses them by itself, through the `registry` skill:

```
you> top 3 of [5, 1, 9, 7, 3]
mnx> [9, 7, 5]
     From the agent `list.top_3` — The 3 largest numbers in a list, largest first.
     Tests it passed just now: assert top_3(list(range(100))) == [99, 98, 97]
```

## What is in it

| Family | Agents | What they do |
|---|---:|---|
| `number` | 3,147 | divisibility and remainders for each divisor, powers of each base |
| `bits` | 1,768 | set, clear, toggle and test each bit position |
| `list` | 1,747 | top and bottom k, chunk by k, every kth, rotate by k |
| `text` | 1,120 | pad and truncate to each width |
| `date` | 846 | k days before and after a date, weekday, quarter, week number, leap year, days in month |
| `stdlib` | 716 | real standard-library functions with their docstrings (**reference tier**) |
| `unit` | 548 | every ordered pair inside ten dimensions, plus the temperature scales |
| `recipe` | 164 | the tested recipes in `data/code` |
| `stats` | 105 | mean, median, mode, variance, standard deviation, range, every percentile |
| `base` | 70 | to and from every base from 2 to 36 |
| `cipher` | 25 | Caesar with each shift |
| `digits`, `math` | 53 | digit at each place, factorial, Fibonacci, divisors, prime factors, totient, integer square root |
| **total** | **10,309** | of which **9,593 verified** and 716 reference |

## The two tiers, and why the line matters

**verified (9,593).** A function plus asserts that *ran and passed while the
registry was being built*. Nothing is written to the manifest until its tests
pass: the build rejected agents until the generators were right, and
`verify --sample` re-runs a random sample any time. MNX1 may answer with a
verified agent's result, and when it does it re-runs that agent's asserts first
and prints them with the answer.

**reference (716).** Real standard-library functions, found by walking the
installed stdlib, with their own docstrings. Nothing ran them, so MNX1 only
points at them: "from the standard library (I haven't run these, so check
them)". That line is the whole value of keeping the tiers apart.

Recipe agents are checked in the coder's `python -I` sandbox, because a
recipe's own test block is arbitrary code — a couple of them start a server or
read input, which has no business running inside the build.

## How MNX1 chooses

1. BM25 over every agent's name, description and tags.
2. Re-rank near-ties with two signals from the question itself: a number in the
   agent's name that appears in the question counts for a lot (`percentile_90`
   over `percentile_9`), and name words appearing in the question *in order*
   count for a little (`celsius_to_fahrenheit` over `fahrenheit_to_celsius`).
3. Read the arguments out of the question. List literals first, then dates,
   strings and numbers; the number already baked into the agent's name is not
   also its argument, so "to base 7 of 100" calls `to_base_7(100)`, not
   `to_base_7(7)`.
4. If the best two agents are still close, run both: two agents that return the
   same answer are two witnesses, not an ambiguity. If they disagree, say
   nothing.
5. Re-run the chosen agent's asserts, then call it. Any exception means no
   answer.

It stays out of the coder's way on purpose: a question with a `def`, an
`assert`, a code fence, or "write a function..." is never answered from the
registry, because the coder verifies against *your* tests. This was not
theoretical — the first wiring answered two held-out coding tasks from the
registry and got both wrong. The guard, and a confidence that sits below the
verified specialists, fixed it; `tests/test_registry.py` holds that case.

## Scale

`--scale` widens the numeric families: scale 1 is 2,905 agents, scale 5 is
7,841, scale 7 is the 10,309 shipped here. Be clear-eyed about what the extra
agents are: the same families with more parameter values (divisors up to 1,400,
bit positions up to 441, widths up to 560). They are genuinely distinct
functions with distinct outputs, every one executed and passing, but they are
not 10,000 distinct *ideas* — the distinct families number a few dozen, and
`data/registry/counts.json` always says exactly what the total is made of.

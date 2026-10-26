# The agent registry: 14,329 small agents

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
| `number` | 4,824 | divisibility and remainders per divisor, powers of each base, gcd / lcm / coprimality against each modulus |
| `list` | 2,307 | top and bottom k, chunk by k, every kth, rotate by k, kth largest and kth smallest |
| `bits` | 1,768 | set, clear, toggle and test each bit position |
| `text` | 1,344 | pad and truncate to each width, the first k words, snake / kebab / camel / Pascal / title case, slugify, word and vowel counts, initials, palindromes, character frequencies |
| `window` | 1,116 | moving average, rolling sum, rolling max and rolling min over each window length |
| `date` | 846 | k days before and after a date, weekday, quarter, week number, leap year, days in month |
| `stdlib` | 716 | real standard-library functions with their docstrings (**reference tier**) |
| `unit` | 548 | every ordered pair inside ten dimensions, plus the temperature scales |
| `sequence` | 13 | nth prime, next prime, the first n primes, Lucas, triangular, pentagonal, hexagonal, Catalan, tribonacci, primorial, double factorial, Collatz length, perfect numbers |
| `percent` | 297 | p percent of a number, a p percent discount and a p percent markup, for every p |
| `recipe` | 164 | the tested recipes in `data/code` |
| `stats` | 115 | mean, median, mode, variance, standard deviation, range, every percentile, geometric and harmonic means, MAD, IQR, sample stdev, coefficient of variation, running totals, normalising, distinct counts |
| `base` | 70 | to and from every base from 2 to 36 |
| `round` | 55 | round to k decimal places; round, floor and ceil to each round multiple |
| `geometry` | 44 | areas, perimeters, volumes and surface areas of the common shapes, Heron's formula, point distance, every regular polygon from 3 to 30 sides |
| `digits`, `math`, `cipher` | 78 | digit at each place, factorial, Fibonacci, divisors, prime factors, totient, integer square root, Caesar with each shift |
| `matrix` | 10 | transpose, row and column sums, trace, diagonal, flatten, shape, rotate, 2x2 and 3x3 determinants |
| `validate` | 6 | Luhn, ISBN-10, IPv4, hex colour, email shape, digits-only |
| `set` | 6 | intersection, union, difference, symmetric difference, Jaccard similarity, subset |
| `roman` | 2 | to and from Roman numerals |
| **total** | **14,329** | of which **13,613 verified** and 716 reference |

## The two tiers, and why the line matters

**verified (13,613).** A function plus asserts that *ran and passed while the
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

1. BM25 twice: once over the question as asked, once over the question with its
   numbers removed, and the two result sets merged. A number is a rare token, so
   BM25 treats it as highly informative — "prime factors of 360" put
   `is_coprime_with_360` first and `prime_factors_of` nowhere at all. Searching
   the words on their own puts the right agent back in the running.
2. Re-rank with two signals from the question itself. A parameter baked into the
   agent's name that appears as a number in the question counts for a lot
   (`percentile_90` over `percentile_9`) — but only when one of the words the
   agent is *called* also appears in the question, because matching the number
   alone is how `is_coprime_with_360` won a question about prime factors. Name
   words appearing in the question *in order* count for a little
   (`celsius_to_fahrenheit` over `fahrenheit_to_celsius`).
3. Read the arguments out of the question, per agent. Bracket spans first and
   whole, so `[[1,2],[3,4]]` is one matrix rather than its first row; then
   dates, quoted strings, tokens that are data in their own right
   (`192.168.0.1`, an address with an `@`, a `#rrggbb` colour), then numbers.
   The parameter already baked into the agent's name is dropped once from what
   is left, wherever it sits in the name: "to base 7 of 100" calls
   `to_base_7(100)`, and "25 percent of 80" calls `percent_25_of(80)`.
4. Drop the matches the question gives nothing to run on, instead of giving up
   on the question. `is_coprime_with_360` has no argument left once 360 is its
   parameter, and dropping it is what lets the agent behind it be found.
5. Run the best few, re-running each one's own asserts first. An agent that
   fails its tests or raises on these arguments is the wrong agent, not a dead
   end: "to roman numeral 1994" matches `from_roman_numeral` first, that one
   raises on a number, and `to_roman_numeral` gets its turn.
6. If the best two are still close, compare what they returned — each on *its
   own* arguments, so this weighs 25% of 80 against 80% of 25. Two agents with
   the same answer are two witnesses, not an ambiguity. If they disagree, say
   nothing.

Every rule above except the first two was written because a question got a
wrong answer or no answer at all; `tests/test_registry.py` holds each case.

It stays out of the coder's way on purpose: a question with a `def`, an
`assert`, a code fence, or "write a function..." is never answered from the
registry, because the coder verifies against *your* tests. This was not
theoretical — the first wiring answered two held-out coding tasks from the
registry and got both wrong. The guard, and a confidence that sits below the
verified specialists, fixed it; `tests/test_registry.py` holds that case.

## Scale

`--scale` widens the numeric families: scale 1 is 3,865 agents, scale 5 is
10,840, scale 7 is the 14,329 shipped here. Be clear-eyed about what the extra
agents are: the same families with more parameter values (divisors up to 1,400,
bit positions up to 441, widths up to 560). They are genuinely distinct
functions with distinct outputs, every one executed and passing, but they are
not 14,000 distinct *ideas* — the distinct families number a few dozen, and
`data/registry/counts.json` always says exactly what the total is made of.

Not every family is parameterised, and one that was had to be taken back out. A
`first_360_primes()` family looked like 420 free agents, and it was a trap: it
takes no argument, so any question with 360 anywhere in it made one of them look
runnable, and "prime factors of 360" started answering with a list of the first
360 primes. One `first_n_primes(n)` is worth more than 420 that bake n in. The
same rule killed the packed-tuple signatures: an agent that wants two lists has
to say `(a, b)`, because `(pair)` cannot be read out of a question.

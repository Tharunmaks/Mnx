# MNX

MNX is a small AI assistant built from scratch. It uses **no pretrained
language model and no API key**, and the agent runs on the Python standard
library alone, offline, in milliseconds.

```text
$ python -m mnx.agent "solve x^3 - 6x^2 + 11x - 6 = 0"
x = 1 or x = 2 or x = 3

$ python -m mnx.agent "All dolphins are mammals. All mammals breathe air. Flipper is a dolphin. Does Flipper breathe air?"
Yes, Flipper breathes air.

$ python -m mnx.agent "[1,2,3,4] -> [2,4], [5,6,7,8,10] -> [6,8,10], [11,12,14] -> ?"
def f(x):
    return [e for e in x if e % 2 == 0]
f([11, 12, 14]) = [12, 14]

$ python -m mnx.agent "Who will win the next election?"
I don't know. None of my skills could produce an answer I trust, ...
```

The repository also contains **MNX 2.0 Max Coder**, a from-scratch PyTorch
transformer for code (documented in [docs/coder.md](docs/coder.md)). The agent
can use it as an optional proposer whose samples are checked by tests.

---

## The honest answer to "smarter than the big models"

A small system built from scratch will not beat frontier LLMs *in general*.
Those models gain breadth from trillions of training tokens and billions of
parameters, and nothing in this repository replaces that. It can't write an
essay, translate, summarise a book or chat about anything at all.

What a small system *can* do is beat them on **narrow tasks where the answer
can be computed or checked**, and be more trustworthy everywhere else:

| | Large LLM | MNX |
|---|---|---|
| `123456789 × 987654321`, `2^200`, `100!` | often slips a digit | exact (rational arithmetic, big integers) |
| Solving equations and linear systems | usually right, unverified | exact, each root substituted back as a check |
| Syllogisms and if-then deduction | can be talked into invalid conclusions | sound forward chaining with a printed proof, and "cannot be determined" when it can't be |
| Inferring a program from examples | guesses | exhaustive search, re-runs the program on every example |
| Facts it doesn't have | may hallucinate | says "I don't know" |
| Open-ended language, broad world knowledge | strong | **not supported** |
| Cost | GPUs or a paid API | a laptop CPU, no network |

The aim, then, is a small system that is **always right or silent** on what it
covers, and easy to extend.

## Research: what makes small models punch above their weight

The design follows results from the last few years showing that model size
can be traded for **external knowledge, tools, search and verification**:

1. **Retrieval instead of memorisation.** RAG (Lewis et al., 2020), RETRO
   (Borgeaud et al., 2021) and Atlas (Izacard et al., 2022) showed that models
   which look facts up in a corpus can match much larger closed-book models on
   knowledge-heavy tasks. Facts in a text file are also easy to inspect,
   correct and cite. → `mnx/agent/retrieval.py` (BM25 plus sentence extraction).
2. **Tools and symbolic solvers.** Toolformer (Schick et al., 2023) and
   program-aided LMs (PAL; Gao et al., 2022) hand arithmetic to a calculator
   or interpreter instead of the network. AlphaGeometry (Trinh et al., 2024)
   pairs a language model with a symbolic deduction engine, and the engine
   does most of the proving. → exact algebra, units, calendar and logic skills.
3. **Test-time search with a verifier.** Verifiers for maths word problems
   (Cobbe et al., 2021), self-consistency voting (Wang et al., 2022) and
   AlphaCode's sample-then-filter approach (Li et al., 2022) all show that
   generating many candidates and *checking* them works better than trusting
   a single guess. Snell et al. (2024) found that extra test-time compute can
   beat a much larger model. → program synthesis (`synth.py`), held-out-term
   checks for sequences, and the neural sample-and-verify skill.
4. **Program synthesis from examples.** FlashFill (Gulwani, 2011), bottom-up
   enumeration with observational-equivalence pruning (Udupa et al., 2013;
   Albarghouthi et al., 2013) and DreamCoder (Ellis et al., 2021) learn exact
   programs from a handful of examples, which is far more data-efficient than
   gradient descent. → `mnx/agent/skills/synth.py`.
5. **Small trained transformers on good data.** TinyStories (Eldan & Li,
   2023) and phi-1, "Textbooks Are All You Need" (Gunasekar et al., 2023),
   showed that small models trained on clean, focused or distilled data
   (Hinton et al., 2015) are surprisingly capable within their domain. → the
   existing `MNXCoder` transformer, used here only as a *proposer*, never as
   the final authority.
6. **Knowing when not to answer.** Selective prediction (answering only above
   a confidence threshold) turns a mediocre model into a reliable one on the
   questions it does answer. → every skill returns a calibrated confidence,
   and the agent abstains below a threshold.

## Architecture

```text
question
   │
   ▼
router ── multinomial naive Bayes on words, bigrams and shape tokens
   │       (<num>, <op>, <eq>, <arrow>, <numlist>, <date>, <syllogism>…),
   │       trained from scratch in milliseconds on data/router/intents.tsv
   ▼
skills, tried in router order (each may return a candidate or decline)
   ├─ math       hand-written parser → exact Fractions; polynomials; rational
   │             root theorem, quadratic formula, Durand–Kerner; Gaussian
   │             elimination; Miller–Rabin and Pollard's rho; d/dx, ∫, factor
   ├─ units      exact conversion factors; affine temperatures
   ├─ dates      weekday, differences, offsets, leap years (checked against
   │             Zeller's congruence)
   ├─ logic      English premises → rules → forward chaining with modus
   │             ponens and modus tollens; proofs, contradiction detection
   ├─ sequence   hypotheses from simplest to most complex (constant, known
   │             sequences, geometric, polynomial, linear recurrences,
   │             differences, ratios, interleaved), checked on held-out terms
   ├─ synth      bottom-up program search in a typed DSL (~75 operators,
   │             map, filter, conditionals), observational equivalence, then
   │             weighted voting over all consistent programs (2^-size)
   ├─ memory     "remember that ..." adds facts to the index (optional JSONL)
   ├─ knowledge  BM25 over data/knowledge/*.md (plus your own --kb folders),
   │             sentence extraction, query expansion, cited sources
   ├─ chat       templates for greetings and help
   └─ neural-code (optional) sample MNXCoder completions and keep one that
                 passes the asserts in a sandboxed subprocess
   │
   ▼
rank candidates by confidence + verification bonus + router prior
   │
   ▼
answer with a trace (router, proof, checks, sources), or "I don't know"
```

Every skill tries to **verify its own answer**. Equation roots are
substituted back. Factorisations are multiplied out. Unit conversions are
converted back. Weekdays are cross-checked against a second algorithm.
Synthesised programs are re-executed. Each sequence rule is fitted on the
fewest terms it needs and must then predict the rest. Run with `--trace` to
see these steps.

## Quickstart

```bash
# no dependencies needed for the agent (pip install -e . adds an `mnx-agent` command)
python -m mnx.agent "what is 15% of 240"
python -m mnx.agent --trace "solve x + y = 10, x - y = 2"
python -m mnx.agent                         # interactive; ':trace' toggles traces

# your own knowledge and persistent memory
python -m mnx.agent --kb ~/notes --memory ~/.mnx/memory.jsonl
you> remember that the standup is at 9:30
you> when is the standup?
```

Python API:

```python
from mnx.agent import MNXAgent

agent = MNXAgent()                 # knowledge_dirs=[...], memory_path=..., ckpt=...
ans = agent.ask("factor x^2 - 5x + 6")
print(ans.text, ans.verified, ans.confidence)
print("\n".join(ans.trace))
```

Things to try:

| Skill | Example |
|---|---|
| math | `2^100`, `1/3 + 1/6`, `solve 2x^2 - 8 = 0`, `x^5 - x - 1 = 0`, `is 7919 prime`, `prime factors of 600851475143`, `expand (x+1)^3`, `derivative of x^3 + 2x`, `integrate 3x^2`, `20 choose 10` |
| units | `convert 26.2 miles to km`, `98.6 F in C`, `how many ounces in a pound`, `1 TiB in GB` |
| dates | `what day of the week was July 20, 1969`, `how many days between 2024-01-01 and 2024-12-25`, `what date is 100 days after 2024-01-01` |
| logic | `If something is a dog then it is loyal. Rex is not loyal. Is Rex a dog?` |
| sequence | `1, 3, 7, 15, 31, ?`, `1, 2, 4, 7, 11, 16, next` |
| synth | `'John Smith' -> 'JS', 'Ada Lovelace' -> 'AL', 'Alan Turing' -> ?`, `f(3)=6, f(4)=24, f(5)=120, f(6)=?` |
| knowledge | `who discovered penicillin`, `what is the time complexity of binary search` |

### Optional: the neural proposer

With PyTorch installed and a checkpoint trained as in
[docs/coder.md](docs/coder.md), the agent can propose code with the
transformer and keep only a sample that passes your tests:

```bash
pip install torch && python -m mnx.train --config tiny --data data/sample --out runs/tiny --steps 300
python -m mnx.agent --ckpt runs/tiny $'def add(a, b):\n    """Return the sum of a and b."""\nassert add(2, 3) == 5\nassert add(-1, 1) == 0'
```

That example passes, but only because `add` is in the tiny model's 51-file
training set. The bundled model is a toy, so expect "None of 16 samples
passed" for anything new. The mechanism is the point here:
a better-trained model plugs straight in.

## Evaluation

```bash
python -m mnx.agent.evaluate                                     # tuning set
python -m mnx.agent.evaluate --benchmark data/eval/holdout.jsonl # held-out set
```

"Wrong" means the agent gave an answer and it was incorrect. That's the
number that matters for trust. "Abstain" means it said it didn't know.

| Set | Questions | Correct | Wrong | Abstained |
|---|---:|---:|---:|---:|
| `benchmark.jsonl` (written while building, used for tuning) | 83 | 83 (100%) | 0 | 0 |
| `holdout.jsonl` (written afterwards; first run, no tuning) | 45 | 40 (89%) | 1 | 4 |

The router's 5-fold cross-validated accuracy on its training data is 84%. A
routing mistake costs time but not correctness, because every skill decides
for itself whether it can answer.

Held-out misses, left unfixed so the numbers stay honest:

- **Wrong (1):** a logic question with a second question embedded mid-text
  (`... Is it true that Batty lays eggs? Batty is a bat. Does Batty lay
  eggs?`). The sentence splitter doesn't split on `?`, so the premise "Batty
  is a bat" was lost and it answered "cannot be determined".
- **Abstained (4):** `2, 4, 12, 48, 240` (ratios 2, 3, 4, 5 starting from a
  ratio of 2 is outside the hypothesis set), `'apple' -> 'APPLE!'` (the DSL
  has no string constants taken from the examples), and two knowledge
  questions whose wording didn't match the knowledge file ("percentage of
  the atmosphere", "what does RAM stand for").

Both sets were written by the same author as the system and cover only what
it was designed for, so **these numbers are not comparable to public LLM
benchmarks**. They show the system does what it claims within its scope:
zero hallucinated answers on the 15 out-of-scope questions, and one wrong
answer in 128.

## Limits

- **Narrow coverage.** Questions have to fit a skill's patterns. Rephrase
  into a supported form, or add a skill.
- **Shallow language understanding.** The parsers are regular expressions
  and a recursive-descent grammar, not general NLU. Logic handles
  categorical and if-then premises with unary predicates; it doesn't handle
  "some", relations ("taller than"), quantities or time.
- **Knowledge is only as good as the files.** The bundled knowledge base has
  about 200 facts. Point `--kb` at your own notes for anything else.
  Retrieval returns sentences; it doesn't combine facts across documents.
- **Program search is bounded.** It covers programs up to size 8 in its DSL
  and gives up after about 4 seconds. With few examples, the simplest
  consistent program may still not be the one you meant. The trace lists the
  alternatives, and the confidence drops when they disagree.
- **Numeric roots** for degree ≥ 3 polynomials without rational roots are
  floating point, checked against a residual tolerance.
- **No learning from use** beyond the memory file. The router is retrained
  from `intents.tsv` at start-up, and adding examples there is how you teach
  it new phrasings.

## Extending

A skill is a class with a `name` and an `answer(question) -> Answer | None`
method (see `mnx/agent/base.py`). Return `None` when the question isn't
yours, set `verified=True` only after an independent check, and add a few
routing examples to `data/router/intents.tsv`. Register the skill in
`MNXAgent.__init__`.

## Project layout

```text
mnx/agent/
  agent.py        MNXAgent: route → try skills → rank → answer or abstain
  __main__.py     CLI (python -m mnx.agent)
  base.py         Answer / Skill types, number formatting
  expr.py         from-scratch CAS core: tokenizer, parser, exact evaluator, polynomials, root finding
  router.py       naive Bayes intent router + cross-validation
  retrieval.py    BM25 index, sentence extraction, memory store
  evaluate.py     benchmark runner (python -m mnx.agent.evaluate)
  skills/         math, units, dates, logic, sequences, synth, knowledge, chat, neural
data/knowledge/   bundled knowledge base (markdown, one fact per sentence)
data/router/      router training examples
data/eval/        benchmark.jsonl (tuning) and holdout.jsonl (held out)
mnx/*.py          MNX 2.0 Max Coder transformer: tokenizer, model, data, train, generate
docs/coder.md     transformer documentation
tests/            pytest suite (agent tests need only the standard library)
```

## Tests

```bash
pip install pytest
pytest -q                     # agent tests always; transformer tests too if torch is installed
```

## License

MIT.

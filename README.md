# MNX

MNX is a small AI assistant built from scratch. It uses **no pretrained
language model and no API key**, and the agent runs on the Python standard
library alone, offline, in milliseconds.

**MNX1** is the orchestrator: it routes each question to a specialist agent
(coder, mathematician, reasoner, researcher, conversation), checks the answer,
and learns from every chat: your corrections, your feedback, and its own
verified answers. See a full [sample session](docs/sample-session.md) of
questions and the answers it gave.

```text
$ python -m mnx.agent "solve x^3 - 6x^2 + 11x - 6 = 0"
x = 1 or x = 2 or x = 3

$ python -m mnx.agent "All dolphins are mammals. All mammals breathe air. Flipper is a dolphin. Does Flipper breathe air?"
Yes, Flipper breathes air.

$ python -m mnx.agent "[1,2,3,4] -> [2,4], [5,6,7,8,10] -> [6,8,10], [11,12,14] -> ?"
def f(x):
    return [e for e in x if e % 2 == 0]
f([11, 12, 14]) = [12, 14]

$ python -m mnx.agent "write a python function that checks if a string is a palindrome"
def is_palindrome(s: str) -> bool:
    cleaned = [c.lower() for c in s if c.isalnum()]
    return cleaned == cleaned[::-1]
Tests it passed:
assert is_palindrome("A man, a plan, a canal: Panama")
...

$ python -m mnx.agent "generate an image of a cat"
I can't create images, video or audio: I'm a text-only assistant. ...

$ python -m mnx.agent "Who will win the next election?"
I don't know. None of my skills could produce an answer I trust, ...
```

The repository also contains **MNX 2.0 Max Coder**, a from-scratch PyTorch
transformer (documented in [docs/coder.md](docs/coder.md)). It can be trained
on a CPU from the **expert stream**, a database of 3 million verified
questions and answers that MNX1's agents generate, and exported to phones
with ExecuTorch ([docs/mobile.md](docs/mobile.md)). The agent uses it as an
optional proposer whose samples are checked by tests.

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

**Which model does it beat?** None on the public coding benchmark. On
[HumanEval](https://github.com/openai/human-eval) (164 Python problems,
pass@1) MNX1 solves 42 (25.6%) — but only **2 of the 82 problems nobody
looked at while building it** (see the warning under Evaluation). Against
published scores:

| Model | HumanEval |
|---|---:|
| GPT-4 ([technical report](https://arxiv.org/pdf/2303.08774), 0-shot) | 67.0% |
| GPT-3.5 (same table) | 48.1% |
| Qwen2-0.5B-Instruct ([model card](https://featherless.ai/models/Qwen/Qwen2-0.5B-Instruct/readme)) | 17.1% |
| Qwen1.5-0.5B-Chat (same card) | 9.1% |
| **MNX1** (`scripts/humaneval.py`, problems 82–163, never looked at) | **2.4%** |

Those numbers come from different evaluation setups, so treat them as rough.
What MNX1 does differently: of the 43 problems it answers, 42 are right (98%);
on the other 121 it says it isn't sure and shows a labelled guess or nothing,
where an LLM would always return code. It is far behind every LLM on breadth
and far ahead of them on not being wrong. The table above describes task types,
not benchmark results. `scripts/compare_models.py` runs any local model
through [Ollama](https://ollama.com) on MNX's own test sets with the same
scoring, so you can compare directly, for example against `qwen2.5:0.5b`.

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
   ├─ feedback ("correct" / "wrong, the answer is X")? → learn from it
   ├─ corrected before? → your answer, credited to you
   ├─ solved before by a slow agent? → replay the verified answer
   ▼
router ── multinomial naive Bayes on words, bigrams and shape tokens
   │       (<num>, <op>, <eq>, <arrow>, <numlist>, <date>, <syllogism>…),
   │       trained from scratch in milliseconds on data/router/intents.tsv
   ▼
specialist agents, each a group of skills tried in router order
(each skill may return a candidate or decline):
   coder: code, synth, neural-code   mathematician: math, units, dates, sequence
   reasoner: logic   researcher: knowledge, memory   conversation: chat
   ├─ code       the coding agent (below): verified recipes, tests run in a
   │             sandbox, synthesis from asserts, run and check your code
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
answer with a trace (agent, router, proof, checks, sources), or "I don't know"
   │
   ▼
learn: log the chat, self-train the router on verified answers, remember
solutions, save new verified code as recipes
```

### The coding agent

The coder never ships code it hasn't run.

- **Write a function.** "implement merge sort", "write a function called fib
  that returns the nth fibonacci number". It searches 165 hand-written recipes
  in `data/code/` (BM25 over the description, tags and name, with a check
  that "decimal to binary" doesn't match `binary_to_decimal`), renames the
  function to the name you asked for, and runs the recipe's own tests in an
  isolated subprocess (`python -I`, time limit) before answering.
- **Solve from a pasted function.** Give a `def` header with examples in its
  docstring (`>>> f(2)` then `6`, or `f(2) ==> 6`) and they become the tests.
- **Solve from tests.** Give a `def` header and some `assert` lines. It tries
  recipes, then program search over the DSL, then the neural proposer, and
  ships the first candidate that passes every assert. When several programs
  fit the tests but disagree on other inputs, it says so and lowers its
  confidence.
- **Run** code (`run this: ...`) in the sandbox and show the output, and
  **check** code for syntax errors and undefined names.
- Code it wrote and verified is saved as a new recipe in the state folder,
  so the next similar request is answered from memory.

### The agent registry

Besides the specialists, MNX1 carries a registry of **10,309 small
agents** (9,593 with asserts that passed at build time, 716 standard-library
pointers that were not run). It searches them by what the question is about,
re-runs the chosen agent's tests, applies it to the arguments in the question,
and prints the result with the tests behind it. It never answers a `def` plus
asserts from the registry — that belongs to the coder, which verifies against
your own tests. See [docs/registry.md](docs/registry.md).

### How MNX1 learns

Everything is kept in a state folder (`~/.mnx` by default, `--state` to
change it, `--no-learn` to turn it off):

| File | What it holds |
|---|---|
| `transcript.jsonl` | every question, the agent and skill that answered, and the answer |
| `corrections.jsonl` | "wrong, the answer is X": returned for that question (and close rephrasings) from then on |
| `learned_intents.tsv` | router examples from verified answers (confidence ≥ 0.9) and corrections, replayed into the router at start-up |
| `experience.jsonl` | verified answers from slow skills, replayed instantly next time |
| `feedback.jsonl` | "correct" / "wrong" votes |
| `recipes/` | code it wrote and verified |

This is lightweight learning (counts, a lookup store and new recipes), not
gradient training. It makes MNX1 better at the questions you actually ask,
and it never learns from an answer that wasn't verified or confirmed by you.

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
python -m mnx.agent                         # interactive; ':trace' toggles traces, ':stats' shows what it learned

# coding
python -m mnx.agent "implement binary search"
python -m mnx.agent $'def double(x):\n    \"\"\"Double it.\"\"\"\nassert double(2) == 4\nassert double(5) == 10'

# teach it
you> what is the capital of Wakanda?
mnx> I don't know. ...
you> wrong, the answer is Birnin Zana
you> what's the capital of wakanda?
mnx> Birnin Zana

# your own knowledge and persistent memory
python -m mnx.agent --kb ~/notes --memory ~/.mnx/memory.jsonl
you> remember that the standup is at 9:30
you> when is the standup?
```

Python API:

```python
from mnx.agent import MNX1, MNXAgent

agent = MNX1()                     # learns into ~/.mnx; MNXAgent(state_dir=None) learns for this session only
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
| code | `implement dijkstra`, `write a function that converts roman numerals to integers`, `run this: print(2**10)`, `check this code: ...` |

### Training the neural model on the expert stream

The agents generate training data they can vouch for, and the transformer
learns from it on a CPU:

```bash
pip install torch
python -m mnx.agent.qa_stream --n 3000000 --out runs/qa          # 3M distinct verified Q&A pairs, ~95 MB gzipped
python -m mnx.agent.qa_stream --check runs/qa                     # replay a sample through MNX1's agents
python -m mnx.agent.expert_stream --out runs/expert_stream.txt    # verified code: recipes, generated tasks, stdlib
python -m mnx.train --config cpu --stream runs/qa --data runs/expert_stream.txt --doc-sep '\n<|doc|>\n' \
    --out runs/cpu --steps 6500 --batch-size 16 --lr 2e-3 --warmup 150 --fim-rate 0 --threads 4
python -m mnx.agent --ckpt runs/cpu                               # the coder now samples from it
```

- **The database** covers arithmetic, powers, percentages, gcd/lcm,
  primality, factorisation, linear and quadratic equations, 2×2 systems,
  units, dates, sequences, syllogisms and small coding tasks with tests.
  Every answer is computed exactly, never written by hand, and every question
  is distinct. Replaying 2,000 random pairs through MNX1, which computes its
  answers separately, gave the same answer on all 1,826 it answered. The
  replay also found two real MNX1 bugs ("19 * 216 = ?" wasn't parsed, and
  "gcd of 413 and 497" read the numbers as 413,497); both are fixed.
- **Streaming training** reads, shuffles and tokenizes the database as it
  goes, so 3 million pairs never sit in memory. `--resume` continues where
  the last run stopped, which is how you train on all 3M over several runs.
- **Test-conditioned prompts.** Each training function has its asserts as
  comments above it, and the coder prompts the model the same way.
- **Voting.** The coder samples 16 programs, keeps the ones that pass your
  asserts, runs them on new probe inputs, and returns one only if most of
  them behave the same. A program that passed by luck is not shipped.

What one run on a 4-core CPU achieved (1 hour 48 minutes, 10.4M parameters,
26.6M tokens, about 500k of the 3M pairs, then 25 minutes on code only):

| Measure | Result |
|---|---|
| Validation loss | 8.2 → 0.85 (0.55 on code after the code fine-tune) |
| 600 unseen database questions, exact answer | 314 (52%): coding tasks 244/244, syllogisms 36/36, primality 17/17; arithmetic 6/152, dates 1/67 |
| `code_holdout2.jsonl` tasks, model alone, 16 samples | 3 of 20 solved |
| MNX1 with the model on both coding holdouts | the same as without it, and no new wrong answers |

So the model learned the formats and the patterns it can memorise, but not
arithmetic (a 10M model can't learn to multiply from examples; that's what
the math agent is for), and it doesn't yet add to MNX1's coding score. The
first measurement scored 0 of 20 because of a prompt bug: a prompt that ended
in a newline split a token the model had always seen whole. That bug is
fixed. A bigger model, the full 3M pairs and a GPU are the obvious next
steps.

### 24/7 active-learning trainer

`python -m mnx.trainer_bot --ckpt runs/cpu --workdir runs/bot` keeps quizzing
the model, grades each answer (token F1 plus exact numeric match), spends extra
gradient steps on the batches it fails, replays what it got wrong before, and
walks a five-level curriculum from arithmetic up to writing code. It
checkpoints atomically, so a run that is killed resumes where it stopped.
`--status` prints the rolling score by question type. See
[docs/trainer-bot.md](docs/trainer-bot.md).

### On a phone

Train on a PC, then `python -m mnx.export_mobile --ckpt runs/cpu --out mobile/ --check`
writes a 15 MB int8 ExecuTorch program for Android and iOS CPUs (XNNPACK).
The check reruns it with the ExecuTorch runtime and confirms it picks the
same next token as PyTorch. The agents themselves are pure Python and run in
Termux. See [docs/mobile.md](docs/mobile.md).

### Optional: a local code model as a proposer (off by default, no PC needed)

MNX1's own answers use no model and no network. If you want wider coverage
than recipes and program search reach, you can let it consult a small code
model running on **your own device** — a phone works, via llama.cpp in
Termux:

```bash
# in Termux, once
./llama.cpp/build/bin/llama-server -m ~/qwen-coder.gguf -c 4096 --port 8080 &
python -m mnx.agent --model qwen-coder --model-url http://localhost:8080/v1
```

Nothing the model says is trusted. MNX1 runs every candidate in its sandbox,
drops the ones that fail your asserts, and makes the survivors agree on fresh
probe inputs before it answers. When a problem arrives with no asserts at all,
it asks the model for test cases too, scores the candidates
[CodeT](https://arxiv.org/abs/2207.10397)-style, and says in the answer that
those tests were generated rather than given. Qwen2.5-Coder-1.5B-Instruct is
about 1.1 GB at 4-bit and scores 70.7 on HumanEval by itself, so this is the
realistic way to get a high coverage number out of MNX1 on a phone. Setup and
caveats: [docs/local-model.md](docs/local-model.md).

## Evaluation

```bash
python -m mnx.agent.evaluate                                     # tuning set
python -m mnx.agent.evaluate --benchmark data/eval/holdout.jsonl # held-out set
python -m mnx.agent.evaluate --benchmark data/eval/code_holdout.jsonl # held-out coding tasks
python -m mnx.agent.evaluate --benchmark data/eval/code_holdout2.jsonl --ckpt runs/cpu  # optional neural proposer
pip download human-eval --no-deps -d /tmp/he && unzip -o /tmp/he/human_eval-*.whl -d /tmp/he
python scripts/humaneval.py /tmp/he/human_eval/data/HumanEval.jsonl.gz                 # public benchmark
```

"Wrong" means the agent gave an answer and it was incorrect. That's the
number that matters for trust. "Abstain" means it said it didn't know.

| Set | Questions | Correct | Wrong | Abstained |
|---|---:|---:|---:|---:|
| `benchmark.jsonl` (written while building, used for tuning) | 83 | 83 (100%) | 0 | 0 |
| `holdout.jsonl` (written afterwards; first run, no tuning) | 45 | 40 (89%) | 1 | 4 |
| `code_holdout.jsonl` (15 coding tasks with hidden tests) | 15 | 12 (80%) | 0 | 3 |
| `code_holdout2.jsonl` (20 def + assert tasks) — *no longer held out, see below* | 20 | 9 (45%) | 1 | 10 |
| [HumanEval](https://github.com/openai/human-eval) problems 0–81 — *tuning half* | 82 | 40 (49%) | 0 | 42 |
| **[HumanEval](https://github.com/openai/human-eval) problems 82–163 — never looked at** | 82 | **2 (2.4%)** | 1 | 79 |

The coding holdout scored 10/15 with 2 wrong on its first run: "decimal to
binary" was answered with a `binary_to_decimal` recipe and "merge two
dictionaries" with `merge_sorted_lists`. Those were retrieval bugs, fixed
in general (a direction check and a higher match threshold), not by adding
recipes for the holdout tasks. The 4 it still declines are reverse_words,
last_item on mixed-type lists, shout (needs a `'!'` constant) and
merge_dicts. Learning is off during evaluation.

`code_holdout2.jsonl` was committed before the expert stream existed. On its
first run MNX1 scored 10/20 with 4 wrong: program search fitted the 1 or 2
asserts shown with the wrong program (`max_of_three` returned `max(a, b)`,
`average` returned `max(2, min(xs))`). HumanEval showed the same problem at
scale: 15 solved but 26 wrong on the first run. The fix is general, worked
out on HumanEval problems 0–81 only:

- When the programs that fit the asserts disagree on new inputs, the
  description votes: programs whose operations its words mention ("sum",
  "even", "reverse"…) count for more.
- A program that uses operations or constants the description never
  mentions is treated as a coincidence.
- If that still leaves doubt, MNX1 doesn't answer. It shows its best guess,
  labelled as checked only against your asserts, and asks for more asserts.

That took HumanEval from 15 solved / 26 wrong to 12 solved / 4 wrong, and
`code_holdout2` from 10/4 to 3/1. Fewer answers, far fewer wrong ones, in
line with "right or silent". With more asserts the search usually pins the
program down (the sample session's `mystery` puzzle uses 9).

The router's 5-fold cross-validated accuracy on its training data is 83%. A
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

**Read the HumanEval rows together.** The 49% on problems 0–81 is not a
capability measurement. `data/code` was grown from 78 to 165 recipes while
those problems were on screen, and several recipes cover the same classic
tasks they ask for (prefixes, running balances, prime factors). That is
memorisation of the tuning half, and it moved the untouched half only from
0 to 2. `code_holdout2.jsonl` stopped being a holdout for the same reason:
some of the new recipes (`average`, `clamp`, `is_sorted`) answer its tasks
directly, which is why it jumped from 15% to 45%. **The honest headline for
coding is 2 of 82 on problems 82–163.** Two further attempts moved it by
nothing: giving program search five times the budget solved one extra
problem in fifteen, and teaching the coder to read docstring examples let it
work from a pasted function alone (41 of 164 with no asserts supplied,
against 42 when they are) without changing the unseen score. The recipes are still worth having:
each is tested, and they make MNX1 useful on everyday requests. They just
don't make it a better programmer.

Both self-written sets cover only what the system was designed for, so
**those numbers are not comparable to public LLM benchmarks**. They show the system does what it claims within its scope:
zero hallucinated answers on the 15 out-of-scope questions, and 1 wrong
answer in 163.

## Limits

- **It is not a general programmer.** On HumanEval problems it had never
  seen, it solves about 1 in 40. It is strong on tasks that match a recipe
  or a small searchable program, and silent on the rest.
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
- **Coding is Python only, and bounded.** The coder writes code from its
  recipes and from program search, so it handles well-known functions and
  small functions specified by tests. It is not a general software engineer:
  it can't build a new app, refactor a codebase or write code it has no
  recipe or search space for. In those cases it says it can't, instead of
  guessing.
- **Learning is lightweight.** It learns corrections, phrasings and
  solutions, not new reasoning ability. A correction applies to that
  question and close rephrasings.
- **Not better than every model.** Inside its skills it is exact, checked
  and fast; outside them a large model will answer far more questions.
- **The local-model bridge is a bridge, not MNX1.** `--model` borrows
  coverage from somebody else's trained weights. With it, the score is mostly
  that model's; without it, the numbers above are MNX1's own. The one path
  where MNX1 ships code it verified only against tests it wrote itself is the
  no-asserts path described in
  [docs/local-model.md](docs/local-model.md), and it labels those answers.

## Extending

A skill is a class with a `name` and an `answer(question) -> Answer | None`
method (see `mnx/agent/base.py`). Return `None` when the question isn't
yours, set `verified=True` only after an independent check, and add a few
routing examples to `data/router/intents.tsv`. Register the skill in
`MNXAgent.__init__`.

## Project layout

```text
mnx/agent/
  agent.py        MNX1 / MNXAgent: shortcuts → route → specialist agents → rank → answer or abstain
  learning.py     transcript, feedback, corrections, self-training, replay
  qa_stream.py    the 3M verified Q&A database: generate, read, cross-check with MNX1
  expert_stream.py verified code corpus (recipes, generated tasks, stdlib)
  __main__.py     CLI (python -m mnx.agent)
  base.py         Answer / Skill types, number formatting
  expr.py         from-scratch CAS core: tokenizer, parser, exact evaluator, polynomials, root finding
  router.py       naive Bayes intent router + cross-validation
  retrieval.py    BM25 index, sentence extraction, memory store
  evaluate.py     benchmark runner (python -m mnx.agent.evaluate)
  skills/         code, math, units, dates, logic, sequences, synth, knowledge, chat, neural,
                  local_model (opt-in bridge to a code model on localhost), agents (the registry skill)
  registry.py     10,309 small agents: build, verify, search, run
data/code/        165 verified code recipes (description, code, tests)
data/registry/    the agent registry manifest (agents.jsonl.gz) and what it is made of
data/knowledge/   bundled knowledge base (markdown, one fact per sentence)
data/router/      router training examples
data/eval/        benchmark.jsonl (tuning); holdout.jsonl, code_holdout.jsonl, code_holdout2.jsonl (held out)
scripts/          demo_session.py (sample session), humaneval.py (public benchmark), eval_qa.py and eval_neural.py,
                  compare_models.py (score a local Ollama model on the same test sets)
docs/sample-session.md  the output of that script
mnx/*.py          MNX 2.0 Max Coder transformer: tokenizer, model, data (incl. streaming), train, generate
mnx/export_mobile.py  int8 ExecuTorch export for phones
mnx/trainer_bot.py    24/7 active-learning CPU trainer (quiz, grade, replay, curriculum)
docs/mobile.md    running MNX1 and the model on a phone
docs/trainer-bot.md  the 24/7 CPU Q&A trainer bot
docs/registry.md  the agent registry and how MNX1 picks from it
docs/local-model.md  the opt-in local code model proposer (Termux or Ollama)
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

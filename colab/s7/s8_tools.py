#!/usr/bin/env python3
"""Session 8 helpers: reasoning / logic / research data from the Ai-room generators (answers computed by code).

    s8_tools.py reason --code CODE --data DATA --out OUT --airoom AIROOM [--n 500000] [--eval-n 400]

Writes eval_reason.jsonl (never trained on: no question shared with the session-7 Ai-room training
set, its eval set, or the new training set) and reason.train.bin / reason.val.bin. Items are scored by
s7_tools.py score (kind = rs_<subject>, so the score splits by subject).
"""
import argparse, json, os, random, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s7_tools as T

SUBJECTS = ["logic", "discrete_math", "probability", "research", "statistics", "number_theory", "optimization", "data_science", "cs", "algorithms"]
EVAL_SEED, TRAIN_SEED = 737373, 838383


def gen(path, seed, n, avoid=frozenset(), subjects=SUBJECTS):
    sys.path.insert(0, path)
    from ai_room.teacher import generators as G
    from ai_room.teacher import teacher as _reg  # noqa: F401
    pools = [(x, g, f) for x in subjects for g in range(1, 13) for f in G.families_for(x, g)]
    rng, seen, made, tries = random.Random(seed), set(), 0, 0
    while made < n and tries < n * 30:
        tries += 1
        x, g, f = rng.choice(pools)
        try: q = G.generate(f, x, g, rng)
        except Exception: continue
        if q["type"] == "code": continue
        text = q["question"] + ("\n" + "\n".join(q["options"]) if q.get("options") else "")
        h = T.h8(text)
        if h in seen or h in avoid: continue
        seen.add(h); made += 1
        yield {"q": text, "a": q["answer"], "why": (q.get("explanation") or "").strip(), "kind": "rs_" + x, "type": q["type"],
               "tol": q.get("tol"), "exact": q.get("exact"), "src": "airoom", "fam": f.id}


def cmd_reason(a):
    tok = T.find_tokenizer(a.code); os.makedirs(a.out, exist_ok=True)
    bos, nl = T.doc_format(tok, a.data)
    # everything session 7 trained on or was scored on (Ai-room seeds 616161 / 515151), plus the usual-style training questions
    s7 = {T.h8(i["q"]) for i in T.airoom_gen(a.airoom, T.AIROOM_TRAIN_SEED, 400000)}
    s7e = {T.h8(i["q"]) for i in T.airoom_gen(a.airoom, T.AIROOM_EVAL_SEED, 300)}
    avoid = s7 | s7e
    ev = list(gen(a.airoom, EVAL_SEED, a.eval_n, avoid))
    evh = {T.h8(i["q"]) for i in ev}
    with open(os.path.join(a.out, "eval_reason.jsonl"), "w") as f:
        for it in ev: f.write(json.dumps(it) + "\n")
    print("reason eval items:", len(ev), dict(sorted(T._count(ev).items())), flush=True)
    ids, n = [], 0
    for it in gen(a.airoom, TRAIN_SEED, a.n, avoid | evh):
        ids.extend(([tok.bos_id] if bos else []) + tok.encode(T.airoom_doc(it), allow_special=False) + [tok.eos_id]); n += 1
    arr = np.array(ids, dtype=np.uint16); cut = int(len(arr) * 0.98); cut += int(np.where(arr[cut:] == tok.eos_id)[0][0]) + 1
    arr[:cut].tofile(os.path.join(a.out, "reason.train.bin")); arr[cut:].tofile(os.path.join(a.out, "reason.val.bin"))
    print(f"reason docs {n}, tokens {len(arr)}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("reason"); r.add_argument("--code", required=True); r.add_argument("--data", required=True); r.add_argument("--out", required=True)
    r.add_argument("--airoom", required=True); r.add_argument("--n", type=int, default=500000); r.add_argument("--eval-n", type=int, default=400)
    a = ap.parse_args(); cmd_reason(a)

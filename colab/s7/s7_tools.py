#!/usr/bin/env python3
"""Session 7 helpers for the 306M MNX model (Colab T4).

    s7_tools.py fresh --code /content/code --data /content/data --out /content/data7 --n 600000
    s7_tools.py score --code /content/code --ckpt ckpt.pt --items eval_unseen.jsonl --out score.json

fresh: finds every question already in the training bins, writes an unseen eval set (seed 424242) and a
       fresh training set (seed 7007) that shares no question with either, tokenized like the existing data.
score: greedy answers on the unseen eval set; exact match on the first line, code must pass its asserts.
       Same grading as scripts/eval_qa.py in the Mnx repo. Never trains on these questions.
"""
import argparse, glob, hashlib, importlib.util, json, os, re, subprocess, sys, tempfile, time
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
EVAL_SEED, FRESH_SEED = 424242, 7007


def repo(code):
    r = os.environ.get("MNX_REPO") or os.path.join(code, "repo")
    if r not in sys.path: sys.path.insert(0, r)
    return r


def find_tokenizer(code):
    repo(code)
    from mnx.tokenizer import BPETokenizer
    for p in sorted(glob.glob(os.path.join(code, "**", "*.json"), recursive=True)):
        try:
            d = json.load(open(p))
        except Exception:
            continue
        if isinstance(d, dict) and d.get("model") == "mnx-bpe":
            t = BPETokenizer.from_dict(d)
            if t.vocab_size == int(os.environ.get("S7_VOCAB", 16384)):
                print("tokenizer", p, "vocab", t.vocab_size, flush=True); return t
    raise SystemExit("no 16384-vocab mnx-bpe tokenizer.json found under " + code)


def h8(s):
    return int.from_bytes(hashlib.md5(s.encode("utf-8", "replace")).digest()[:8], "little", signed=True)


def table(tok):
    t = [tok.vocab.get(i, b"") for i in range(tok.vocab_size)]
    for i, s in tok.id_to_special.items(): t[i] = s.encode()
    return t


QRE = re.compile(rb"(?:^|<\|eos\|>)(?:<\|bos\|>)?Q: (.*?)\nA:", re.S)


def train_question_hashes(tok, data, cache):
    if os.path.exists(cache):
        return set(np.load(cache).tolist())
    tb, hs = table(tok), set()
    for p in sorted(glob.glob(os.path.join(data, "*.train.bin"))):
        if os.path.basename(p).startswith("fresh_qa"): continue
        a = np.memmap(p, dtype=np.uint16, mode="r"); n0 = len(hs); t0 = time.time()
        step, i = 20_000_000, 0
        while i < len(a):
            j = min(len(a), i + step)
            if j < len(a):  # cut at an eos so a question is never split
                e = np.where(a[j:j + 200000] == tok.eos_id)[0]
                j = j + int(e[0]) + 1 if len(e) else j
            buf = b"".join(map(tb.__getitem__, a[i:j].tolist()))
            for m in QRE.finditer(buf): hs.add(h8(m.group(1).decode("utf-8", "replace")))
            i = j
        print(f"  {os.path.basename(p)}: {len(hs) - n0} new questions ({time.time() - t0:.0f}s)", flush=True)
    np.save(cache, np.array(sorted(hs), dtype=np.int64)); return hs


def doc_format(tok, data):
    """Read two real docs from qa_stream.train.bin and check our encoding reproduces them token for token."""
    a = np.array(np.memmap(os.path.join(data, "qa_stream.train.bin"), dtype=np.uint16, mode="r")[:50000])
    e = np.where(a == tok.eos_id)[0]
    for k in range(1, 6):
        d = a[e[k] + 1:e[k + 1]].tolist()
        bos = bool(d) and d[0] == tok.bos_id
        text = tok.decode(d[1:] if bos else d)
        if text.startswith("Q: ") and "\nA:" in text:
            ok = (([tok.bos_id] if bos else []) + tok.encode(text, allow_special=False)) == d
            print("format: bos=%s trailing_newline=%s re-encodes exactly=%s\n  e.g. %r" % (bos, text.endswith("\n"), ok, text[:160]), flush=True)
            if not ok: raise SystemExit("could not reproduce the training tokenization; stopping so nothing bad gets trained")
            return bos, text.endswith("\n")
    raise SystemExit("no Q:/A: doc found at the start of qa_stream.train.bin")


# ---- Ai-room (github.com/Tharunmaks/ai-room): exact problem families, answers computed by code ----
AIROOM_EVAL_SEED, AIROOM_TRAIN_SEED = 515151, 616161


def airoom_gen(path, seed, n, avoid=frozenset()):
    sys.path.insert(0, path)
    import random
    from ai_room.teacher import generators as G
    from ai_room.teacher import teacher as _reg  # noqa: F401  registers every family
    subs = sorted({x for f in G.FAMILIES.values() for x in f.subjects})
    pools = [(x, g, f) for x in subs for g in range(1, 13) for f in G.families_for(x, g)]
    rng, seen, made, tries = random.Random(seed), set(), 0, 0
    while made < n and tries < n * 30:
        tries += 1
        x, g, f = rng.choice(pools)
        try: q = G.generate(f, x, g, rng)
        except Exception: continue
        if q["type"] == "code": continue  # code stays with qa_gen (graded by running tests)
        text = q["question"] + ("\n" + "\n".join(q["options"]) if q.get("options") else "")
        h = h8(text)
        if h in seen or h in avoid: continue
        seen.add(h); made += 1
        yield {"q": text, "a": q["answer"], "why": (q.get("explanation") or "").strip(), "kind": "ar_" + q["type"],
               "type": q["type"], "tol": q.get("tol"), "exact": q.get("exact"), "src": "airoom", "fam": f.id}


def airoom_doc(it):
    why = (" " + it["why"].replace("\n", " ")) if it.get("why") else ""
    return f"Q: {it['q']}\nA: {it['a']}\n" + (f"Why:{why}\n" if why else "")


def airoom_ok(it, ans, path):
    sys.path.insert(0, path)
    from ai_room.grading import grade_question
    q = {"type": it["type"], "answer": it["a"], "tol": it.get("tol"), "exact": it.get("exact")}
    return grade_question(q, ans)[0] >= 1.0


def cmd_fresh(a):
    import qa_gen
    tok = find_tokenizer(a.code); os.makedirs(a.out, exist_ok=True)
    bos, nl = doc_format(tok, a.data)
    json.dump({"bos": bos, "nl": nl}, open(os.path.join(a.out, "format.json"), "w"))
    seen = train_question_hashes(tok, a.data, os.path.join(a.out, "train_q_hashes.npy"))
    print("questions already in training data:", len(seen), flush=True)
    ev = []
    for it in qa_gen.stream(a.eval_n * 4, EVAL_SEED):
        if it["kind"] == "code" and "\nassert " not in it["q"]: continue  # a code item needs tests to be gradable
        if h8(it["q"]) not in seen: ev.append(it)
        if len(ev) == a.eval_n: break
    evh = {h8(it["q"]) for it in ev}
    with open(os.path.join(a.out, "eval_unseen.jsonl"), "w") as f:
        for it in ev: f.write(json.dumps(it) + "\n")
    print("unseen eval items:", len(ev), dict(sorted(_count(ev).items())), flush=True)
    ids, n, skipped, t0 = [], 0, 0, time.time()
    for it in qa_gen.stream(a.n, FRESH_SEED):
        h = h8(it["q"])
        if h in seen or h in evh: skipped += 1; continue
        text = f"Q: {it['q']}\nA: {it['a']}" + ("\n" if nl else "")  # same as qa_stream.as_document when nl
        ids.extend(([tok.bos_id] if bos else []) + tok.encode(text, allow_special=False) + [tok.eos_id]); n += 1
        if n % 100000 == 0: print(f"  {n} fresh docs, {len(ids)} tokens ({time.time() - t0:.0f}s)", flush=True)
    if a.airoom:
        aev = list(airoom_gen(a.airoom, AIROOM_EVAL_SEED, a.eval_n))
        aevh = {h8(it["q"]) for it in aev}
        with open(os.path.join(a.out, "eval_airoom.jsonl"), "w") as f:
            for it in aev: f.write(json.dumps(it) + "\n")
        print("ai-room unseen eval items:", len(aev), dict(sorted(_count(aev).items())), flush=True)
        aids, an = [], 0
        for it in airoom_gen(a.airoom, AIROOM_TRAIN_SEED, a.airoom_n, avoid=aevh):
            aids.extend(([tok.bos_id] if bos else []) + tok.encode(airoom_doc(it), allow_special=False) + [tok.eos_id]); an += 1
        aarr = np.array(aids, dtype=np.uint16); acut = int(len(aarr) * 0.98)
        acut += int(np.where(aarr[acut:] == tok.eos_id)[0][0]) + 1
        aarr[:acut].tofile(os.path.join(a.out, "airoom.train.bin")); aarr[acut:].tofile(os.path.join(a.out, "airoom.val.bin"))
        print(f"ai-room docs {an}, tokens {len(aarr)}", flush=True)
    arr = np.array(ids, dtype=np.uint16); cut = int(len(arr) * 0.98)
    cut += int(np.where(arr[cut:] == tok.eos_id)[0][0]) + 1
    arr[:cut].tofile(os.path.join(a.out, "fresh_qa.train.bin")); arr[cut:].tofile(os.path.join(a.out, "fresh_qa.val.bin"))
    print(f"fresh docs {n} (skipped {skipped} already-seen), tokens {len(arr)}", flush=True)


def _count(items):
    c = defaultdict(int)
    for it in items: c[it["kind"]] += 1
    return c


def run_tests(code, tests, timeout=5):
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code + "\n\n" + "\n".join(tests) + "\n"); p = f.name
    try:
        return subprocess.run([sys.executable, p], capture_output=True, timeout=timeout).returncode == 0
    except subprocess.TimeoutExpired:
        return False
    finally:
        os.unlink(p)


def cmd_score(a):
    import torch
    tok = find_tokenizer(a.code)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False, mmap=True)
    C = ck["config"]; adp = os.path.join(a.code, "trainer", "adapters", "mnx1_custom_s6.py")
    sp = importlib.util.spec_from_file_location("adapter", adp); ad = importlib.util.module_from_spec(sp); sp.loader.exec_module(ad)
    w = ad.build_model({**C["model_cfg"], "dropout": 0.0, "grad_ckpt": False}); w.load_state_dict(ck["model"])
    st = ck.get("state", {}); del ck
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = w.m.to(dev).eval()
    if dev == "cuda": m.half()
    fmt = json.load(open(a.fmt)) if a.fmt else {}
    items = [json.loads(l) for l in open(a.items) if l.strip()][: a.n or None]
    stats, rows, t0 = defaultdict(lambda: [0, 0]), [], time.time()
    with torch.no_grad():
        for k, it in enumerate(items):
            ids = ([tok.bos_id] if fmt.get("bos") else []) + tok.encode(f"Q: {it['q']}\nA:", allow_special=False)
            x = torch.tensor([ids], device=dev)
            out = m.generate(x, 200 if it["kind"] == "code" else (320 if it.get("src") == "airoom" else 40), temperature=0.0, eos_id=tok.eos_id)[0, len(ids):].tolist()
            if tok.eos_id in out: out = out[: out.index(tok.eos_id)]
            ans = tok.decode(out, skip_special=True)
            if it.get("src") == "airoom":
                body = ans.strip()
                if "\nQ:" in body: body = body[: body.index("\nQ:")]
                if body.startswith("[run]"):   # session 9: the model answered with a program; run it (sandboxed subprocess)
                    import s9_pal
                    got = s9_pal.answer_text(body); ans = ans + "\n=> " + got
                else:
                    got = body.split("\n")[0].strip()
                ok = airoom_ok(it, got, a.airoom)
            elif it["kind"] == "code":
                tests = [l for l in it["q"].splitlines() if l.startswith("assert ")]
                body = ans.strip()
                if "\nQ:" in body: body = body[: body.index("\nQ:")]
                ok = bool(tests) and run_tests(body, tests)
            else:
                ok = ans.strip().split("\n")[0].strip() == it["a"].strip()
            stats[it["kind"]][0] += 1; stats[it["kind"]][1] += ok
            rows.append({"kind": it["kind"], "q": it["q"][:200], "want": it["a"][:200], "got": ans[-600:], "ok": bool(ok)})
            if (k + 1) % 50 == 0: print(f"  {k + 1}/{len(items)} scored, {sum(r['ok'] for r in rows)} right ({time.time() - t0:.0f}s)", flush=True)
    tot = sum(v[1] for v in stats.values())
    for kind, (n, c) in sorted(stats.items()): print(f"  {kind:<11} {c:>4}/{n:<4}")
    print(f"  TOTAL       {tot:>4}/{len(items)}  ({100 * tot / max(1, len(items)):.1f}% on questions it never trained on)", flush=True)
    json.dump({"ckpt": a.ckpt, "step": st.get("step"), "heldout_loss": st.get("best"), "total": tot, "n": len(items),
               "by_kind": {k: v for k, v in stats.items()}, "items": rows}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fresh"); f.add_argument("--code", required=True); f.add_argument("--data", required=True)
    f.add_argument("--out", required=True); f.add_argument("--n", type=int, default=600000); f.add_argument("--eval-n", type=int, default=300)
    f.add_argument("--airoom", default=""); f.add_argument("--airoom-n", type=int, default=400000)
    s = sub.add_parser("score"); s.add_argument("--code", required=True); s.add_argument("--ckpt", required=True)
    s.add_argument("--items", required=True); s.add_argument("--out", required=True); s.add_argument("--n", type=int, default=0); s.add_argument("--fmt", default=""); s.add_argument("--airoom", default="")
    a = ap.parse_args(); {"fresh": cmd_fresh, "score": cmd_score}[a.cmd](a)

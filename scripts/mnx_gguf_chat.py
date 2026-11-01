#!/usr/bin/env python3
"""Chat with an MNX GGUF model through a local llama.cpp server. Pure standard library (runs in Termux).

1. Start the server:   llama-server -m mnx306m-q8_0.gguf --port 8099 -c 2048
2. Run this script:    python mnx_gguf_chat.py --tokenizer tokenizer.json

Why a script and not `llama-cli`: MNX splits every digit into its own token, which llama.cpp's built-in
tokenizer cannot do, so the prompt is tokenized here (mnx/tokenizer.py) and sent to the server as token ids.
Prompts use the format the model was trained on:  "Q: <question>\\nA:"  -> the answer is the first line.
"""
import argparse, json, os, sys, urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
try:
    from mnx.tokenizer import BPETokenizer
except ImportError:  # Termux: tokenizer.py copied next to this script
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from tokenizer import BPETokenizer


def complete(url, ids, n_predict):
    req = urllib.request.Request(url + "/completion", json.dumps(
        {"prompt": ids, "n_predict": n_predict, "temperature": 0.0, "top_k": 1, "cache_prompt": True,
         "return_tokens": True}).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=300))["tokens"]


def answer(tok, url, question, n_predict=200):
    ids = tok.encode("Q: %s\nA:" % question, allow_special=False)
    out = complete(url, ids, n_predict)
    if tok.eos_id in out: out = out[: out.index(tok.eos_id)]
    text = tok.decode(out, skip_special=True)
    return text.split("\nQ:")[0].strip("\n")   # the model may start a new "Q:" after its answer; cut there


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="tokenizer.json"); ap.add_argument("--url", default="http://127.0.0.1:8099")
    ap.add_argument("--max-tokens", type=int, default=200)
    a = ap.parse_args()
    tok = BPETokenizer.load(a.tokenizer)
    print("MNX 306M. Type a question (empty line to quit). Strong on code tasks, logic puzzles, primes; weak on long arithmetic and dates.")
    while True:
        try: q = input("you> ").strip()
        except (EOFError, KeyboardInterrupt): break
        if not q: break
        q = q.replace("\\n", "\n")   # type \n for a line break (code tasks have assert lines)
        print("mnx>", answer(tok, a.url, q, a.max_tokens).strip() or "(no answer)")

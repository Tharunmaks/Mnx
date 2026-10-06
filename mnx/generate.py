"""Inference CLI for MNX 2.0 Max Coder.

Completion::

    python -m mnx.generate --ckpt runs/tiny --prompt "def fib(n):"

Fill-in-the-middle (the model fills the gap between prefix and suffix)::

    python -m mnx.generate --ckpt runs/tiny --fim \\
        --prefix "def add(a, b):\\n    " --suffix "\\n    return result\\n"
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional, Sequence

import torch

from mnx.data import make_fim_prompt, split_fim_output
from mnx.model import MNXCoder
from mnx.tokenizer import BPETokenizer
from mnx.train import load_checkpoint


def complete(
    model: MNXCoder,
    tok: BPETokenizer,
    prompt: str,
    max_new_tokens: int = 64,
    temperature: float = 0.8,
    top_k: Optional[int] = 40,
    top_p: Optional[float] = None,
    seed: Optional[int] = None,
    add_bos: bool = True,
) -> str:
    """Return the text generated after ``prompt`` (prompt not included)."""
    if seed is not None:
        torch.manual_seed(seed)
    ids = tok.encode(prompt, add_bos=add_bos, allow_special=False)
    x = torch.tensor([ids], dtype=torch.long)
    out = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k, top_p=top_p, eos_id=tok.eos_id)
    new_ids: List[int] = out[0, len(ids) :].tolist()
    if tok.eos_id in new_ids:
        new_ids = new_ids[: new_ids.index(tok.eos_id)]
    return tok.decode(new_ids, skip_special=True)


def fill_in_middle(
    model: MNXCoder,
    tok: BPETokenizer,
    prefix: str,
    suffix: str,
    max_new_tokens: int = 64,
    temperature: float = 0.8,
    top_k: Optional[int] = 40,
    top_p: Optional[float] = None,
    seed: Optional[int] = None,
) -> str:
    """Return the predicted middle between ``prefix`` and ``suffix``."""
    if seed is not None:
        torch.manual_seed(seed)
    ids = make_fim_prompt(prefix, suffix, tok)
    x = torch.tensor([ids], dtype=torch.long)
    out = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k, top_p=top_p, eos_id=tok.eos_id)
    middle = split_fim_output(out[0].tolist(), tok)
    return tok.decode(middle, skip_special=True)


def _unescape(s: str) -> str:
    """Interpret ``\\n`` and ``\\t`` escapes passed on the command line."""
    return s.encode("utf-8").decode("unicode_escape")


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Generate code with an MNX 2.0 Max Coder checkpoint.")
    p.add_argument("--ckpt", required=True, help="run directory (containing ckpt.pt + tokenizer.json) or ckpt.pt path")
    p.add_argument("--prompt", default=None, help="prompt text for completion mode (use '-' to read stdin)")
    p.add_argument("--fim", action="store_true", help="fill-in-the-middle mode; requires --prefix and --suffix")
    p.add_argument("--prefix", default="", help="FIM prefix text")
    p.add_argument("--suffix", default="", help="FIM suffix text")
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.8, help="0 = greedy decoding")
    p.add_argument("--top-k", type=int, default=40, help="0 disables top-k")
    p.add_argument("--top-p", type=float, default=None, help="nucleus sampling threshold, e.g. 0.95")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--raw", action="store_true", help="print only the generated text (no prompt echo)")
    args = p.parse_args(argv)

    model, tok, meta = load_checkpoint(args.ckpt, device=args.device)
    top_k = args.top_k if args.top_k and args.top_k > 0 else None

    if args.fim:
        prefix, suffix = _unescape(args.prefix), _unescape(args.suffix)
        middle = fill_in_middle(
            model, tok, prefix, suffix, args.max_new_tokens, args.temperature, top_k, args.top_p, args.seed
        )
        if args.raw:
            print(middle, end="")
        else:
            print(f"--- MNX 2.0 Max Coder [{model.cfg.name}, step {meta.get('step')}] FIM ---")
            print(prefix + middle + suffix)
        return

    if args.prompt is None:
        p.error("--prompt is required unless --fim is given")
    prompt = sys.stdin.read() if args.prompt == "-" else _unescape(args.prompt)
    text = complete(model, tok, prompt, args.max_new_tokens, args.temperature, top_k, args.top_p, args.seed)
    if args.raw:
        print(text, end="")
    else:
        print(f"--- MNX 2.0 Max Coder [{model.cfg.name}, step {meta.get('step')}] ---")
        print(prompt + text)


if __name__ == "__main__":
    main()

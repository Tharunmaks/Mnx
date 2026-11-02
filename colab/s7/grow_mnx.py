#!/usr/bin/env python3
"""Grow an MNX checkpoint deeper without changing what it computes.

Every transformer block is pre-norm and residual (x + attn(norm x); x + ffn(norm x)). A new block whose attention
output projection (wo) and FFN down projection are zero adds exactly 0 to the residual stream, so the grown
model gives bit-for-bit the same logits as the old one. The new block starts as a copy of its neighbour
(everything but those two matrices), so training can switch it on smoothly. Old weights are untouched.

usage: grow_mnx.py IN.pt OUT.pt --layers 36 [--verify CODE_DIR]
"""
import argparse, copy, re, sys
import torch


def grow(ck, new_layers):
    cfg = ck["config"]["model_cfg"]; L = cfg["n_layers"]; assert new_layers > L
    sd = ck["model"]; pre = "m." if any(k.startswith("m.blocks.") for k in sd) else ""
    extra = new_layers - L
    # insert a copy after layers spaced evenly through the stack
    after = sorted({min(L - 1, int((i + 1) * L / extra) - 1) for i in range(extra)})
    while len(after) < extra:  # spacing collisions: fill from the top
        after = sorted(set(after) | {max(set(range(L)) - set(after))})
    order = []  # (source layer, is_new)
    for i in range(L):
        order.append((i, False))
        if i in after: order.append((i, True))
    assert len(order) == new_layers, (len(order), new_layers)
    out = {k: v for k, v in sd.items() if not k.startswith(pre + "blocks.")}
    for j, (i, new) in enumerate(order):
        for k, v in sd.items():
            m = re.match(re.escape(pre) + r"blocks\.%d\.(.*)" % i, k)
            if not m: continue
            t = v.clone()
            if new and (m.group(1) == "attn.wo.weight" or m.group(1) == "ffn.down.weight"): t.zero_()
            out["%sblocks.%d.%s" % (pre, j, m.group(1))] = t
    ck2 = {"model": out, "state": {"grown_from_layers": L, "new_layers": new_layers, "inserted_after": after, "from_state": ck.get("state")},
           "config": copy.deepcopy(ck["config"])}
    ck2["config"]["model_cfg"]["n_layers"] = new_layers
    return ck2


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("inp"); ap.add_argument("out"); ap.add_argument("--layers", type=int, required=True)
    ap.add_argument("--verify", default="", help="code dir containing trainer/adapters (checks logits are identical)")
    a = ap.parse_args()
    ck = torch.load(a.inp, map_location="cpu", weights_only=False)
    ck2 = grow(ck, a.layers); torch.save(ck2, a.out)
    n = sum(v.numel() for k, v in ck2["model"].items()); print("wrote", a.out, "layers", a.layers, "params %.1fM" % (n / 1e6))
    if a.verify:
        import importlib.util, os
        sys.path.insert(0, a.verify + "/repo")
        sp = importlib.util.spec_from_file_location("ad", a.verify + "/trainer/adapters/mnx1_custom_s6.py"); ad = importlib.util.module_from_spec(sp); sp.loader.exec_module(ad)
        def load(c):
            m = ad.build_model({**c["config"]["model_cfg"], "dropout": 0.0, "grad_ckpt": False}); m.load_state_dict(c["model"]); return m.m.eval()
        old, new = load(ck), load(ck2)
        torch.manual_seed(0); x = torch.randint(0, 16000, (2, 96))
        with torch.no_grad():
            a1, a2 = old(x), new(x)
            a1 = a1[0] if isinstance(a1, tuple) else a1; a2 = a2[0] if isinstance(a2, tuple) else a2
        print("max |logit diff| old vs grown:", float((a1 - a2).abs().max()), "argmax equal:", bool((a1.argmax(-1) == a2.argmax(-1)).all()))

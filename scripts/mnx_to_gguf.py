#!/usr/bin/env python3
"""Convert an MNX checkpoint (the 306M Colab/Kaggle models) to a llama.cpp GGUF file.

MNX is a Llama-style decoder (RMSNorm, GQA, SwiGLU, RoPE, tied embeddings), so it maps onto the
`llama` architecture. Two details matter:
  * MNX rotates half-split RoPE pairs (x_j, x_{j+d/2}); llama.cpp's `llama` arch expects adjacent
    pairs, so wq / wk are permuted exactly like Hugging Face's llama converter does.
  * llama.cpp cannot express MNX's pre-tokenizer regex (it splits single digits), so the GGUF carries a
    placeholder byte-level vocab. Always send TOKEN IDS produced by mnx.tokenizer (see mnx_gguf_chat.py).

usage: mnx_to_gguf.py CKPT.pt TOKENIZER.json OUT.gguf --llama-cpp /path/to/llama.cpp [--dtype f16|f32]
"""
import argparse, json, sys

import numpy as np
import torch


def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


def permute(w, n_head):  # HF split-half -> llama.cpp adjacent-pair rotary layout
    return w.reshape(n_head, 2, w.shape[0] // n_head // 2, *w.shape[1:]).swapaxes(1, 2).reshape(w.shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt"); ap.add_argument("tokenizer"); ap.add_argument("out")
    ap.add_argument("--llama-cpp", required=True); ap.add_argument("--dtype", default="f16", choices=["f16", "f32"])
    a = ap.parse_args()
    sys.path.insert(0, a.llama_cpp + "/gguf-py")
    import gguf

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False, mmap=True)
    cfg = ck["config"]["model_cfg"]; sd = {k[2:] if k.startswith("m.") else k: v for k, v in ck["model"].items()}
    d, L, H, KV, FF, V = cfg["d_model"], cfg["n_layers"], cfg["n_heads"], cfg["n_kv_heads"], cfg["d_ff"], cfg["vocab_size"]
    tied = torch.equal(sd["lm_head.weight"], sd["tok_emb.weight"]) if "lm_head.weight" in sd else True
    print("config", cfg, "tied", tied)
    tok = json.load(open(a.tokenizer)); specials = tok["special_tokens"]; merges = [tuple(m) for m in tok["merges"]]
    assert 256 + len(specials) + len(merges) == V, "vocab mismatch"

    # placeholder byte-level vocab (ids must line up; strings made unique)
    b2u = bytes_to_unicode(); toks = [b2u[i] for i in range(256)] + list(specials); seen = set(toks)
    types = [gguf.TokenType.NORMAL] * 256 + [gguf.TokenType.CONTROL] * len(specials)
    merge_strs = []
    for i, (x, y) in enumerate(merges):
        s = toks[x] + toks[y]
        merge_strs.append(toks[x] + " " + toks[y])
        if s in seen: s = s + chr(0xE000 + (i % 6000)) + chr(0xE000 + (i // 6000))
        seen.add(s); toks.append(s); types.append(gguf.TokenType.NORMAL)
    assert len(toks) == V and len(set(toks)) == V

    w = gguf.GGUFWriter(a.out, "llama")
    w.add_name("mnx-306m"); w.add_context_length(cfg["max_seq_len"]); w.add_embedding_length(d); w.add_block_count(L)
    w.add_feed_forward_length(FF); w.add_head_count(H); w.add_head_count_kv(KV); w.add_layer_norm_rms_eps(1e-5)
    w.add_rope_dimension_count(d // H); w.add_rope_freq_base(10000.0); w.add_vocab_size(V)
    w.add_file_type(gguf.LlamaFileType.MOSTLY_F16 if a.dtype == "f16" else gguf.LlamaFileType.ALL_F32)
    w.add_tokenizer_model("gpt2"); w.add_tokenizer_pre("default"); w.add_token_list(toks); w.add_token_types(types)
    w.add_token_merges(merge_strs)
    bos, eos, pad = 256 + specials.index("<|bos|>"), 256 + specials.index("<|eos|>"), 256 + specials.index("<|pad|>")
    w.add_bos_token_id(bos); w.add_eos_token_id(eos); w.add_pad_token_id(pad); w.add_add_bos_token(False)

    dt = np.float16 if a.dtype == "f16" else np.float32
    def put(name, t, as_f32=False):
        arr = t.detach().float().numpy()
        w.add_tensor(name, arr.astype(np.float32 if (as_f32 or arr.ndim == 1) else dt))
    put("token_embd.weight", sd["tok_emb.weight"])
    if not tied: put("output.weight", sd["lm_head.weight"])
    put("output_norm.weight", sd["norm.weight"])
    for i in range(L):
        p = f"blocks.{i}."
        put(f"blk.{i}.attn_norm.weight", sd[p + "attn_norm.weight"])
        put(f"blk.{i}.attn_q.weight", torch.from_numpy(permute(sd[p + "attn.wq.weight"].float().numpy(), H)))
        put(f"blk.{i}.attn_k.weight", torch.from_numpy(permute(sd[p + "attn.wk.weight"].float().numpy(), KV)))
        put(f"blk.{i}.attn_v.weight", sd[p + "attn.wv.weight"])
        put(f"blk.{i}.attn_output.weight", sd[p + "attn.wo.weight"])
        put(f"blk.{i}.ffn_norm.weight", sd[p + "ffn_norm.weight"])
        put(f"blk.{i}.ffn_gate.weight", sd[p + "ffn.gate.weight"])
        put(f"blk.{i}.ffn_up.weight", sd[p + "ffn.up.weight"])
        put(f"blk.{i}.ffn_down.weight", sd[p + "ffn.down.weight"])
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    print("wrote", a.out)


if __name__ == "__main__":
    main()

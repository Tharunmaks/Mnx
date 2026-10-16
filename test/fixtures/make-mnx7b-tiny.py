"""Builds test/fixtures/mnx7b-tiny: a tiny random-weight model with exactly
the architecture of Mnx 7B's train.py, plus reference outputs computed with
that PyTorch code, so mnx7b/run.py (numpy) can be checked against it.
Needs torch and tokenizers; the outputs are committed, so tests don't.
    python3 test/fixtures/make-mnx7b-tiny.py
"""
import json, os, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mnx7b-tiny")
os.makedirs(OUT, exist_ok=True)
torch.manual_seed(0)
C = dict(vocab=320, dim=64, layers=3, heads=4, ctx=96, ex_layer=1, ex_dim=16, ex_hidden=4, orders=[1, 2, 3], target_total=0)
T = C["ctx"]

# ---- copied from train.py (model part), unchanged except sizes ----
class Blk(nn.Module):
    def __init__(s, d, h):
        super().__init__(); s.h = h; s.n1 = nn.RMSNorm(d); s.qkv = nn.Linear(d, 3 * d, bias=False); s.o = nn.Linear(d, d, bias=False)
        s.n2 = nn.RMSNorm(d); s.w1 = nn.Linear(d, 4 * d, bias=False); s.w3 = nn.Linear(d, 4 * d, bias=False); s.w2 = nn.Linear(4 * d, d, bias=False)
    def forward(s, x, cos, sin):
        b, t, d = x.shape; q, k, v = s.qkv(s.n1(x)).view(b, t, 3, s.h, d // s.h).permute(2, 0, 3, 1, 4)
        def rot(z): z1, z2 = z.chunk(2, -1); return torch.cat([z1 * cos - z2 * sin, z1 * sin + z2 * cos], -1)
        y = F.scaled_dot_product_attention(rot(q), rot(k), v, is_causal=True).transpose(1, 2).reshape(b, t, d)
        x = x + s.o(y); h = s.n2(x); return x + s.w2(F.silu(s.w1(h)) * s.w3(h))
class Core(nn.Module):
    def __init__(s):
        super().__init__(); d = C['dim']; s.emb = nn.Embedding(C['vocab'], d); s.blocks = nn.ModuleList([Blk(d, C['heads']) for _ in range(C['layers'])])
        s.nf = nn.RMSNorm(d); s.ein = nn.Linear(d, C['ex_dim'], bias=False); s.eout = nn.Linear(C['ex_dim'], d, bias=False)
        hd = d // C['heads']; f = 1.0 / (10000 ** (torch.arange(0, hd, 2).float() / hd)); a = torch.outer(torch.arange(T).float(), f)
        s.register_buffer('cos', a.cos()[None, None], persistent=False); s.register_buffer('sin', a.sin()[None, None], persistent=False)
    def forward(s, idx, expert_fn):
        x = s.emb(idx); t = idx.shape[1]; c, sn = s.cos[:, :, :t], s.sin[:, :, :t]
        for i, b in enumerate(s.blocks):
            if i == C['ex_layer']: x = x + s.eout(expert_fn(s.ein(x)))
            x = b(x, c, sn)
        return F.linear(s.nf(x), s.emb.weight)
core = Core()
for p in core.parameters(): nn.init.normal_(p, std=0.3)  # big weights so mistakes show up
for b in core.blocks: b.n1.weight.data.uniform_(0.5, 1.5); b.n2.weight.data.uniform_(0.5, 1.5)
E = 997; PE = 2 * C['ex_dim'] * C['ex_hidden']
table = (torch.randn(E, PE) * 0.5).to(torch.bfloat16)
PR = [1000003, 998244353, 7919]
def ex_index(idx):
    ix = idx.long(); hs = []
    for o in C['orders']:
        h = torch.zeros_like(ix)
        for j in range(o): h = h * PR[j % 3] + F.pad(ix, (j, 0))[:, :ix.shape[1]] + 1
        h = (h * 2654435761 + o * 97) % E; hs.append(h)
    return torch.stack(hs, -1)
cur = {}
def expert_fn(z):
    ids = cur['ids']; w = table[ids.reshape(-1)].float()
    w = w.view(*ids.shape, 2, C['ex_dim'], C['ex_hidden']); w1, w2 = w[..., 0, :, :], w[..., 1, :, :]
    h = F.gelu(torch.einsum('btd,btkdh->btkh', z, w1)); return torch.einsum('btkh,btkdh->btd', h, w2) / len(C['orders'])
# --------------------------------------------------------------------

torch.save(dict(core=core.state_dict(), config=C, E=E, step=42), os.path.join(OUT, "core_only.pt"))
with open(os.path.join(OUT, "table.bin"), "wb") as f: f.write(table.view(torch.int16).numpy().tobytes())

# Tokenizer: byte-level BPE with the same special tokens as tok16k.json.
tk = Tokenizer(models.BPE()); tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False); tk.decoder = decoders.ByteLevel()
corpus = ["<user> Hello there, how are you?\n<mnx> I'm fine, thanks!\n<end>", "The quick brown fox jumps over the lazy dog 1234.", "def add(a, b):\n    return a + b", "Café naïve – emoji 🙂 ok"] * 50
tk.train_from_iterator(corpus, trainers.BpeTrainer(vocab_size=C['vocab'], special_tokens=["<pad>", "<user>", "<mnx>", "<end>"], initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
tk.save(os.path.join(OUT, "tok16k.json"))

samples = ["<user> Hello there, how are you?\n<mnx>", "The quick brown fox", "def add(a, b):\n    return a + b", "Café naïve – emoji 🙂 ok", "  spaces   and\ttabs\n\n"]
ids = torch.tensor([tk.encode(samples[0] + " I'm fine").ids])
cur['ids'] = ex_index(ids)
with torch.no_grad(): logits = core(ids, expert_fn)[0]
json.dump(dict(tokens=ids[0].tolist(), logits=logits.numpy().round(5).tolist(), ex_ids=cur['ids'][0].tolist(),
               tokenizer=[dict(text=s, ids=tk.encode(s).ids) for s in samples]), open(os.path.join(OUT, "reference.json"), "w"))
print("wrote", OUT, os.listdir(OUT))

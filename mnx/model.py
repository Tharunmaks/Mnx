"""MNX 2.0 Max Coder model: a decoder-only transformer written from scratch.

Components:
    * RMSNorm pre-normalisation
    * Rotary position embeddings (RoPE)
    * Grouped-query attention with causal masking via
      ``torch.nn.functional.scaled_dot_product_attention``
    * SwiGLU feed-forward network
    * Tied input/output embeddings
    * KV-cache aware incremental decoding in ``generate()``
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

KVCache = List[Tuple[torch.Tensor, torch.Tensor]]


@dataclass(frozen=True)
class MNXConfig:
    """Hyper-parameters describing an ``MNXCoder`` model."""

    name: str = "mnx-2.0-max-coder"
    vocab_size: int = 1024
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    n_kv_heads: int = 2
    d_ff: int = 352
    max_seq_len: int = 256
    dropout: float = 0.0
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    tie_embeddings: bool = True
    init_std: float = 0.02
    bias: bool = False

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if (self.d_model // self.n_heads) % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def replace(self, **kwargs) -> "MNXConfig":
        return dataclasses.replace(self, **kwargs)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "MNXConfig":
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


# ---------------------------------------------------------------- components
class RMSNorm(nn.Module):
    """Root-mean-square layer normalisation (no mean centring, no bias)."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x * rms).to(dtype) * self.weight


def precompute_rope(head_dim: int, max_seq_len: int, theta: float = 10000.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``cos`` and ``sin`` tables of shape ``(max_seq_len, head_dim // 2)``."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    positions = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate ``x`` of shape ``(B, H, T, D)`` using half-split RoPE.

    ``cos``/``sin`` have shape ``(T, D // 2)`` and are broadcast over batch
    and heads.
    """
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    cos = cos.to(x.dtype)[None, None, :, :]
    sin = sin.to(x.dtype)[None, None, :, :]
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    return torch.cat([out1, out2], dim=-1)


class GroupedQueryAttention(nn.Module):
    def __init__(self, cfg: MNXConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.groups = cfg.n_heads // cfg.n_kv_heads
        self.dropout = cfg.dropout

        self.wq = nn.Linear(cfg.d_model, cfg.n_heads * cfg.head_dim, bias=cfg.bias)
        self.wk = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=cfg.bias)
        self.wv = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=cfg.bias)
        self.wo = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.d_model, bias=cfg.bias)
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if past_kv is not None:
            pk, pv = past_kv
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)
        new_kv = (k, v) if use_cache else None

        if self.groups > 1:
            k = k.repeat_interleave(self.groups, dim=1)
            v = v.repeat_interleave(self.groups, dim=1)

        S = k.shape[2]
        if S == T:
            # Plain causal attention (no cache, or prefill into an empty cache).
            y = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
            )
        else:
            # Queries attend to all cached keys plus causally to the new ones.
            past = S - T
            mask = torch.ones(T, S, dtype=torch.bool, device=x.device).tril(diagonal=past)
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, dropout_p=self.dropout if self.training else 0.0
            )

        y = y.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.head_dim)
        return self.resid_dropout(self.wo(y)), new_kv


class SwiGLU(nn.Module):
    """Gated feed-forward network: ``down(silu(gate(x)) * up(x))``."""

    def __init__(self, cfg: MNXConfig) -> None:
        super().__init__()
        self.gate = nn.Linear(cfg.d_model, cfg.d_ff, bias=cfg.bias)
        self.up = nn.Linear(cfg.d_model, cfg.d_ff, bias=cfg.bias)
        self.down = nn.Linear(cfg.d_ff, cfg.d_model, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down(F.silu(self.gate(x)) * self.up(x)))


class TransformerBlock(nn.Module):
    def __init__(self, cfg: MNXConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = GroupedQueryAttention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.ffn = SwiGLU(cfg)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        a, new_kv = self.attn(self.attn_norm(x), cos, sin, past_kv, use_cache)
        x = x + a
        x = x + self.ffn(self.ffn_norm(x))
        return x, new_kv


# -------------------------------------------------------------------- model
class MNXCoder(nn.Module):
    """Decoder-only transformer language model for source code."""

    def __init__(self, cfg: MNXConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(TransformerBlock(cfg) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Recompute activations in the backward pass instead of storing them
        # (trades ~30% extra compute for a large cut in activation memory).
        self.gradient_checkpointing = False

        self.register_buffer("rope_cos", torch.empty(0), persistent=False)
        self.register_buffer("rope_sin", torch.empty(0), persistent=False)

        self.init_weights()

    @torch.no_grad()
    def init_weights(self) -> None:
        """(Re-)initialise all parameters and the RoPE tables in place.

        Called from ``__init__``; call it again after materialising a model
        that was built on the ``meta`` device (``model.to_empty(...)``).
        """
        cfg = self.cfg
        if cfg.tie_embeddings:
            # ``to_empty`` allocates each module's parameters separately,
            # which unties shared weights; restore the tie.
            self.lm_head.weight = self.tok_emb.weight
        device = self.tok_emb.weight.device
        cos, sin = precompute_rope(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.rope_cos = cos.to(device)
        self.rope_sin = sin.to(device)
        if device.type == "meta":
            return
        self.apply(self._init_weights)
        # Scale residual-branch output projections by depth (GPT-2 style).
        for name, p in self.named_parameters():
            if name.endswith("attn.wo.weight") or name.endswith("ffn.down.weight"):
                nn.init.normal_(p, mean=0.0, std=cfg.init_std / math.sqrt(2 * cfg.n_layers))

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.init_std)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)

    # ------------------------------------------------------------ utilities
    def num_parameters(self, non_embedding: bool = False) -> int:
        """Count unique parameters (tied weights are counted once)."""
        seen = set()
        total = 0
        for name, p in self.named_parameters():
            if id(p) in seen:
                continue
            seen.add(id(p))
            if non_embedding and name.startswith("tok_emb"):
                continue
            total += p.numel()
        return total

    @staticmethod
    def format_params(n: int) -> str:
        if n >= 1e9:
            return f"{n / 1e9:.2f}B"
        if n >= 1e6:
            return f"{n / 1e6:.2f}M"
        if n >= 1e3:
            return f"{n / 1e3:.1f}K"
        return str(n)

    # -------------------------------------------------------------- forward
    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        past_kv: Optional[KVCache] = None,
        use_cache: bool = False,
        ignore_index: int = -100,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[KVCache]]:
        """Run the model.

        Args:
            idx: ``(B, T)`` token ids.
            targets: optional ``(B, T)`` ids for next-token cross-entropy.
            past_kv: optional per-layer cached keys/values from a previous call.
            use_cache: return the updated KV cache.

        Returns:
            ``(logits, loss, new_kv)``; ``loss`` is ``None`` without targets
            and ``new_kv`` is ``None`` unless ``use_cache``.
        """
        B, T = idx.shape
        start = past_kv[0][0].shape[2] if past_kv is not None else 0
        if start + T > self.cfg.max_seq_len:
            raise ValueError(
                f"sequence length {start + T} exceeds max_seq_len {self.cfg.max_seq_len}"
            )
        cos = self.rope_cos[start : start + T]
        sin = self.rope_sin[start : start + T]

        x = self.drop(self.tok_emb(idx))
        new_cache: KVCache = []
        for i, block in enumerate(self.blocks):
            layer_past = past_kv[i] if past_kv is not None else None
            if self.gradient_checkpointing and self.training and not use_cache:
                x, kv = torch.utils.checkpoint.checkpoint(block, x, cos, sin, use_reentrant=False)
            else:
                x, kv = block(x, cos, sin, layer_past, use_cache)
            if use_cache:
                new_cache.append(kv)
        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)).float(),
                targets.reshape(-1),
                ignore_index=ignore_index,
            )
        return logits, loss, (new_cache if use_cache else None)

    # ------------------------------------------------------------- sampling
    @staticmethod
    def _sample_next(
        logits: torch.Tensor,
        temperature: float,
        top_k: Optional[int],
        top_p: Optional[float],
    ) -> torch.Tensor:
        """Sample one token per row from ``(B, V)`` logits."""
        if temperature <= 0:
            return logits.argmax(dim=-1, keepdim=True)
        logits = logits / temperature
        if top_k is not None and top_k > 0:
            k = min(top_k, logits.size(-1))
            kth = torch.topk(logits, k, dim=-1).values[:, -1, None]
            logits = logits.masked_fill(logits < kth, float("-inf"))
        if top_p is not None and 0.0 < top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
            probs = F.softmax(sorted_logits, dim=-1)
            cum = probs.cumsum(dim=-1)
            # Remove tokens whose cumulative mass (excluding themselves) exceeds top_p.
            remove = (cum - probs) > top_p
            sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
            logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        eos_id: Optional[int] = None,
        stop_ids: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """Autoregressively extend ``idx`` (``(B, T)``) by up to ``max_new_tokens``.

        Uses a KV cache so each step only processes the newest token. If the
        context would exceed ``max_seq_len`` the cache is rebuilt from the most
        recent ``max_seq_len - 1`` tokens (sliding window). Generation stops
        early once every row has produced ``eos_id`` (or any of ``stop_ids``);
        rows that already finished are padded with ``eos_id``.
        """
        was_training = self.training
        self.eval()
        stops = set(stop_ids or [])
        if eos_id is not None:
            stops.add(eos_id)
        B = idx.size(0)
        finished = torch.zeros(B, dtype=torch.bool, device=idx.device)
        max_len = self.cfg.max_seq_len

        # When the context overflows we re-prefill from a window somewhat
        # shorter than max_seq_len so the rebuild is amortised over many steps.
        window = max_len - max(1, max_len // 8)

        past: Optional[KVCache] = None
        pending = idx  # tokens not yet fed through the model
        for _ in range(max_new_tokens):
            cache_len = past[0][0].shape[2] if past is not None else 0
            if cache_len + pending.size(1) > max_len:
                past = None
                pending = idx[:, -window:]
            logits, _, past = self(pending, past_kv=past, use_cache=True)
            next_tok = self._sample_next(logits[:, -1, :], temperature, top_k, top_p)
            if stops:
                fill = eos_id if eos_id is not None else next(iter(stops))
                next_tok = torch.where(finished[:, None], torch.full_like(next_tok, fill), next_tok)
                finished = finished | torch.isin(next_tok[:, 0], torch.tensor(sorted(stops), device=idx.device))
            idx = torch.cat([idx, next_tok], dim=1)
            pending = next_tok
            if stops and bool(finished.all()):
                break

        if was_training:
            self.train()
        return idx

    def extra_repr(self) -> str:
        return f"params={self.format_params(self.num_parameters())}, cfg={self.cfg}"

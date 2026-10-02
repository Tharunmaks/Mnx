"""Find language models on Hugging Face for the Cloud Bee.

Searches the Hub when it's reachable, and falls back to a built-in list of popular
open models so a request like "train Qwen 2.5 Coder" still works offline.
"""

from __future__ import annotations

import re

import httpx

HF_API = "https://huggingface.co/api/models"

# (model id, billions of parameters, gated: needs an accepted licence + HF_TOKEN)
CATALOG = [
    ("Qwen/Qwen2.5-Coder-0.5B-Instruct", 0.5, False),
    ("Qwen/Qwen2.5-Coder-1.5B-Instruct", 1.5, False),
    ("Qwen/Qwen2.5-Coder-3B-Instruct", 3, False),
    ("Qwen/Qwen2.5-Coder-7B-Instruct", 7, False),
    ("Qwen/Qwen2.5-Coder-14B-Instruct", 14, False),
    ("Qwen/Qwen2.5-0.5B-Instruct", 0.5, False),
    ("Qwen/Qwen2.5-1.5B-Instruct", 1.5, False),
    ("Qwen/Qwen2.5-3B-Instruct", 3, False),
    ("Qwen/Qwen2.5-7B-Instruct", 7, False),
    ("HuggingFaceTB/SmolLM2-360M-Instruct", 0.36, False),
    ("HuggingFaceTB/SmolLM2-1.7B-Instruct", 1.7, False),
    ("deepseek-ai/deepseek-coder-1.3b-instruct", 1.3, False),
    ("microsoft/Phi-3.5-mini-instruct", 3.8, False),
    ("meta-llama/Llama-3.2-1B-Instruct", 1, True),
    ("meta-llama/Llama-3.2-3B-Instruct", 3, True),
    ("google/gemma-2-2b-it", 2.6, True),
    ("mistralai/Mistral-7B-Instruct-v0.3", 7, False),
]

# Forks that can't be fine-tuned with plain transformers + LoRA.
SKIP = re.compile(r"gguf|awq|gptq|exl2|mlx|onnx|int4|int8|4bit|8bit|bnb|fp8|-q\d", re.I)
SIZE = re.compile(r"(?<![\d.])(\d+(?:\.\d+)?)\s*([bm])(?![a-z])", re.I)


def size_of(model_id: str) -> float | None:
    """Billions of parameters, read from the name ("Qwen2.5-Coder-1.5B" → 1.5, "SmolLM2-360M" → 0.36)."""
    m = SIZE.search(model_id.split("/")[-1].replace("_", "-"))
    if not m:
        return None
    n = float(m.group(1))
    return round(n / 1000, 3) if m.group(2).lower() == "m" else n


def tokens(query: str) -> list[str]:
    """'Qwen 2.5 coder' → ['qwen', '2.5', 'coder'] (version numbers stay whole)."""
    return [t for t in re.split(r"[\s/_\-]+", query.lower()) if t]


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def matches(model_id: str, query_tokens: list[str]) -> bool:
    n = _norm(model_id)
    return all(_norm(t) in n for t in query_tokens if _norm(t))


def search_string(query: str) -> str:
    """'qwen 2.5 coder' → 'qwen2.5-coder', the way model ids are spelled on the Hub."""
    q = re.sub(r"([a-z])\s+(\d)", r"\1\2", query.lower().strip())
    return re.sub(r"\s+", "-", q)


def _rank(model: dict) -> tuple:
    mid = model["id"]
    official = mid.split("/")[0].lower() in {"qwen", "meta-llama", "google", "microsoft", "mistralai",
                                              "huggingfacetb", "deepseek-ai", "ibm-granite", "allenai"}
    instruct = bool(re.search(r"instruct|chat|-it\b", mid, re.I))
    return (not official, not instruct, -(model.get("downloads") or 0))


async def find_models(query: str, limit: int = 8) -> tuple[list[dict], str]:
    """Returns (models, source) with source 'hub' or 'catalog'. Each model: id, size_b, gated, downloads."""
    qt = tokens(query)
    if re.fullmatch(r"[\w.-]+/[\w.-]+", query.strip()):
        mid = query.strip()
        return [{"id": mid, "size_b": size_of(mid), "gated": False, "downloads": None}], "exact"
    found: list[dict] = []
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(HF_API, params={"search": search_string(query), "sort": "downloads",
                                                 "direction": "-1", "limit": 60, "filter": "text-generation"})
            r.raise_for_status()
            for m in r.json():
                mid = m.get("id") or m.get("modelId") or ""
                if not mid or SKIP.search(mid) or not matches(mid, qt):
                    continue
                found.append({"id": mid, "size_b": size_of(mid), "gated": bool(m.get("gated")),
                              "downloads": m.get("downloads")})
        source = "hub"
    except (httpx.HTTPError, ValueError):
        source = "catalog"
    if not found:
        found = [{"id": mid, "size_b": size, "gated": gated, "downloads": None}
                 for mid, size, gated in CATALOG if matches(mid, qt)]
        source = "catalog" if source == "catalog" or found else source
    found.sort(key=_rank)
    found = found[:limit]
    found.sort(key=lambda m: (m["size_b"] is None, m["size_b"] or 0))
    return found, source


def suggested_dataset(model_id: str) -> str:
    """A public dataset that suits the model's job, ready to use with the fine-tune recipe."""
    if re.search(r"coder|code", model_id, re.I):
        return "iamtarun/python_code_instructions_18k_alpaca"
    return "HuggingFaceH4/no_robots"


def memory_needed_gb(size_b: float | None, gpu: bool) -> float | None:
    """Rough memory for a LoRA fine-tune: bf16 weights + activations on GPU, fp32 on CPU."""
    if not size_b:
        return None
    return round(size_b * (2.6 if gpu else 5.2) + 2, 1)

"""MNX 2.0 Max Coder: a from-scratch decoder-only transformer for source code.

Submodules are imported lazily so that ``python -m mnx.<module>`` does not
trigger double-import warnings.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

__version__ = "2.0.0"

_LAZY = {
    "BPETokenizer": ("mnx.tokenizer", "BPETokenizer"),
    "SPECIAL_TOKENS": ("mnx.tokenizer", "SPECIAL_TOKENS"),
    "MNXCoder": ("mnx.model", "MNXCoder"),
    "MNXConfig": ("mnx.model", "MNXConfig"),
    "PRESETS": ("mnx.configs", "PRESETS"),
    "get_config": ("mnx.configs", "get_config"),
}

__all__ = [*_LAZY, "__version__"]

if TYPE_CHECKING:  # pragma: no cover
    from mnx.configs import PRESETS, get_config
    from mnx.model import MNXCoder, MNXConfig
    from mnx.tokenizer import SPECIAL_TOKENS, BPETokenizer


def __getattr__(name: str):
    try:
        module_name, attr = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module 'mnx' has no attribute {name!r}") from None
    return getattr(importlib.import_module(module_name), attr)

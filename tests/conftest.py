from pathlib import Path

import pytest

try:
    import torch
except ImportError:  # the agent tests run without PyTorch
    torch = None

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = REPO_ROOT / "data" / "sample"

# The transformer tests need PyTorch; skip them cleanly when it isn't installed.
collect_ignore = [] if torch is not None else [
    "test_data.py", "test_model.py", "test_tokenizer.py", "test_train_generate.py",
]


@pytest.fixture(autouse=True)
def _deterministic():
    if torch is not None:
        torch.manual_seed(0)
        torch.set_num_threads(min(4, torch.get_num_threads()))


@pytest.fixture(scope="session")
def sample_dir() -> Path:
    return SAMPLE_DIR


@pytest.fixture(scope="session")
def tokenizer(sample_dir):
    from mnx.tokenizer import BPETokenizer, iter_source_files

    return BPETokenizer.train_from_files(iter_source_files(sample_dir), vocab_size=512)

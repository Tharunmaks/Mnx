from pathlib import Path

import pytest
import torch

from mnx.tokenizer import BPETokenizer, iter_source_files

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = REPO_ROOT / "data" / "sample"


@pytest.fixture(autouse=True)
def _deterministic():
    torch.manual_seed(0)
    torch.set_num_threads(min(4, torch.get_num_threads()))


@pytest.fixture(scope="session")
def sample_dir() -> Path:
    return SAMPLE_DIR


@pytest.fixture(scope="session")
def tokenizer(sample_dir) -> BPETokenizer:
    return BPETokenizer.train_from_files(iter_source_files(sample_dir), vocab_size=512)

"""Every test starts on the Free plan with its own plan file, whatever the Hive in data/ is on."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture(autouse=True)
def _own_plan(tmp_path, monkeypatch):
    from hive import plan as plan_mod
    monkeypatch.delenv("MNX_PLAN", raising=False)
    monkeypatch.setattr(plan_mod.plan, "path", tmp_path / "plan-isolated.json")

"""Every test starts on the Free plan with no models set up, whatever the Hive in data/ has."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture(autouse=True)
def _own_plan(tmp_path, monkeypatch):
    from hive import plan as plan_mod
    monkeypatch.delenv("MNX_PLAN", raising=False)
    monkeypatch.setattr(plan_mod.plan, "path", tmp_path / "plan-isolated.json")


@pytest.fixture(autouse=True)
def _own_models(tmp_path, monkeypatch):
    from hive import models as models_mod
    monkeypatch.setattr(models_mod, "models", models_mod.Models(tmp_path / "models-isolated"))

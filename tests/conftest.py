"""Every test starts with no Bees, no payments and no models set up, whatever the Hive in data/ has; nothing a test
does is written to data/."""

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
def _own_hive(tmp_path, monkeypatch):
    from hive import billing as billing_mod, bots as bots_mod
    from hive.bees import registry
    monkeypatch.setattr(registry, "state_file", tmp_path / "hive-isolated.json")
    monkeypatch.setattr(registry, "keys_file", tmp_path / "keys-isolated.json")
    monkeypatch.setattr(registry, "keys", {})
    saved = list(registry.bees)
    registry.bees.clear()  # in place: the gateway holds the same list
    monkeypatch.setattr(billing_mod.billing, "orders_file", tmp_path / "orders-isolated.json")
    monkeypatch.setattr(bots_mod, "store", bots_mod.Store(tmp_path / "bots-isolated"))
    yield
    registry.bees[:] = saved


@pytest.fixture(autouse=True)
def _own_models(tmp_path, monkeypatch):
    from hive import expert, models as models_mod
    monkeypatch.setattr(expert, "es_serve_path", lambda: None)  # not whatever engine this machine has built
    monkeypatch.setattr(models_mod, "models", models_mod.Models(tmp_path / "models-isolated"))

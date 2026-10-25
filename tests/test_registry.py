"""The agent registry and the skill that uses it."""

from __future__ import annotations

import gzip
import json

import pytest

from mnx.agent.registry import (Agent, Registry, base_agents, call, check, date_agents, families,
                                list_agents, stdlib_agents, unit_agents, written_agents)
from mnx.agent.skills.agents import RegistrySkill, arguments_in, params_of


def test_every_generated_agent_passes_its_own_tests():
    """A sample across the families: if an agent's asserts fail it is not an agent."""
    import itertools

    for maker in (base_agents(), unit_agents(), date_agents(3), list_agents(4), written_agents()):
        for agent in itertools.islice(maker, 40):
            ok, why = check(agent)
            assert ok, f"{agent.id}: {why}"


def test_a_broken_agent_is_rejected():
    bad = Agent(id="x.broken", family="number", name="broken", description="wrong on purpose",
                tier="verified", code="def broken(n):\n    return n + 1\n", tests=["assert broken(1) == 99"])
    ok, why = check(bad)
    assert not ok and "AssertionError" in why


def test_running_an_agent_gives_the_documented_answer():
    agents = {a.id: a for a in base_agents(7, 7)}
    assert call(agents["base.to_base_7"], 100) == "202"
    agents = {a.id: a for a in unit_agents()}
    assert abs(call(agents["unit.celsius_to_fahrenheit"], 100) - 212.0) < 1e-9
    assert abs(call(agents["unit.mile_to_kilometre"], 1) - 1.609344) < 1e-9


def test_reference_agents_are_pointers_not_answers():
    import itertools

    ref = next(itertools.islice(stdlib_agents(limit=5), 1))
    assert ref.tier == "reference" and ref.module and not ref.code
    assert check(ref)[0]  # nothing to run, so nothing fails
    with pytest.raises(ValueError):
        call(ref, 1)


def test_the_registry_searches_by_what_an_agent_is_for(tmp_path):
    reg = Registry(list(base_agents(2, 4)) + list(written_agents()))
    best = reg.search("median of a list of numbers", k=1)
    assert best and best[0][1].id == "stats.median_of"
    path = reg.write(tmp_path)
    assert path.exists()
    back = Registry.load(tmp_path)
    assert len(back) == len(reg) and back.counts() == reg.counts()
    rows = [json.loads(l) for l in gzip.open(path, "rt")]
    assert all(r["id"] and r["tier"] in ("verified", "reference") for r in rows)


def test_arguments_are_read_out_of_a_question():
    assert params_of("def f(xs, k):\n    return xs\n", "f") == ["xs", "k"]
    assert arguments_in("median of [3, 1, 2]", 1) == [[3, 1, 2]]
    # the number in the agent's name chose the agent and must not be its argument too
    assert arguments_in("to base 7 of 100", 1, parameter="7") == [100]
    assert arguments_in("rotate [1,2,3] by 2", 1, parameter="2") == [[1, 2, 3]]
    assert arguments_in("what weekday is 2026-07-04", 1) == ["2026-07-04"]
    assert arguments_in("no arguments here", 1) is None


@pytest.mark.parametrize("question,expected", [
    ("convert 100 celsius to fahrenheit", "212.0"),
    ("to base 7 of 100", "'202'"),
    ("top 3 of [5,1,9,7,3]", "[9, 7, 5]"),
    ("prime factors of 360", "[2, 2, 2, 3, 3, 5]"),
    ("what quarter is 2026-11-02", "4"),
])
def test_the_skill_answers_from_a_verified_agent(question, expected):
    ans = RegistrySkill().answer(question)
    assert ans is not None, question
    assert ans.verified and ans.text.splitlines()[0] == expected


def test_the_skill_keeps_out_of_the_coder_s_way():
    """A def plus asserts belongs to the coder, which tests against the user's own tests."""
    assert RegistrySkill().answer("def shout(s):\n    assert shout('hi') == 'HI!'") is None
    assert RegistrySkill().answer("write a python function that reverses a string") is None


def test_the_skill_says_nothing_when_it_cannot_be_sure():
    skill = RegistrySkill()
    assert skill.answer("hello how are you") is None
    assert skill.answer("what is love") is None

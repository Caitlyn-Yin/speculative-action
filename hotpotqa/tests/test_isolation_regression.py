"""P5 regression test — speculation must not contaminate the realized trajectory.

The bug: the speculative branch steps the *same* env object as the realized
trajectory (runner.py, `self.step(self.env, ..., simulate=True)`), and
`WikiEnv.guess_step()` assigns `self.page` / `self.obs` unconditionally even
under `simulate=True` (environment.py:94-97). The damage surfaces at the next
realized `lookup[...]`, which reads `self.page` via `construct_lookup_list()`
(environment.py:64-73).

Invariant under test: for a fixed seed and a scripted speculator, the realized
`lookup[]` observation at turn >= 2 is byte-identical whether speculation is on
or off.

The test is run in both directions on purpose:
  * isolate_speculation=True  -> realized lookup[] must be IDENTICAL
  * isolate_speculation=False -> realized lookup[] must DIVERGE
A green result in the second direction would mean the test cannot see the bug
it exists to catch, so divergence there is asserted, not tolerated.

No network and no LLM: Wikipedia and the speculator are both scripted.
"""

import pytest

from src import constants
from src.runner import HotPotQARun


SEED = 248  # constants.random_seed

# Deterministic stand-in for a real Wikipedia page; sentence-split by
# get_page_obs()/construct_lookup_list() on ". ".
REAL_PAGE = (
    "Scott Derrickson is an American director. "
    "He was born in Denver, Colorado. "
    "He directed the film Sinister. "
    "Derrickson also directed Doctor Strange.\n"
)

# What the scripted speculator "imagines" instead. Deliberately shares the
# lookup keyword so the contaminated lookup still returns a hit rather than
# "No more results" -- a subtler, more realistic corruption.
SPEC_PAGE = (
    "Scott Derrickson is a Canadian novelist. "
    "He was born in Toronto, Ontario. "
    "He wrote the novel Winter Light.\n"
)

LOOKUP_KEYWORD = "born"


def _build_runner(monkeypatch, isolate):
    """Runner with a scripted env: no Wikipedia, no LLM, no randomness."""
    monkeypatch.setattr(constants, "isolate_speculation", isolate)

    runner = HotPotQARun(
        model_name="scripted-actor",
        guess_model_name="scripted-speculator",
        to_print_output=False,
    )
    base = runner.env.unwrapped

    # Scripted Wikipedia: search[...] always yields REAL_PAGE.
    def fake_search_step(entity):
        base.page = REAL_PAGE
        base.obs = base.get_page_obs(base.page)
        base.lookup_keyword = base.lookup_list = base.lookup_cnt = None

    monkeypatch.setattr(base, "search_step", fake_search_step)

    # Scripted speculator: guess_llm.call() returns SPEC_PAGE verbatim, so
    # guess_step() writes it into base.page exactly as the real one would.
    monkeypatch.setattr(base.guess_llm, "call", lambda prompt, stop=None: SPEC_PAGE)

    runner.env.reset(idx=SEED)
    return runner


def _run_episode(monkeypatch, isolate, speculate):
    """Turn 1: realized search + optional speculation. Turn 2: realized lookup.

    Returns the realized turn-2 lookup observation.
    """
    runner = _build_runner(monkeypatch, isolate)
    env = runner.env

    # --- turn 1: realized action ---
    runner.step(env, "search[Scott Derrickson]")

    # --- turn 1: speculative branch (same env object, as in runner.webthink) ---
    if speculate:
        sim_obs, _, _, _, _ = runner.step(env, "search[Scott Derrickson]", simulate=True)
        # The speculation itself must still be observable to the caller.
        assert sim_obs, "speculative branch returned no observation"

    # --- turn 2: realized lookup, the contamination-sensitive read ---
    obs, _, _, _, _ = runner.step(env, f"lookup[{LOOKUP_KEYWORD}]")
    return obs


def test_realized_lookup_is_identical_when_isolated(monkeypatch):
    """PASS condition of the gate: speculation leaves no trace."""
    without_spec = _run_episode(monkeypatch, isolate=True, speculate=False)
    with_spec = _run_episode(monkeypatch, isolate=True, speculate=True)

    assert with_spec == without_spec, (
        "realized lookup[] diverged with speculation on:\n"
        f"  without speculation: {without_spec!r}\n"
        f"  with speculation:    {with_spec!r}"
    )
    # Guard against a vacuous pass (e.g. both sides "No more results.").
    assert "Denver" in without_spec, (
        f"fixture no longer exercises the real page: {without_spec!r}"
    )


def test_realized_lookup_diverges_when_unisolated(monkeypatch):
    """The test must be able to see the bug: upstream behaviour is contaminated."""
    without_spec = _run_episode(monkeypatch, isolate=False, speculate=False)
    with_spec = _run_episode(monkeypatch, isolate=False, speculate=True)

    assert with_spec != without_spec, (
        "unisolated arm did NOT diverge -- the regression test is blind and "
        "its green result in the isolated arm is meaningless"
    )
    assert "Denver" in without_spec and "Toronto" in with_spec, (
        "divergence is not the expected page-substitution:\n"
        f"  without speculation: {without_spec!r}\n"
        f"  with speculation:    {with_spec!r}"
    )


def test_snapshot_restores_every_declared_field(monkeypatch):
    """Snapshot/restore must cover each field it claims to cover."""
    runner = _build_runner(monkeypatch, isolate=True)
    base = runner.env.unwrapped
    runner.step(runner.env, "search[Scott Derrickson]")
    runner.step(runner.env, f"lookup[{LOOKUP_KEYWORD}]")

    before = {f: getattr(base, f, None) for f in HotPotQARun._SPEC_ISOLATED_FIELDS}
    runner.step(runner.env, "search[Scott Derrickson]", simulate=True)
    after = {f: getattr(base, f, None) for f in HotPotQARun._SPEC_ISOLATED_FIELDS}

    assert after == before, (
        "speculative step mutated snapshotted state: "
        f"{[k for k in before if before[k] != after[k]]}"
    )


@pytest.mark.parametrize("field", ["page", "obs"])
def test_unisolated_arm_clobbers_state(monkeypatch, field):
    """Documents precisely which fields upstream corrupts (environment.py:96-97)."""
    runner = _build_runner(monkeypatch, isolate=False)
    base = runner.env.unwrapped
    runner.step(runner.env, "search[Scott Derrickson]")

    before = getattr(base, field)
    runner.step(runner.env, "search[Scott Derrickson]", simulate=True)
    assert getattr(base, field) != before, (
        f"expected upstream to clobber {field}; fixture may be stale"
    )

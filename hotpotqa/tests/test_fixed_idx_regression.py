"""Regression test — a requested question index must actually be used.

The bug: `HistoryWrapper.reset` accepted `idx` and then called the inner env
with a hardcoded `idx=None` (wrappers.py). `HotPotQAWrapper.reset` treats
`idx=None` as "pick at random" (`np.random.randint(len(self.data))`), so
`runner.webthink(idx=i)` silently ran a *different, random* question on every
reset.

Why it matters beyond tidiness:
  * the isolation gate compares arms over fixed question indices -- with the
    bug, each arm runs a different question and every comparison is noise;
  * `runner.run()` computes a seeded shuffle and then never uses it, so a run
    labelled with an index was not that index.

No network and no LLM: the inner env is a stub.
"""

import pytest

from src import wrappers


class StubEnv:
    """Minimal stand-in for WikiEnv -- only what the wrappers touch."""

    def __init__(self):
        self.steps = 0
        self.answer = None
        self.page = None
        self.traj = {"observations": ["init"], "actions": []}

    def reset(self, seed=None, return_info=False, options=None):
        self.steps = 0
        self.answer = None
        return "obs"

    def step(self, action, step_type="wiki"):
        self.steps += 1
        return "obs", 0, False, {}


def _stack():
    env = wrappers.HotPotQAWrapper(StubEnv(), split="dev")
    return wrappers.HistoryWrapper(env, obs_format="history")


REQUESTED = [7107, 5619, 373, 6904, 1267]  # the gate's pilot idxs


@pytest.mark.parametrize("idx", REQUESTED)
def test_requested_idx_is_honoured_through_the_wrapper_stack(idx):
    env = _stack()
    env.reset(idx=idx)
    assert env.env.data_idx == idx


def test_requested_idx_is_stable_across_repeated_resets():
    """The pre-fix failure mode was a *different* random draw each reset."""
    env = _stack()
    seen = set()
    for _ in range(5):
        env.reset(idx=7107)
        seen.add(env.env.data_idx)
    assert seen == {7107}, f"idx not stable across resets: {sorted(seen)}"


def test_question_text_matches_the_requested_index():
    """Guards the whole path: the observation must be that index's question."""
    env = _stack()
    obs = env.reset(idx=7107)
    expected = env.env.data[7107][0]
    assert expected in obs


def test_idx_none_still_draws_randomly():
    """The random path is upstream behaviour and must stay reachable."""
    env = _stack()
    seen = set()
    for _ in range(25):
        env.reset(idx=None)
        seen.add(env.env.data_idx)
    assert len(seen) > 1, "idx=None should still draw at random"

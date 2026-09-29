"""Tests for the Paper B collection / replay machinery (``src/paperb.py``).

Offline: no server, no Wikipedia, no weights.  The actor is a scripted list of
replies and ``search_step`` is a scripted page table, so what is tested is the
replay engine's own behaviour -- that a forced prefix is executed verbatim, that
the treatment arm records the **executed** observation of ``spec_j`` rather than
the speculator's imagined one, that the control/log comparison can both pass and
fail, and that the labels follow ``docs/PREREG_PAPER_B.md``.

Runs under pytest and standalone (``$PIPELINE_PY tests/test_paperb.py``), the
same as ``tests/test_gates.py``.  It needs the repo package (pandas via
``src/utils.py``), so the system python will not do.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import constants                 # noqa: E402
from src import gates                     # noqa: E402
from src import paperb                    # noqa: E402
from src.runner import HotPotQARun        # noqa: E402

IDX = 0  # any fixed dev question; the scripted env ignores its content

PAGES = {
    "scott derrickson": (
        "Scott Derrickson is an American director. "
        "He was born in Denver, Colorado. "
        "He directed the film Sinister.\n"),
    "ed wood": (
        "Edward Davis Wood Jr. was an American filmmaker. "
        "He was born in Poughkeepsie, New York.\n"),
}


class ScriptedActor:
    """Returns canned ReAct replies in order; records the prompts it saw."""

    def __init__(self, replies, model_name="scripted"):
        self.replies = list(replies)
        self.prompts = []
        self.model_name = model_name

    def call(self, prompt, stop=None):
        self.prompts.append(prompt)
        if not self.replies:
            return "Thought 9: I am out of script.\nAction 9: finish[out of script]"
        return self.replies.pop(0)

    def call_with_logprobs(self, prompt, max_tokens=None, top_logprobs=20, stop=None):
        text = self.call(prompt, stop=stop)
        return {"text": text,
                "tokens": [{"token": t, "logprob": -0.1, "top": {t: -0.1}}
                           for t in text.split(" ")]}

    @staticmethod
    def _strip_thinking(text):
        return text


def build_runner(actor_replies):
    runner = HotPotQARun(model_name="scripted-actor",
                         guess_model_name="scripted-speculator",
                         to_print_output=False)
    actor = ScriptedActor(actor_replies)
    runner.llm = actor
    base = runner.env.unwrapped

    def fake_search_step(entity):
        page = PAGES.get(entity.strip().lower())
        if page is None:
            base.obs = f"Could not find {entity}. Similar: []."
            return
        base.page = page
        base.obs = base.get_page_obs(page)
        base.lookup_keyword = base.lookup_list = base.lookup_cnt = None

    base.search_step = fake_search_step
    # A speculated page must never reach a Paper B arm; make it loud if it does.
    base.guess_llm.call = lambda prompt, stop=None: "IMAGINED PAGE, MUST NOT APPEAR.\n"
    return runner, actor


PROMPT = "PROMPT-HEAD\n"


# ---------------------------------------------------------------------------
# replay(): the forced prefix
# ---------------------------------------------------------------------------

def test_replay_executes_the_forced_prefix_verbatim():
    runner, _actor = build_runner(["Thought 3: done.\nAction 3: finish[sinister]"])
    out = paperb.replay(runner, IDX, PROMPT, [
        {"thought": "find the director", "action": "search[Scott Derrickson]"},
        {"thought": "where was he born", "action": "lookup[born]"},
    ])
    assert [s["action"] for s in out["steps"]][:2] == [
        "search[Scott Derrickson]", "lookup[born]"]
    assert [s["source"] for s in out["steps"]] == ["forced", "forced", "actor"]
    assert "Denver" in out["steps"][1]["obs"], out["steps"][1]["obs"]
    assert out["n_steps"] == 3
    assert out["terminated"] is True
    assert out["answer"] == "sinister"


def test_replay_observations_are_executed_not_imagined():
    runner, _ = build_runner(["Thought 2: done.\nAction 2: finish[x]"])
    out = paperb.replay(runner, IDX, PROMPT,
                        [{"thought": "t", "action": "search[Ed Wood]"}])
    obs = out["steps"][0]["obs"]
    assert "Poughkeepsie" in obs
    assert "IMAGINED" not in obs


def test_replay_actor_continuation_sees_the_forced_history():
    runner, actor = build_runner(["Thought 2: done.\nAction 2: finish[denver]"])
    paperb.replay(runner, IDX, PROMPT,
                  [{"thought": "look him up", "action": "search[Scott Derrickson]"}])
    assert actor.prompts, "the actor was never called"
    first = actor.prompts[0]
    assert first.startswith(PROMPT)
    assert "Action 1: search[Scott Derrickson]" in first
    assert "look him up" in first
    assert "Denver" in first          # the forced step's real observation


def test_replay_honours_the_step_cap():
    # The actor never finishes; the loop must stop at n-1 recorded steps.
    runner, _ = build_runner(["Thought %d: go on.\nAction %d: search[Ed Wood]" % (i, i)
                              for i in range(1, 12)])
    out = paperb.replay(runner, IDX, PROMPT, [], n=4)
    assert out["n_steps"] == 3
    assert all(s["source"] == "actor" for s in out["steps"])


def test_replay_skips_an_unparseable_actor_reply_like_webthink():
    # constants.max_agent_retries == 1, so the first reply plus one retry are
    # consumed on step 1; both are actionless, so step 1 records nothing and the
    # loop moves on -- exactly webthink's `except ValueError: continue`.
    assert constants.max_agent_retries == 1
    runner, _ = build_runner(["I refuse to emit an action.",
                              "I refuse again.",
                              "Thought 2: ok.\nAction 2: finish[z]"])
    out = paperb.replay(runner, IDX, PROMPT, [], n=8)
    assert out["bad_actor_replies"] == 1
    assert [s["i"] for s in out["steps"]] == [2]
    assert out["n_steps"] == 1
    assert out["answer"] == "z"


# ---------------------------------------------------------------------------
# The population filter and pair construction
# ---------------------------------------------------------------------------

def test_in_population_is_the_complement_of_exact_match():
    assert not paperb.in_population("Search[Ed Wood]", "search[ed wood]")
    assert not paperb.in_population(" search[Ed Wood] ", "search[Ed Wood]")
    assert paperb.in_population("search[Edward Wood]", "search[Ed Wood]")
    assert paperb.in_population("lookup[Ed Wood]", "search[Ed Wood]")


def test_same_action_matches_criterion_2():
    ctx = gates.Context()
    for spec, real in (("search[the Beyonce]", "search[Beyoncé]"),
                       ("search[Mercury]", "search[Mercury (planet)]"),
                       ("lookup[born]", "search[born]")):
        pair = gates.Pair(pair_id="t", question="q", real_action=real,
                          spec_action=spec)
        expected = gates.CRITERIA["normalized"].fn(pair, ctx).binary
        assert paperb.same_action(spec, real) is expected, (spec, real)


def _episode():
    return {
        "idx": 42,
        "question": "Who directed Sinister?",
        "gt_answer": "Scott Derrickson",
        "em": 1, "n_steps": 2, "f1": 1.0, "answer": "Scott Derrickson",
        "retrieval_backend": "title_exact", "spec_actions_from": "speculator",
        "spec_model": "Qwen/Qwen3-4B", "actor_model": "Qwen/Qwen3-8B",
        "steps": [
            {"i": 1, "thought": "t1", "action": "search[Sinister (film)]",
             "obs": "Sinister is a 2012 film directed by Scott Derrickson.",
             "spec_actions": ["search[Sinister]", "Search[sinister (film)]",
                              "lookup[director]"],
             "sim_obs": "IMAGINED", "spec_tokens": [{"token": "x", "logprob": -0.2}]},
            {"i": 2, "thought": "t2", "action": "finish[Scott Derrickson]",
             "obs": "Episode finished, reward = 1\n",
             "spec_actions": ["finish[Scott Derrickson]"],
             "sim_obs": None, "spec_tokens": None},
        ],
    }


def test_pairs_from_episode_filters_and_builds_history():
    pairs = paperb.pairs_from_episode(_episode(), "unit")
    # step 1: "Search[sinister (film)]" is an exact match modulo case -> dropped.
    # step 2: the only candidate equals the real action -> dropped.
    assert [p["pair_id"] for p in pairs] == ["unit-q42-s1-j0", "unit-q42-s1-j2"]
    p = pairs[0]
    assert p["history"] == []
    assert p["spec_obs"] is None and p["spec_obs_source"] == "unavailable"
    assert p["real_obs"].startswith("Sinister is a 2012 film")
    assert p["labels"] == {}
    assert p["spec_tokens"][0]["token"] == "x"
    assert p["meta"]["sim_obs_imagined"] == "IMAGINED"
    # Round-trips into the criterion schema.
    pair = gates.Pair.from_dict(p)
    assert pair.step_i == 1 and pair.spec_index == 0


def test_pairs_carry_the_realized_history_of_earlier_steps():
    ep = _episode()
    ep["steps"][1]["spec_actions"] = ["finish[Derrickson]"]
    pairs = paperb.pairs_from_episode(ep, "unit")
    last = pairs[-1]
    assert last["step_i"] == 2
    assert [t["action"] for t in last["history"]] == ["search[Sinister (film)]"]
    assert last["history"][0]["thought"] == "t1"


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

def _arm(actions, em, f1=1.0, answer="a", start=1):
    return {"steps": [{"i": start + n, "action": a}
                      for n, a in enumerate(actions)],
            "n_steps": len(actions), "em": em, "f1": f1, "answer": answer}


def test_labels_s1_s2_s3_on_agreement():
    control = _arm(["search[A]", "search[B]", "finish[x]"], em=1)
    treatment = _arm(["search[A]", "search[the b]", "finish[x]"], em=1)
    lab = paperb.label_pair(1, control, treatment)
    assert lab["S1"] is True          # normalized next actions agree
    assert lab["S2"] is True
    assert lab["S3"] is True          # em equal, steps equal
    assert lab["harmful"] is False and lab["delayed"] is False
    assert lab["d_em"] == 0 and lab["d_steps"] == 0


def test_s2_looks_three_steps_ahead_and_s1_does_not():
    control = _arm(["search[A]", "search[B]", "finish[x]"], em=1)
    treatment = _arm(["search[A]", "lookup[q]", "search[B]", "finish[x]"], em=1)
    lab = paperb.label_pair(1, control, treatment)
    assert lab["S1"] is False
    assert lab["S2"] is True
    assert lab["delayed"] is True     # 4 steps vs 3
    assert lab["S3"] is False


def test_s2_is_false_beyond_the_window():
    control = _arm(["s[A]", "search[B]", "f[x]"], em=1)
    treatment = _arm(["s[A]", "l[1]", "l[2]", "l[3]", "search[B]"], em=1)
    lab = paperb.label_pair(1, control, treatment)
    assert lab["S2"] is False


def test_harmful_and_s3_track_em():
    control = _arm(["search[A]", "finish[right]"], em=1)
    treatment = _arm(["search[A]", "finish[wrong]"], em=0, f1=0.0)
    lab = paperb.label_pair(1, control, treatment)
    assert lab["harmful"] is True
    assert lab["S3"] is False
    assert lab["d_em"] == -1 and lab["d_f1"] == -1.0


def test_s3_allows_an_improvement():
    control = _arm(["search[A]", "search[B]", "finish[wrong]"], em=0, f1=0.0)
    treatment = _arm(["search[A]", "finish[right]"], em=1)
    lab = paperb.label_pair(1, control, treatment)
    assert lab["S3"] is True and lab["delayed"] is False and lab["harmful"] is False


def test_s1_s2_are_na_when_an_arm_ends_at_step_i():
    control = _arm(["search[A]", "finish[x]"], em=1)
    treatment = _arm(["finish[x]"], em=1)          # ends AT step 1
    lab = paperb.label_pair(1, control, treatment)
    assert lab["S1"] is None and lab["S2"] is None
    assert "ends at step i" in lab["S1_na_reason"]
    assert lab["S3"] is True                        # still computable


# ---------------------------------------------------------------------------
# Nondeterminism control
# ---------------------------------------------------------------------------

def _logged():
    return {"n_steps": 2, "em": 1, "steps": [
        {"i": 1, "action": "search[A]", "obs": "page A"},
        {"i": 2, "action": "finish[x]", "obs": "Episode finished, reward = 1\n"}]}


def test_control_reproducing_the_log_is_deterministic():
    control = {"n_steps": 2, "em": 1, "steps": [
        {"i": 1, "action": "search[A]", "obs": "page A"},
        {"i": 2, "action": "finish[x]", "obs": "Episode finished, reward = 1\n"}]}
    got = paperb.control_reproduces_log(_logged(), control)
    assert got["deterministic"] is True and got["first_divergent_step"] is None


def test_control_diverging_is_flagged_with_its_first_step():
    control = {"n_steps": 2, "em": 0, "steps": [
        {"i": 1, "action": "search[A]", "obs": "page A"},
        {"i": 2, "action": "finish[y]", "obs": "Episode finished, reward = 0\n"}]}
    got = paperb.control_reproduces_log(_logged(), control)
    assert got["deterministic"] is False
    assert got["first_divergent_step"] == 2


def test_control_of_different_length_is_flagged():
    control = {"n_steps": 1, "em": 1, "steps": [
        {"i": 1, "action": "search[A]", "obs": "page A"}]}
    got = paperb.control_reproduces_log(_logged(), control)
    assert got["deterministic"] is False


# ---------------------------------------------------------------------------
# Runner plumbing: who proposes the candidates, and token capture
# ---------------------------------------------------------------------------

def test_speculator_proposes_the_candidates_when_configured():
    runner, actor = build_runner(
        ["Thought 1: mine.\nAction 1: search[Scott Derrickson]",
         "Thought 2: done.\nAction 2: finish[sinister]"])
    spec = ScriptedActor(["Thought 1: guess.\nAction 1: Search[Ed Wood], "
                          "Lookup[wood], Finish[ed wood]"])
    runner.spec_llm = spec

    old_from, old_cap = constants.spec_actions_from, constants.capture_spec_logprobs
    constants.spec_actions_from = "speculator"
    constants.capture_spec_logprobs = True
    try:
        info = runner.webthink(IDX, prompt=PROMPT, to_print=False, n=3, simulate=True)
        # Built under the same config, as the collector does: the record stamps
        # constants, so building it after the restore would stamp the default.
        record = paperb.episode_record(runner, IDX, info)
    finally:
        constants.spec_actions_from = old_from
        constants.capture_spec_logprobs = old_cap

    sim = runner.env.sim_trajectory_dict
    assert sim["actions"][0] == ["Search[Ed Wood]", "Lookup[wood]", "Finish[ed wood]"]
    assert spec.prompts, "the speculator was never asked for candidates"
    assert runner.spec_token_records[0]["i"] == 1
    assert runner.spec_token_records[0]["tokens"], "no token logprobs captured"
    assert info.get("em") in (0, 1)

    assert record["spec_actions_from"] == "speculator"
    assert record["steps"][0]["spec_tokens"], "tokens did not reach the record"
    assert record["question"], "question not recovered from the dataset"


def test_actor_proposes_the_candidates_by_default():
    runner, actor = build_runner(
        ["Thought 1: mine.\nAction 1: search[Scott Derrickson]",
         "Thought 1: guesses.\nAction 1: Search[Ed Wood], Lookup[wood], Finish[w]",
         "Thought 2: done.\nAction 2: finish[sinister]"])
    spec = ScriptedActor(["Thought 1: unused.\nAction 1: search[Never]"])
    runner.spec_llm = spec
    assert constants.spec_actions_from == "actor"
    runner.webthink(IDX, prompt=PROMPT, to_print=False, n=3, simulate=True)
    assert not spec.prompts, "upstream path must not call the speculator for actions"
    assert runner.spec_token_records == []


# ---------------------------------------------------------------------------
# standalone runner (pytest is not guaranteed on this node)
# ---------------------------------------------------------------------------

def _main():
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = []
    for name, fn in tests:
        try:
            fn()
            print(f"  ok   {name}")
        except Exception as exc:                        # noqa: BLE001
            import traceback
            failed.append(name)
            print(f"  FAIL {name}: {exc.__class__.__name__}: {exc}")
            traceback.print_exc()
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_main())

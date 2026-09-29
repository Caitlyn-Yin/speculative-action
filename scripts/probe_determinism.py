#!/usr/bin/env python
"""Is the actor server deterministic on a repeated identical request?

`diag_nondeterminism.py` showed that two replays of the same prefix, in the same
process, can pick different actions at the same step, while the replay loop and
the collection loop agree with each other -- i.e. the pilot's 13-23% control
divergence is the *server*, not a prompt mismatch.  This script measures that
directly and cheaply, with no loop in the way:

  * build the actor prompt for step k of a logged question by re-executing the
    logged prefix 1..k-1 (a forced prefix makes no actor call, so the prompt is a
    pure function of the log and the corpus);
  * issue the identical chat completion R times, back to back;
  * report how many distinct replies and distinct parsed actions came back.

Run it against a server started with and without `--enable-prefix-caching` /
`--enforce-eager` to attribute the nondeterminism.

Usage (from hotpotqa/):
    RETRIEVAL_BACKEND=title_exact $PIPELINE_PY ../scripts/probe_determinism.py \
        --collected /tmp/specmem/paperb/pilot25_title_exact \
        --idx 5619 --step 4 --repeats 10
"""

import argparse
import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
HOTPOTQA = os.path.join(REPO, "hotpotqa")
os.chdir(HOTPOTQA)
sys.path.insert(0, HOTPOTQA)

from src import constants              # noqa: E402
from src import paperb                 # noqa: E402
from src.prompts import PromptTemplates  # noqa: E402
from src.runner import HotPotQARun     # noqa: E402


class PromptCapture:
    """Records the prompt of the FIRST actor call and aborts the loop."""

    class Stop(Exception):
        pass

    def __init__(self, inner):
        self.inner = inner
        self.prompt = None
        self.model_name = inner.model_name

    def call(self, prompt, stop=None):
        self.prompt = prompt
        raise PromptCapture.Stop()

    def _strip_thinking(self, text):
        return self.inner._strip_thinking(text)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--collected", required=True)
    ap.add_argument("--idx", type=int, nargs="+", required=True)
    ap.add_argument("--step", type=int, default=None,
                    help="step whose prompt is probed (default: the step after "
                         "the logged prefix, i.e. the last logged step)")
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--retrieval-backend", default=None,
                    choices=["live", "title_exact", "bm25"])
    ap.add_argument("--label", default="default",
                    help="free-text label for the server configuration probed")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    cfg = {}
    cm = os.path.join(args.collected, "collect_manifest.json")
    if os.path.exists(cm):
        with open(cm, encoding="utf-8") as fh:
            cfg = json.load(fh).get("config", {})
    constants.retrieval_backend = (args.retrieval_backend
                                   or cfg.get("retrieval_backend")
                                   or constants.retrieval_backend)
    if cfg.get("actor_model"):
        constants.actor_model_name = cfg["actor_model"]

    records = {r["idx"]: r for r in (
        json.loads(l) for l in open(
            os.path.join(args.collected, "trajectories.jsonl"), encoding="utf-8")
        if l.strip())}

    prompt = paperb.build_react_prompt()
    runner = HotPotQARun(model_name=constants.actor_model_name,
                         guess_model_name=constants.spec_model_name,
                         to_print_output=False)
    real_llm = runner.llm
    runner.base_traj_path = "/tmp/specmem/paperb/probe_logs"

    report = {"label": args.label, "backend": constants.retrieval_backend,
              "actor": constants.actor_model_name, "repeats": args.repeats,
              "questions": {}}

    for idx in args.idx:
        logged = records[idx]
        step = args.step or len(logged["steps"])
        forced = [{"thought": s["thought"], "action": s["action"]}
                  for s in logged["steps"] if s["i"] < step]

        cap = PromptCapture(real_llm)
        runner.llm = cap
        try:
            paperb.replay(runner, idx, prompt, forced)
        except PromptCapture.Stop:
            pass
        finally:
            runner.llm = real_llm
        if cap.prompt is None:
            print(f"idx {idx}: no actor call at step {step} (prefix covers it)")
            continue

        replies = [real_llm.call(cap.prompt, stop=None) for _ in range(args.repeats)]
        actions = []
        for text in replies:
            _t, acts = runner.separate_thought_and_actions(step, text)
            actions.append(acts[0] if acts else None)
        counts = collections.Counter(actions)
        report["questions"][idx] = {
            "step": step,
            "prompt_chars": len(cap.prompt),
            "distinct_replies": len(set(replies)),
            "distinct_actions": len(counts),
            "action_counts": dict(counts),
            "logged_action": next((s["action"] for s in logged["steps"]
                                   if s["i"] == step), None),
        }
        q = report["questions"][idx]
        print(f"idx {idx} step {step}: prompt {q['prompt_chars']} chars, "
              f"{args.repeats} identical requests -> "
              f"{q['distinct_replies']} distinct replies, "
              f"{q['distinct_actions']} distinct actions {q['action_counts']} "
              f"(logged: {q['logged_action']})")

    deterministic = all(v["distinct_replies"] == 1
                        for v in report["questions"].values())
    report["deterministic"] = deterministic
    print(f"\n[{args.label}] deterministic on repeated identical requests: "
          f"{deterministic}")
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False)
        print(f"report -> {args.out}")
    return 0 if deterministic else 2


if __name__ == "__main__":
    sys.exit(main())

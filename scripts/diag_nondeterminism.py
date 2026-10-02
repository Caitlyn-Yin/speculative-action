#!/usr/bin/env python
"""Why do control re-runs diverge from the logged trajectory?

The 25-question pilot failed go/no-go (a): 22.6% (title_exact) / 13.6% (bm25) of
controls did not reproduce their logged trajectory, against a 5% bar, and every
divergence was action-level (the actor chose differently), never observation-level.
Two candidate causes, which this script separates:

  (1) **Server nondeterminism** — the same prompt, run twice, yields a different
      greedy action.  Then two independent replays of the same prefix disagree
      with *each other*.
  (2) **A prompt mismatch between the collection loop and the replay loop** —
      then the replays agree with each other and differ from the log
      systematically, and the first differing prompt can be diffed byte for byte.

Procedure, per question: re-run `webthink` (both with and without speculation)
and `paperb.replay` (empty prefix, and prefixes 1..1 / 1..2), recording every
actor prompt.  Then compare the realized action sequences and, on the first
divergence, the prompts character by character.

Usage (from hotpotqa/):
    RETRIEVAL_BACKEND=title_exact $PIPELINE_PY ../scripts/diag_nondeterminism.py \
        --collected $SPEC_RUNS_DIR/pilot25_title_exact --idx 5619 6904
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
HOTPOTQA = os.path.join(REPO, "hotpotqa")
os.chdir(HOTPOTQA)
sys.path.insert(0, HOTPOTQA)

from src import constants              # noqa: E402
from src import durability             # noqa: E402
from src import paperb                 # noqa: E402
from src.runner import HotPotQARun     # noqa: E402


class Recorder:
    """Transparent proxy around an LLMClient that keeps every prompt."""

    def __init__(self, inner):
        self.inner = inner
        self.prompts = []
        self.model_name = inner.model_name

    def call(self, prompt, stop=None):
        self.prompts.append(prompt)
        return self.inner.call(prompt, stop=stop)

    def call_with_logprobs(self, prompt, max_tokens=None, top_logprobs=20, stop=None):
        self.prompts.append(prompt)
        return self.inner.call_with_logprobs(prompt, max_tokens=max_tokens,
                                             top_logprobs=top_logprobs, stop=stop)

    def _strip_thinking(self, text):
        return self.inner._strip_thinking(text)


def actions_of(steps):
    return [s["action"] for s in steps]


def first_diff(a, b):
    for k in range(min(len(a), len(b))):
        if a[k] != b[k]:
            return k
    return None if len(a) == len(b) else min(len(a), len(b))


def char_diff(p, q):
    n = min(len(p), len(q))
    for k in range(n):
        if p[k] != q[k]:
            return {"first_differing_char": k,
                    "len_a": len(p), "len_b": len(q),
                    "context_a": p[max(0, k - 120):k + 120],
                    "context_b": q[max(0, k - 120):k + 120]}
    if len(p) != len(q):
        longer, shorter = (p, q) if len(p) > len(q) else (q, p)
        return {"first_differing_char": n, "len_a": len(p), "len_b": len(q),
                "tail_of_longer": longer[n:n + 240],
                "which_is_longer": "a" if len(p) > len(q) else "b"}
    return None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--collected", required=True)
    ap.add_argument("--idx", type=int, nargs="+", required=True)
    ap.add_argument("--retrieval-backend", default=None,
                    choices=["live", "title_exact", "bm25"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    durability.require_durable_outputs(
        out=args.out, trajectories=constants.run_output_root)

    cm_path = os.path.join(args.collected, "collect_manifest.json")
    cfg = {}
    if os.path.exists(cm_path):
        with open(cm_path, encoding="utf-8") as fh:
            cfg = json.load(fh).get("config", {})
    constants.retrieval_backend = (args.retrieval_backend
                                   or cfg.get("retrieval_backend")
                                   or constants.retrieval_backend)
    if cfg.get("actor_model"):
        constants.actor_model_name = cfg["actor_model"]
    if cfg.get("spec_model"):
        constants.spec_model_name = cfg["spec_model"]

    records = {r["idx"]: r for r in (
        json.loads(l) for l in open(
            os.path.join(args.collected, "trajectories.jsonl"), encoding="utf-8")
        if l.strip())}

    prompt = paperb.build_react_prompt()
    runner = HotPotQARun(model_name=constants.actor_model_name,
                         guess_model_name=constants.spec_model_name,
                         to_print_output=False)
    rec = Recorder(runner.llm)
    runner.llm = rec
    runner.base_traj_path = os.path.join(constants.run_output_root, "diag_logs")

    report = {"backend": constants.retrieval_backend,
              "actor": constants.actor_model_name,
              "spec": constants.spec_model_name,
              "questions": {}}

    for idx in args.idx:
        logged = records[idx]
        log_actions = actions_of(logged["steps"])
        arms = {}

        def run_replay(name, forced):
            rec.prompts = []
            out = paperb.replay(runner, idx, prompt, forced)
            arms[name] = {"actions": actions_of(out["steps"]),
                          "prompts": list(rec.prompts),
                          "n_steps": out["n_steps"], "em": out["em"]}

        run_replay("replay_empty_a", [])
        run_replay("replay_empty_b", [])
        for upto in (1, 2):
            if len(logged["steps"]) > upto:
                run_replay(f"replay_forced_{upto}",
                           [{"thought": s["thought"], "action": s["action"]}
                            for s in logged["steps"] if s["i"] <= upto])

        for name, simulate in (("webthink_plain", False), ("webthink_sim", True)):
            rec.prompts = []
            constants.capture_spec_logprobs = simulate and bool(
                cfg.get("capture_spec_logprobs"))
            constants.spec_actions_from = cfg.get("spec_actions_from",
                                                  constants.spec_actions_from)
            info = runner.webthink(idx, prompt=prompt, to_print=False,
                                   n=constants.n_steps_to_run, simulate=simulate)
            traj = runner.env.normal_trajectory_dict
            arms[name] = {"actions": list(traj["actions"]),
                          "prompts": list(rec.prompts),
                          "n_steps": len(traj["actions"]),
                          "em": int(info.get("em", 0))}

        q = {"logged_actions": log_actions,
             "arms": {k: {"actions": v["actions"], "n_steps": v["n_steps"],
                          "em": v["em"]} for k, v in arms.items()},
             "comparisons": {}}

        def compare(a, b):
            if a not in arms or b not in arms:
                return None
            k = first_diff(arms[a]["actions"], arms[b]["actions"])
            out = {"same": k is None, "first_differing_step": None if k is None else k + 1}
            if k is not None:
                # The prompt that produced step k+1 in each arm. Actor prompts
                # are in order, one per recorded step (plus retries, which this
                # run has none of if the counts match the step counts).
                pa = arms[a]["prompts"]
                pb = arms[b]["prompts"]
                # An arm with a forced prefix makes no actor call for those
                # steps, so align from the end of the shared history instead.
                out["n_prompts"] = [len(pa), len(pb)]
                if pa and pb:
                    out["prompt_diff_at_divergence"] = char_diff(pa[-1], pb[-1])
            return out

        for a, b in (("replay_empty_a", "replay_empty_b"),
                     ("replay_empty_a", "webthink_plain"),
                     ("webthink_plain", "webthink_sim"),
                     ("replay_forced_1", "replay_forced_2"),
                     ("replay_forced_1", "webthink_sim")):
            q["comparisons"][f"{a}_vs_{b}"] = compare(a, b)

        q["logged_vs"] = {
            name: {"same": arms[name]["actions"] == log_actions,
                   "first_differing_step": (
                       None if arms[name]["actions"] == log_actions
                       else (first_diff(log_actions, arms[name]["actions"]) or 0) + 1)}
            for name in arms}
        report["questions"][idx] = q

        print(f"\n=== idx {idx} ===")
        print(f"  logged        {log_actions}")
        for name in arms:
            mark = "==log" if arms[name]["actions"] == log_actions else "  !!  "
            print(f"  {name:18s}{mark} {arms[name]['actions']}")
        for k, v in q["comparisons"].items():
            if v:
                print(f"  {k}: same={v['same']} "
                      f"first_diff_step={v['first_differing_step']}")
                d = v.get("prompt_diff_at_divergence")
                if d:
                    print(f"     prompt chars a={d['len_a']} b={d['len_b']} "
                          f"first_differing_char={d['first_differing_char']}")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False)
        print(f"\nreport -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

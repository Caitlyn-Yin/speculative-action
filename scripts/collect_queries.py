#!/usr/bin/env python
"""Collect the search queries a real agent issues, for the fidelity study.

The fidelity measurement in docs/LOCAL_WIKI.md needs 100 *agent-issued*
queries, not just gold titles: the interesting failure mode of a frozen corpus
is the malformed, over-specified, wrongly-cased thing a model actually types
("search[Walden Media founding year]"), which is nothing like a gold title.
`hotpotqa/cache/wiki/` did not survive the pod recycle and only 12 unique
queries are recoverable from the committed `trajs/`, so the rest come from a
live run against the real servers.

    source scripts/env.sh
    bash scripts/serve_local.sh both
    $PIPELINE_PY scripts/collect_queries.py --n-questions 40

Writes `hotpotqa/cache/wiki/queries.jsonl` (append-only, deduplicated on read)
with one record per realized `search[...]`, plus the question index it came
from. Speculated actions are NOT collected: they were never executed, so they
are not part of the environment's query distribution.

This is a data-collection run, not an experiment: no speculation, isolation
irrelevant, retrieval_backend forced to `live` so the queries are drawn from
the same distribution the fidelity comparison scores.
"""

import argparse
import json
import os
import random
import re
import sys

HOTPOTQA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "hotpotqa")
sys.path.insert(0, HOTPOTQA)
os.chdir(HOTPOTQA)

from src import constants                       # noqa: E402
from src import durability                      # noqa: E402
from src.runner import HotPotQARun              # noqa: E402
from src.utils import Utils                     # noqa: E402
from src.prompts import PromptTemplates         # noqa: E402

OUT_DIR = os.path.join(HOTPOTQA, "cache", "wiki")
OUT_FILE = os.path.join(OUT_DIR, "queries.jsonl")
SEARCH_RE = re.compile(r"[Ss]earch\[([^\]]+)\]")


def existing(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def from_committed_trajs():
    """Queries recoverable from the trajectories already in the repo."""
    out = []
    traj_dir = os.path.join(HOTPOTQA, "trajs")
    if not os.path.isdir(traj_dir):
        return out
    for name in sorted(os.listdir(traj_dir)):
        if not name.endswith(".json"):
            continue
        with open(os.path.join(traj_dir, name), encoding="utf-8") as fh:
            try:
                episodes = json.load(fh)
            except json.JSONDecodeError:
                continue
        if isinstance(episodes, dict):
            episodes = [episodes]
        for ep in episodes:
            for action in ep.get("actions", []):
                m = SEARCH_RE.fullmatch(str(action).strip())
                if m:
                    out.append({"query": m.group(1),
                                "question_idx": ep.get("question_idx"),
                                "source": "trajs/" + name})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-questions", type=int, default=40)
    ap.add_argument("--target-unique", type=int, default=100)
    ap.add_argument("--seed", type=int, default=constants.random_seed)
    ap.add_argument("--n-steps", type=int, default=constants.n_steps_to_run)
    args = ap.parse_args()

    durability.require_durable_outputs(
        queries=OUT_DIR, trajectories=constants.run_output_root)

    os.makedirs(OUT_DIR, exist_ok=True)
    constants.retrieval_backend = "live"
    os.environ.setdefault("WIKI_CACHE", "1")     # pin the live responses we touch

    records = existing(OUT_FILE)
    seen = {r["query"] for r in records}
    for rec in from_committed_trajs():
        if rec["query"] not in seen:
            seen.add(rec["query"])
            records.append(rec)
    print(f"{len(records)} queries before the live run "
          f"({len(seen)} unique)", flush=True)

    if len(seen) >= args.target_unique:
        print("target already met; no live run needed")
    else:
        examples = Utils.read_json(
            os.path.join(constants.prompts_folder, constants.prompt_file)
        ).get("webthink_simple6")
        prompt = Utils.join_prompt(PromptTemplates.REACT_INSTRUCTION, examples,
                                   PromptTemplates.PROMPT_INSTRUCTION)
        run = HotPotQARun(model_name=constants.actor_model_name,
                          guess_model_name=constants.spec_model_name,
                          to_print_output=False)
        rng = random.Random(args.seed)
        idxs = list(range(len(run.env.data)))
        rng.shuffle(idxs)

        for k, idx in enumerate(idxs[:args.n_questions]):
            run.current_index = idx
            try:
                info = run.webthink(idx=idx, prompt=prompt, to_print=False,
                                    n=args.n_steps, simulate=False)
            except Exception as exc:                # a data-collection run
                print(f"  idx {idx}: {type(exc).__name__}: {exc}", flush=True)
                continue
            found = 0
            for action in run.env.normal_trajectory_dict.get("actions", []):
                m = SEARCH_RE.fullmatch(str(action).strip())
                if not m:
                    continue
                q = m.group(1)
                found += 1
                if q in seen:
                    continue
                seen.add(q)
                records.append({"query": q, "question_idx": idx,
                                "source": "live_run"})
            print(f"  [{k+1}/{args.n_questions}] idx {idx}: {found} searches, "
                  f"{len(seen)} unique so far", flush=True)
            if len(seen) >= args.target_unique:
                break

    with open(OUT_FILE, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(records)} records ({len(seen)} unique) to {OUT_FILE}")


if __name__ == "__main__":
    main()

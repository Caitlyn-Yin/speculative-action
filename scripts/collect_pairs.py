#!/usr/bin/env python
"""Paper B stage 1 — collect trajectories and extract the speculation pairs.

One pass of ``runner.webthink(simulate=True)`` per question, exactly as the
isolation gate runs it, over the first ``--n-questions`` of the seeded shuffle
(the same order ``runner.run()`` computes, so an index here means the same
question everywhere in this project).  Writes:

    <out>/trajectories.jsonl   one record per question (realized steps, the k
                               speculated actions per step, the speculator's
                               token stream, the imagined sim_obs for provenance)
    <out>/pairs_raw.jsonl      one row per in-population (step i, spec j) pair,
                               gates.Pair-shaped but with no spec_obs and no
                               labels -- those come from replay_audit.py
    <out>/collect_manifest.json

The population is the prereg's: ``lower(spec_j) != lower(real_i)``, i.e. every
pair Speculative Actions' exact match rejects.

Usage (from hotpotqa/, servers up per scripts/serve_local.sh):

    RETRIEVAL_BACKEND=title_exact SPEC_ACTIONS_FROM=speculator \
    CAPTURE_SPEC_LOGPROBS=1 $PIPELINE_PY ../scripts/collect_pairs.py \
        --n-questions 25 --out /tmp/specmem/paperb/title_exact
"""

import argparse
import json
import os
import random
import subprocess
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
HOTPOTQA = os.path.join(REPO, "hotpotqa")
os.chdir(HOTPOTQA)
sys.path.insert(0, HOTPOTQA)

from src import constants              # noqa: E402
from src import paperb                 # noqa: E402
from src.runner import HotPotQARun     # noqa: E402


def pilot_idxs(n):
    """First n of the seeded shuffle — runner.run()'s order (see run_invariant)."""
    idxs = list(range(constants.num))
    random.Random(constants.random_seed).shuffle(idxs)
    return idxs[:n]


def git_version():
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                             capture_output=True, text=True, timeout=10).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO,
                               capture_output=True, text=True, timeout=10).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-questions", type=int, default=25)
    ap.add_argument("--out", required=True)
    ap.add_argument("--retrieval-backend", default=None,
                    choices=["live", "title_exact", "bm25"])
    ap.add_argument("--spec-model", default=None,
                    help="speculator model (prereg: Qwen/Qwen3-4B)")
    ap.add_argument("--actor-model", default=None)
    ap.add_argument("--spec-actions-from", default=None,
                    choices=["actor", "speculator"])
    ap.add_argument("--no-logprobs", action="store_true",
                    help="skip the speculator token capture (criterion 9 goes NA)")
    ap.add_argument("--run", default=None, help="run label for pair_ids")
    args = ap.parse_args(argv)

    if args.retrieval_backend:
        constants.retrieval_backend = args.retrieval_backend
    if args.spec_model:
        constants.spec_model_name = args.spec_model
    if args.actor_model:
        constants.actor_model_name = args.actor_model
    if args.spec_actions_from:
        constants.spec_actions_from = args.spec_actions_from
    constants.capture_spec_logprobs = not args.no_logprobs

    run_label = args.run or f"pilot-{constants.retrieval_backend}"
    os.makedirs(args.out, exist_ok=True)
    traj_path = os.path.join(args.out, "trajectories.jsonl")
    pairs_path = os.path.join(args.out, "pairs_raw.jsonl")

    idxs = pilot_idxs(args.n_questions)
    prompt = paperb.build_react_prompt()

    print(f"run={run_label} backend={constants.retrieval_backend} "
          f"actor={constants.actor_model_name} spec={constants.spec_model_name} "
          f"spec_actions_from={constants.spec_actions_from} "
          f"k={constants.guess_num_actions} step_cap={constants.n_steps_to_run} "
          f"logprobs={constants.capture_spec_logprobs}")
    print(f"questions: {idxs}")

    runner = HotPotQARun(
        model_name=constants.actor_model_name,
        guess_model_name=constants.spec_model_name,
        to_print_output=False,
    )
    runner.base_traj_path = os.path.join(args.out, "logs")

    n_pairs, errors, t0 = 0, [], time.time()
    with open(traj_path, "w", encoding="utf-8") as tf, \
            open(pairs_path, "w", encoding="utf-8") as pf:
        for q, idx in enumerate(idxs, 1):
            runner.current_index = idx
            started = time.time()
            try:
                info = runner.webthink(idx, prompt=prompt, to_print=False,
                                       n=constants.n_steps_to_run, simulate=True)
            except Exception as exc:
                errors.append({"idx": idx, "error": f"{type(exc).__name__}: {exc}",
                               "traceback": traceback.format_exc()})
                print(f"  [{q}/{len(idxs)}] idx={idx} ERROR {type(exc).__name__}: {exc}",
                      flush=True)
                continue

            record = paperb.episode_record(runner, idx, info)
            record["wall_s"] = round(time.time() - started, 2)
            tf.write(json.dumps(record, ensure_ascii=False) + "\n")
            tf.flush()

            pairs = paperb.pairs_from_episode(record, run_label)
            for p in pairs:
                pf.write(json.dumps(p, ensure_ascii=False) + "\n")
            pf.flush()
            n_pairs += len(pairs)

            n_spec = sum(len(s["spec_actions"]) for s in record["steps"])
            print(f"  [{q}/{len(idxs)}] idx={idx} steps={record['n_steps']} "
                  f"em={record['em']} spec={n_spec} pairs={len(pairs)} "
                  f"{record['wall_s']}s", flush=True)

    manifest = {
        "run": run_label,
        "questions": idxs,
        "n_questions": len(idxs),
        "n_pairs": n_pairs,
        "errors": errors,
        "wall_s": round(time.time() - t0, 1),
        "code_version": git_version(),
        "config": {
            "actor_model": constants.actor_model_name,
            "spec_model": constants.spec_model_name,
            "spec_actions_from": constants.spec_actions_from,
            "capture_spec_logprobs": constants.capture_spec_logprobs,
            "retrieval_backend": constants.retrieval_backend,
            "k": constants.guess_num_actions,
            "n_steps_to_run": constants.n_steps_to_run,
            "temperature": constants.temperature,
            "guess_temperature": constants.guess_temperature,
            "top_p": constants.top_p,
            "guess_top_p": constants.guess_top_p,
            "random_seed": constants.random_seed,
            "local_seed": constants.local_seed,
            "isolate_speculation": constants.isolate_speculation,
            "max_spec_action_tokens": constants.max_spec_action_tokens,
        },
        "outputs": {"trajectories": traj_path, "pairs_raw": pairs_path},
    }
    man_path = os.path.join(args.out, "collect_manifest.json")
    with open(man_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)

    print(f"\n{len(idxs) - len(errors)}/{len(idxs)} questions, {n_pairs} pairs")
    print(f"trajectories -> {traj_path}")
    print(f"pairs_raw    -> {pairs_path}")
    print(f"manifest     -> {man_path}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())

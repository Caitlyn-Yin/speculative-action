#!/usr/bin/env python
"""Paper B stage 2 — the re-execution audit that produces the replay labels.

For every in-population pair ``p = (run, question, step i, spec j)`` from
``collect_pairs.py``:

* **control** — re-execute the logged actions ``1..i``, then the actor continues
  to the end.  One control per (question, step i), shared by that step's pairs
  and cached on disk.
* **treatment** — re-execute ``1..i-1``, execute ``spec_j`` (with the logged
  thought of step i), record its **actual** observation, then the actor
  continues.  The speculator's imagined ``sim_obs`` is never used.

Labels S1/S2/S3/harmful/delayed and the deltas are computed by
``src.paperb.label_pair``; a pair whose control does not reproduce the logged
trajectory is marked ``nondeterministic`` and excluded from the primary
analysis, with the rate reported (prereg, Arms).

Both the control cache and the output are append-only and keyed, so the run is
resumable after a pod restart: re-invoking with the same ``--out`` skips every
pair already written.

Writes:
    <out>/controls.jsonl   one record per (idx, step i) control re-execution
    <out>/pairs.jsonl      gates.Pair rows with spec_obs + labels (the file
                           score_criteria.py consumes)
    <out>/audit_summary.json  counts, nondeterminism rate, label rates,
                           go/no-go table

Usage (from hotpotqa/, servers up):

    RETRIEVAL_BACKEND=title_exact $PIPELINE_PY ../scripts/replay_audit.py \
        --collected /tmp/specmem/paperb/title_exact \
        --out /tmp/specmem/paperb/title_exact
"""

import argparse
import json
import os
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
from src import gates                  # noqa: E402
from src import paperb                 # noqa: E402
from src.runner import HotPotQARun     # noqa: E402


def read_jsonl(path):
    out = []
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                # A killed run can leave a partial trailing line; skip it, the
                # same rule backends.AppendOnlyCache uses.
                continue
    return out


def git_version():
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                             capture_output=True, text=True, timeout=10).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO,
                               capture_output=True, text=True, timeout=10).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def forced_prefix(record, upto):
    """Logged (thought, action) for steps 1..upto."""
    return [{"thought": s["thought"], "action": s["action"]}
            for s in record["steps"] if s["i"] <= upto]


def same_tool(a, b):
    return gates.parse_action(a)[0] == gates.parse_action(b)[0]


def summarize(rows, manifest_cfg, n_questions, fidelity):
    """Counts, rates and the prereg's go/no-go table."""
    n = len(rows)
    det = [r for r in rows if not r["meta"]["nondeterministic"]]
    nondet_rate = (n - len(det)) / n if n else 0.0

    same_tool_rows = [r for r in det
                      if same_tool(r["spec_action"], r["real_action"])]
    search_rows = [r for r in same_tool_rows
                   if gates.parse_action(r["real_action"])[0] == "search"]

    def rate(rs, key):
        vals = [r["labels"][key] for r in rs if r["labels"].get(key) is not None]
        return (sum(1 for v in vals if v) / len(vals) if vals else None), len(vals)

    s3_rate, s3_n = rate(det, "S3")
    s1_rate, s1_n = rate(same_tool_rows, "S1")
    s1_search_rate, s1_search_n = rate(search_rows, "S1")
    s2_rate, s2_n = rate(same_tool_rows, "S2")
    harm_rate, harm_n = rate(det, "harmful")
    del_rate, del_n = rate(det, "delayed")
    s3_same_tool, s3_st_n = rate(same_tool_rows, "S3")

    gates_tbl = {
        "a_nondeterministic_rate_below_5pct": {
            "value": round(nondet_rate, 4), "bar": "< 0.05",
            "pass": nondet_rate < 0.05},
        "b_at_least_50_non_em_same_tool_pairs": {
            "value": len(same_tool_rows), "bar": ">= 50",
            "pass": len(same_tool_rows) >= 50},
        "c_title_exact_fidelity_at_least_90pct": {
            "value": fidelity, "bar": ">= 0.90",
            "pass": (fidelity is not None and fidelity >= 0.90),
            "source": "docs/LOCAL_WIKI.md (300-query fidelity study); not "
                      "recomputed here"},
    }
    stop_rule = {
        "rule": "prereg: if S3 rate < 2% in both backends at pilot, stop and "
                "investigate as a probable bug",
        "s3_rate_this_backend": None if s3_rate is None else round(s3_rate, 4),
        "below_2pct": (s3_rate is not None and s3_rate < 0.02),
    }
    return {
        "n_pairs": n,
        "n_questions": n_questions,
        "n_deterministic_pairs": len(det),
        "nondeterministic_rate": round(nondet_rate, 4),
        "n_non_em_same_tool_pairs": len(same_tool_rows),
        "n_non_em_same_tool_search_pairs": len(search_rows),
        "rates_over_deterministic_pairs": {
            "S3": {"rate": None if s3_rate is None else round(s3_rate, 4), "n": s3_n},
            "S3_same_tool": {"rate": None if s3_same_tool is None else round(s3_same_tool, 4),
                             "n": s3_st_n},
            "S1_same_tool": {"rate": None if s1_rate is None else round(s1_rate, 4), "n": s1_n},
            "S1_same_tool_search": {"rate": None if s1_search_rate is None
                                    else round(s1_search_rate, 4), "n": s1_search_n},
            "S2_same_tool": {"rate": None if s2_rate is None else round(s2_rate, 4), "n": s2_n},
            "harmful": {"rate": None if harm_rate is None else round(harm_rate, 4), "n": harm_n},
            "delayed": {"rate": None if del_rate is None else round(del_rate, 4), "n": del_n},
        },
        "go_no_go": gates_tbl,
        "stop_rule": stop_rule,
        "citable": {"note": "prereg: cells with n < 100 pairs are reported but "
                            "NOT citable", "n_pairs": n, "citable": n >= 100},
        "config": manifest_cfg,
        "code_version": git_version(),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--collected", required=True,
                    help="directory written by collect_pairs.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--retrieval-backend", default=None,
                    choices=["live", "title_exact", "bm25"])
    ap.add_argument("--limit", type=int, default=None,
                    help="stop after N pairs (smoke runs)")
    ap.add_argument("--fidelity", type=float, default=0.94,
                    help="title_exact hit/miss agreement vs live, from "
                         "docs/LOCAL_WIKI.md; go/no-go (c) reads this")
    args = ap.parse_args(argv)

    records = {r["idx"]: r for r in read_jsonl(
        os.path.join(args.collected, "trajectories.jsonl"))}
    pairs = read_jsonl(os.path.join(args.collected, "pairs_raw.jsonl"))
    if not pairs:
        sys.exit(f"no pairs in {args.collected}/pairs_raw.jsonl")

    collect_manifest = {}
    cm_path = os.path.join(args.collected, "collect_manifest.json")
    if os.path.exists(cm_path):
        with open(cm_path, encoding="utf-8") as fh:
            collect_manifest = json.load(fh)
    cfg = collect_manifest.get("config", {})

    # The audit must re-execute under the same retrieval backend as collection.
    backend = (args.retrieval_backend or cfg.get("retrieval_backend")
               or constants.retrieval_backend)
    constants.retrieval_backend = backend
    if cfg.get("retrieval_backend") and cfg["retrieval_backend"] != backend:
        sys.exit(f"backend mismatch: collected under {cfg['retrieval_backend']!r}, "
                 f"audit asked for {backend!r}")
    for key, attr in (("actor_model", "actor_model_name"),
                      ("spec_model", "spec_model_name"),
                      ("n_steps_to_run", "n_steps_to_run")):
        if cfg.get(key) is not None:
            setattr(constants, attr, cfg[key])

    os.makedirs(args.out, exist_ok=True)
    controls_path = os.path.join(args.out, "controls.jsonl")
    pairs_path = os.path.join(args.out, "pairs.jsonl")

    controls = {(c["idx"], c["step_i"]): c for c in read_jsonl(controls_path)}
    done_rows = read_jsonl(pairs_path)
    done_ids = {r["pair_id"] for r in done_rows}
    todo = [p for p in pairs if p["pair_id"] not in done_ids]
    if args.limit:
        todo = todo[:args.limit]
    print(f"backend={backend} pairs={len(pairs)} already_done={len(done_ids)} "
          f"todo={len(todo)} controls_cached={len(controls)}")

    prompt = paperb.build_react_prompt()
    runner = HotPotQARun(
        model_name=constants.actor_model_name,
        guess_model_name=constants.spec_model_name,
        to_print_output=False,
    )
    runner.base_traj_path = os.path.join(args.out, "logs")

    t0, errors = time.time(), []
    cf = open(controls_path, "a", encoding="utf-8")
    pf = open(pairs_path, "a", encoding="utf-8")
    try:
        for k, pair in enumerate(todo, 1):
            idx = pair["meta"]["idx"]
            step_i = pair["step_i"]
            record = records.get(idx)
            if record is None:
                errors.append({"pair_id": pair["pair_id"],
                               "error": "no trajectory record for this question"})
                continue

            try:
                key = (idx, step_i)
                if key not in controls:
                    control = paperb.replay(
                        runner, idx, prompt, forced_prefix(record, step_i))
                    control["idx"], control["step_i"] = idx, step_i
                    control["determinism"] = paperb.control_reproduces_log(
                        record, control)
                    controls[key] = control
                    cf.write(json.dumps(control, ensure_ascii=False) + "\n")
                    cf.flush()
                control = controls[key]

                forced = forced_prefix(record, step_i - 1)
                logged_thought = next(
                    (s["thought"] for s in record["steps"] if s["i"] == step_i), "")
                forced.append({"thought": logged_thought,
                               "action": pair["spec_action"]})
                treatment = paperb.replay(runner, idx, prompt, forced)
            except Exception as exc:
                errors.append({"pair_id": pair["pair_id"],
                               "error": f"{type(exc).__name__}: {exc}",
                               "traceback": traceback.format_exc()})
                print(f"  [{k}/{len(todo)}] {pair['pair_id']} ERROR "
                      f"{type(exc).__name__}: {exc}", flush=True)
                continue

            spec_step = next((s for s in treatment["steps"] if s["i"] == step_i), None)
            row = dict(pair)
            row["spec_obs"] = None if spec_step is None else spec_step["obs"]
            row["spec_obs_source"] = "executed" if spec_step is not None else "unavailable"
            row["labels"] = paperb.label_pair(step_i, control, treatment)
            meta = dict(row.get("meta") or {})
            meta.update({
                "nondeterministic": not control["determinism"]["deterministic"],
                "control_determinism": control["determinism"],
                "control": {kk: control[kk] for kk in
                            ("n_steps", "em", "f1", "answer", "terminated")},
                "treatment": {kk: treatment[kk] for kk in
                              ("n_steps", "em", "f1", "answer", "terminated")},
                "treatment_steps": [
                    {"i": s["i"], "action": s["action"], "source": s["source"]}
                    for s in treatment["steps"]],
            })
            row["meta"] = meta
            pf.write(json.dumps(row, ensure_ascii=False) + "\n")
            pf.flush()

            lab = row["labels"]
            print(f"  [{k}/{len(todo)}] {pair['pair_id']} S1={lab['S1']} "
                  f"S2={lab['S2']} S3={lab['S3']} d_em={lab['d_em']} "
                  f"d_steps={lab['d_steps']} "
                  f"nondet={meta['nondeterministic']}", flush=True)
    finally:
        cf.close()
        pf.close()

    rows = read_jsonl(pairs_path)
    summary = summarize(rows, cfg or {
        "retrieval_backend": backend,
        "actor_model": constants.actor_model_name,
        "spec_model": constants.spec_model_name,
    }, n_questions=len(records), fidelity=args.fidelity)
    summary["errors"] = errors
    summary["wall_s"] = round(time.time() - t0, 1)
    sum_path = os.path.join(args.out, "audit_summary.json")
    with open(sum_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    print(f"\npairs   -> {pairs_path} ({len(rows)} rows)")
    print(f"summary -> {sum_path}")
    print(json.dumps({k: summary[k] for k in
                      ("n_pairs", "nondeterministic_rate",
                       "n_non_em_same_tool_pairs", "rates_over_deterministic_pairs",
                       "go_no_go", "stop_rule")}, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())

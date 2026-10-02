#!/usr/bin/env python
"""Paper B stage 3 — statistics over the replay labels and the criterion scores.

Implements the Statistics and Hypotheses sections of ``docs/PREREG_PAPER_B.md``:
cluster bootstrap over **questions**, 10k resamples, 95% percentile CIs, and the
four pre-registered hypotheses with their falsifiers.  Backend contrasts are
**paired** on the question set (both backends run the same 25/100 questions), so
a resample draws questions and takes both backends' pairs for them.

Nondeterministic pairs (control did not reproduce the logged trajectory) are
excluded from every number here and their rate is reported; cells with n < 100
pairs are marked not citable, per the prereg.

Inputs, per backend: the audit's ``pairs.jsonl`` and, when available, the
battery's ``criteria_scores.jsonl`` (from ``score_criteria.py``).

Thresholds for H3/H4: the prereg fixes no operating point, so the "primary"
threshold per scored criterion is **ours** and is listed in
``PRIMARY_THRESHOLD`` below (published value where a source publishes one --
SpecBox 0.80, SPORK 0.90 -- and the grid's natural midpoint or zero otherwise).
Every grid point is reported alongside, so nothing rests on that choice.

Usage:

    $PIPELINE_PY scripts/analyze_paperb.py \
        --backend title_exact $SPEC_RUNS_DIR/pilot25_title_exact \
        --backend bm25 $SPEC_RUNS_DIR/pilot25_bm25 \
        --out $SPEC_RUNS_DIR/pilot25_meta/analysis
"""

import argparse
import json
import math
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
HOTPOTQA = os.path.join(REPO, "hotpotqa")
sys.path.insert(0, HOTPOTQA)

from src import durability  # noqa: E402
from src import gates  # noqa: E402

N_BOOT = 10000
SEED = 0

#: Ours (see the module docstring).  ``None`` = the criterion is binary.
PRIMARY_THRESHOLD = {
    "exact_sa": None,
    "normalized": None,
    "battery": None,
    "edit_distance_dsp": 0.3,     # distance: accept when <= threshold
    "embed_call": 0.80,           # SpecBox tau_c
    "judge_v2_verbal": 0.50,
    "judge_v2_logprob": 0.0,
    "dualspec_critic": 0.0,
    "spec_confidence": 0.90,      # SPORK theta
    "obs_equal": None,
    "spechop_rules": None,
    "embed_obs": 0.80,
    "sufficient_context": None,
}
#: The one criterion whose score is a distance, so acceptance is <=.
LOWER_IS_ACCEPT = {"edit_distance_dsp"}


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

def read_jsonl(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def load_backend(d):
    pairs = read_jsonl(os.path.join(d, "pairs.jsonl"))
    scores = {r["pair_id"]: r for r in read_jsonl(
        os.path.join(d, "criteria_scores.jsonl"))}
    for p in pairs:
        p["_scores"] = scores.get(p["pair_id"], {}).get("criteria", {})
        p["_scores_alt"] = ((scores.get(p["pair_id"], {}).get("judge_alt") or {})
                            .get("criteria", {}) or {})
    summary_path = os.path.join(d, "audit_summary.json")
    summary = {}
    if os.path.exists(summary_path):
        with open(summary_path, encoding="utf-8") as fh:
            summary = json.load(fh)
    return pairs, summary, len(scores)


# ---------------------------------------------------------------------------
# Cluster bootstrap
# ---------------------------------------------------------------------------

def cluster_bootstrap(clusters, stat_fn, n_boot=N_BOOT, seed=SEED):
    """Percentile CI for `stat_fn` under resampling of whole clusters.

    `clusters` maps a cluster key (a question) to whatever `stat_fn` consumes a
    list of.  Returns ``{"point", "lo", "hi", "n_clusters", "n_valid_resamples"}``;
    ``point``/``lo``/``hi`` are None when the statistic is undefined (e.g. an
    empty denominator).
    """
    keys = sorted(clusters)
    items_all = [x for k in keys for x in clusters[k]]
    point = stat_fn(items_all)
    if point is None or not keys:
        return {"point": None, "lo": None, "hi": None,
                "n_clusters": len(keys), "n_valid_resamples": 0}
    rng = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        draw = [clusters[keys[rng.randrange(len(keys))]] for _ in keys]
        items = [x for grp in draw for x in grp]
        v = stat_fn(items)
        if v is not None:
            vals.append(v)
    vals.sort()
    if not vals:
        return {"point": round(point, 6), "lo": None, "hi": None,
                "n_clusters": len(keys), "n_valid_resamples": 0}
    lo = vals[max(0, int(0.025 * len(vals)) - 1)]
    hi = vals[min(len(vals) - 1, int(math.ceil(0.975 * len(vals))) - 1)]
    return {"point": round(point, 6), "lo": round(lo, 6), "hi": round(hi, 6),
            "n_clusters": len(keys), "n_valid_resamples": len(vals)}


def by_question(rows):
    out = {}
    for r in rows:
        out.setdefault(r["meta"]["idx"], []).append(r)
    return out


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def label_rate(key):
    def stat(items):
        vals = [r["labels"].get(key) for r in items]
        vals = [v for v in vals if v is not None]
        return (sum(1 for v in vals if v) / len(vals)) if vals else None
    return stat


def accepts(row, crit, threshold):
    """Did `crit` accept this pair at `threshold`?  None when NA/absent."""
    res = row["_scores"].get(crit)
    if not res or res.get("na_reason"):
        return None
    if threshold is None:
        return None if res.get("binary") is None else bool(res["binary"])
    score = res.get("score")
    if score is None:
        return None
    return (score <= threshold if crit in LOWER_IS_ACCEPT
            else score >= threshold)


def precision_wrt(crit, threshold, label="S3"):
    """P(label | criterion accepts).  None when nothing was accepted."""
    def stat(items):
        num = den = 0
        for r in items:
            a = accepts(r, crit, threshold)
            lab = r["labels"].get(label)
            if a is None or lab is None:
                continue
            if a:
                den += 1
                num += 1 if lab else 0
        return (num / den) if den else None
    return stat


def recall_wrt(crit, threshold, label="S3"):
    def stat(items):
        num = den = 0
        for r in items:
            a = accepts(r, crit, threshold)
            lab = r["labels"].get(label)
            if a is None or lab is None:
                continue
            if lab:
                den += 1
                num += 1 if a else 0
        return (num / den) if den else None
    return stat


def auroc_of(crit, label="S3", alt=False, sign=None):
    """Mann-Whitney AUROC of a criterion's score against a binary label.

    Ties contribute 0.5.  `sign` flips the score when larger means *reject*
    (the DSP distance); it defaults to -1 for those criteria.
    """
    flip = -1.0 if (sign is None and crit in LOWER_IS_ACCEPT) else (sign or 1.0)

    def stat(items):
        pos, neg = [], []
        for r in items:
            res = (r["_scores_alt"] if alt else r["_scores"]).get(crit)
            lab = r["labels"].get(label)
            if not res or res.get("na_reason") or lab is None:
                continue
            s = res.get("score")
            if s is None:
                continue
            (pos if lab else neg).append(flip * s)
        if not pos or not neg:
            return None
        wins = ties = 0
        for a in pos:
            for b in neg:
                if a > b:
                    wins += 1
                elif a == b:
                    ties += 1
        return (wins + 0.5 * ties) / (len(pos) * len(neg))
    return stat


def paired_diff(stat_fn):
    """Same statistic on two backends, differenced; items are (a_rows, b_rows)."""
    def stat(items):
        a = [r for grp in items for r in grp[0]]
        b = [r for grp in items for r in grp[1]]
        va, vb = stat_fn(a), stat_fn(b)
        if va is None or vb is None:
            return None
        return vb - va
    return stat


def paired_auroc_diff(crit_a, crit_b, label="S3"):
    """AUROC(crit_b) - AUROC(crit_a) on the same rows (H4)."""
    fa, fb = auroc_of(crit_a, label), auroc_of(crit_b, label)

    def stat(items):
        va, vb = fa(items), fb(items)
        if va is None or vb is None:
            return None
        return vb - va
    return stat


# ---------------------------------------------------------------------------
# Subpopulations
# ---------------------------------------------------------------------------

def deterministic(rows):
    return [r for r in rows if not r["meta"].get("nondeterministic")]


def same_tool(rows):
    return [r for r in rows
            if gates.parse_action(r["spec_action"])[0]
            == gates.parse_action(r["real_action"])[0]]


def search_only(rows):
    return [r for r in rows
            if gates.parse_action(r["real_action"])[0] == "search"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", nargs=2, action="append", metavar=("NAME", "DIR"),
                    required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    args = ap.parse_args(argv)

    durability.require_durable_outputs(out=args.out)
    durability.warn_if_outside_spec_base(out=args.out)

    data, summaries, scored_counts = {}, {}, {}
    for name, d in args.backend:
        data[name], summaries[name], scored_counts[name] = load_backend(d)
        if not data[name]:
            sys.exit(f"no pairs.jsonl rows for backend {name} in {d}")

    os.makedirs(args.out, exist_ok=True)
    result = {"n_boot": args.n_boot, "backends": {}, "hypotheses": {},
              "scored_pairs": scored_counts}

    # ---------------- per-backend label rates -----------------------------
    for name, rows in data.items():
        det = deterministic(rows)
        st = same_tool(det)
        st_search = search_only(st)
        block = {
            "n_pairs": len(rows),
            "n_deterministic": len(det),
            "nondeterministic_rate": round(1 - len(det) / len(rows), 4),
            "n_same_tool": len(st),
            "n_same_tool_search": len(st_search),
            "citable": len(det) >= 100,
            "labels": {},
            "criteria": {},
        }
        for key in ("S1", "S2", "S3", "harmful", "delayed"):
            block["labels"][key] = {
                "all": cluster_bootstrap(by_question(det), label_rate(key),
                                         args.n_boot),
                "same_tool": cluster_bootstrap(by_question(st), label_rate(key),
                                               args.n_boot),
            }
        block["labels"]["S1"]["same_tool_search"] = cluster_bootstrap(
            by_question(st_search), label_rate("S1"), args.n_boot)

        if scored_counts[name]:
            for crit, thr in PRIMARY_THRESHOLD.items():
                cell = {
                    "primary_threshold": thr,
                    "precision_S3": cluster_bootstrap(
                        by_question(det), precision_wrt(crit, thr), args.n_boot),
                    "recall_S3": cluster_bootstrap(
                        by_question(det), recall_wrt(crit, thr), args.n_boot),
                }
                if thr is not None:
                    cell["auroc_S3"] = cluster_bootstrap(
                        by_question(det), auroc_of(crit), args.n_boot)
                    grid = {}
                    for r in det:
                        res = r["_scores"].get(crit) or {}
                        for g in (res.get("decisions") or {}):
                            grid.setdefault(g, 0)
                    for g in sorted(grid, key=float):
                        gthr = float(g)
                        grid[g] = {
                            "precision_S3": cluster_bootstrap(
                                by_question(det), precision_wrt(crit, gthr),
                                args.n_boot),
                            "recall_S3": cluster_bootstrap(
                                by_question(det), recall_wrt(crit, gthr),
                                args.n_boot),
                        }
                    cell["grid"] = grid
                block["criteria"][crit] = cell
        result["backends"][name] = block

    names = [n for n, _ in args.backend]

    # ---------------- H1 --------------------------------------------------
    h1 = {"claim": "S3 rate among non-EM same-tool pairs has a 95% CI lower "
                   "bound > 0 in at least one backend",
          "falsifier": "upper bound < 2% in both backends",
          "per_backend": {}}
    for name in names:
        st = same_tool(deterministic(data[name]))
        h1["per_backend"][name] = cluster_bootstrap(
            by_question(st), label_rate("S3"), args.n_boot)
    los = [h1["per_backend"][n]["lo"] for n in names
           if h1["per_backend"][n]["lo"] is not None]
    his = [h1["per_backend"][n]["hi"] for n in names
           if h1["per_backend"][n]["hi"] is not None]
    h1["supported"] = any(lo > 0 for lo in los) if los else None
    h1["falsified"] = (bool(his) and all(hi < 0.02 for hi in his))
    result["hypotheses"]["H1"] = h1

    # ---------------- H2 (paired) ----------------------------------------
    if len(names) >= 2:
        a, b = names[0], names[1]
        pa = by_question(search_only(same_tool(deterministic(data[a]))))
        pb = by_question(search_only(same_tool(deterministic(data[b]))))
        joint = {q: [(pa.get(q, []), pb.get(q, []))]
                 for q in sorted(set(pa) | set(pb))}
        h2 = {"claim": f"S1 rate among non-EM same-tool search pairs is higher "
                       f"under {b} than {a}",
              "falsifier": "difference CI includes 0 or is negative",
              "difference": cluster_bootstrap(joint, paired_diff(label_rate("S1")),
                                              args.n_boot),
              "rates": {a: cluster_bootstrap(pa, label_rate("S1"), args.n_boot),
                        b: cluster_bootstrap(pb, label_rate("S1"), args.n_boot)},
              "order": [a, b]}
        d0 = h2["difference"]
        h2["supported"] = (d0["lo"] is not None and d0["lo"] > 0)
        h2["falsified"] = (d0["lo"] is None or d0["lo"] <= 0 <= d0["hi"]
                           or (d0["hi"] is not None and d0["hi"] < 0))
        result["hypotheses"]["H2"] = h2

        # ------------ H3 (paired, per call-level criterion) ---------------
        h3 = {"claim": "for at least one call-level criterion, precision w.r.t. "
                       "S3 differs between backends",
              "falsifier": "every such difference CI includes 0",
              "per_criterion": {}}
        call_level = [c for c in PRIMARY_THRESHOLD
                      if gates.CRITERIA[c].level == "call"]
        any_diff = False
        if scored_counts.get(a) and scored_counts.get(b):
            qa = by_question(deterministic(data[a]))
            qb = by_question(deterministic(data[b]))
            joint_all = {q: [(qa.get(q, []), qb.get(q, []))]
                         for q in sorted(set(qa) | set(qb))}
            for crit in call_level:
                thr = PRIMARY_THRESHOLD[crit]
                cell = cluster_bootstrap(
                    joint_all, paired_diff(precision_wrt(crit, thr)), args.n_boot)
                differs = (cell["lo"] is not None and cell["hi"] is not None
                           and not (cell["lo"] <= 0 <= cell["hi"]))
                cell["differs"] = differs
                any_diff = any_diff or differs
                h3["per_criterion"][crit] = cell
        h3["supported"] = any_diff
        h3["falsified"] = (bool(h3["per_criterion"]) and not any_diff)
        result["hypotheses"]["H3"] = h3

    # ---------------- H4 (pooled over backends, paired on rows) ----------
    h4 = {"claim": "AUROC for S3 of judge_v2_verbal is LOWER than the same "
                   "rubric scored by the Yes/No log-prob margin "
                   "(judge_v2_logprob), Qwen3-8B judge only",
          "falsifier": "difference CI includes 0 or is positive",
          "per_backend": {}, "note": "difference = AUROC(logprob) - AUROC(verbal)"}
    for name in names:
        det = deterministic(data[name])
        if not scored_counts.get(name):
            continue
        h4["per_backend"][name] = {
            "auroc_verbal": cluster_bootstrap(by_question(det),
                                              auroc_of("judge_v2_verbal"),
                                              args.n_boot),
            "auroc_logprob": cluster_bootstrap(by_question(det),
                                               auroc_of("judge_v2_logprob"),
                                               args.n_boot),
            "difference": cluster_bootstrap(
                by_question(det),
                paired_auroc_diff("judge_v2_verbal", "judge_v2_logprob"),
                args.n_boot),
        }
    diffs = [v["difference"] for v in h4["per_backend"].values()]
    h4["supported"] = any(d["lo"] is not None and d["lo"] > 0 for d in diffs)
    h4["falsified"] = (bool(diffs) and all(
        d["lo"] is None or (d["lo"] <= 0 <= d["hi"]) or
        (d["hi"] is not None and d["hi"] < 0) for d in diffs))
    result["hypotheses"]["H4"] = h4

    result["audit_summaries"] = summaries
    out_path = os.path.join(args.out, "analysis.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)
    print(f"analysis -> {out_path}")

    def fmt(c):
        if not c or c["point"] is None:
            return "n/a"
        lo = "n/a" if c["lo"] is None else f"{c['lo']:.3f}"
        hi = "n/a" if c["hi"] is None else f"{c['hi']:.3f}"
        return f"{c['point']:.3f} [{lo}, {hi}]"

    for name in names:
        b = result["backends"][name]
        print(f"\n=== {name} === pairs={b['n_pairs']} det={b['n_deterministic']} "
              f"nondet_rate={b['nondeterministic_rate']} "
              f"same_tool={b['n_same_tool']} citable={b['citable']}")
        for key in ("S1", "S2", "S3", "harmful", "delayed"):
            print(f"  {key:8s} all {fmt(b['labels'][key]['all'])}   "
                  f"same_tool {fmt(b['labels'][key]['same_tool'])}")
    for h in ("H1", "H2", "H3", "H4"):
        if h in result["hypotheses"]:
            hh = result["hypotheses"][h]
            print(f"\n{h}: supported={hh.get('supported')} "
                  f"falsified={hh.get('falsified')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

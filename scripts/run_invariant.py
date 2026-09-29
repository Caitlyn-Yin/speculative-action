#!/usr/bin/env python
"""Online 3-arm speculation-isolation invariant gate (handoff doc section 4.4).

Four arms over the same fixed question indices, one request at a time:

    SEQ_a   simulate=False                                  (reference; warms cache)
    SEQ_b   simulate=False, re-run after SEQ_a               (nondeterminism control)
    ISO     simulate=True,  isolate_speculation=True         (under test)
    UNISO   simulate=True,  isolate_speculation=False        (power check)

COMPARED OBJECT per question is the *realized* trajectory only: the ordered
(real_action, real_obs) pairs, plus n_steps and em. Speculative outputs
(sim_obs, spec_actions) are recorded for diagnosis but are not part of the
criterion.

PASS        every question: ISO == SEQ_a byte-for-byte on every step not marked
            nondeterministic. Zero tolerance.
POWER       UNISO != SEQ_a on >= 1 question, AND ISO sim_obs != real_obs on >= 1
            step (i.e. speculation actually ran). Otherwise INCONCLUSIVE.

Usage (from hotpotqa/):
    WIKI_CACHE=1 python ../scripts/run_invariant.py --out ../docs/INVARIANT_REPORT.md
"""

import argparse
import json
import os
import random
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
HOTPOTQA = os.path.join(REPO, "hotpotqa")
os.chdir(HOTPOTQA)
sys.path.insert(0, HOTPOTQA)

from src import constants              # noqa: E402
from src import environment            # noqa: E402
from src.runner import HotPotQARun     # noqa: E402
from src.utils import Utils            # noqa: E402
from src.prompts import PromptTemplates  # noqa: E402

ARMS = ["SEQ_a", "SEQ_b", "ISO", "UNISO"]
ARM_CFG = {
    #        simulate, isolate_speculation
    "SEQ_a": (False, True),
    "SEQ_b": (False, True),
    "ISO":   (True,  True),
    "UNISO": (True,  False),
}


def pilot_idxs(n):
    """First n of the seeded shuffle -- the same order runner.run() computes.

    The handoff doc says to use the recorded pilot idxs if LOST_WORK_MANIFEST
    has them; it does not record a list. The fallback is corroborated: idx 0 of
    this shuffle is 7107, which is the question named in the manifest's
    "7107/7 withdrawal".
    """
    idxs = list(range(constants.num))
    random.Random(constants.random_seed).shuffle(idxs)
    return idxs[:n]


def build_prompt():
    examples = Utils.read_json(
        os.path.join(constants.prompts_folder, constants.prompt_file)
    ).get("webthink_simple6")
    return Utils.join_prompt(
        PromptTemplates.REACT_INSTRUCTION, examples, PromptTemplates.PROMPT_INSTRUCTION
    )


def run_arm(arm, idxs, prompt, artifacts_dir):
    """Run one arm over every idx; return per-question realized trajectories."""
    simulate, isolate = ARM_CFG[arm]
    constants.isolate_speculation = isolate

    environment.reset_wiki_cache_stats()
    runner = HotPotQARun(
        model_name=constants.actor_model_name,
        guess_model_name=constants.spec_model_name,
        to_print_output=False,
    )
    runner.base_traj_path = os.path.join(artifacts_dir, "logs", arm)

    results = {}
    for idx in idxs:
        runner.current_index = idx
        t0 = time.time()
        record = {"idx": idx, "steps": [], "sim_steps": [], "error": None,
                  "n_steps": None, "em": None}
        try:
            info = runner.webthink(
                idx, prompt=prompt, to_print=False,
                n=constants.n_steps_to_run, simulate=simulate,
            )
            traj = runner.env.normal_trajectory_dict
            for i, (a, o) in enumerate(zip(traj["actions"], traj["observations"]), 1):
                record["steps"].append({"i": i, "action": a, "obs": o})
            record["n_steps"] = len(traj["actions"])
            record["em"] = int(info.get("em", 0))

            if simulate:
                sim = runner.env.sim_trajectory_dict
                for i, (a, o) in enumerate(zip(sim["actions"], sim["observations"]), 1):
                    record["sim_steps"].append({"i": i, "spec_actions": a, "sim_obs": o})
        except Exception as exc:  # recorded, not swallowed
            record["error"] = f"{type(exc).__name__}: {exc}"
            record["traceback"] = traceback.format_exc()
        record["wall_s"] = round(time.time() - t0, 2)
        results[idx] = record
        print(f"  [{arm}] idx={idx} n_steps={record['n_steps']} "
              f"em={record['em']} {record['wall_s']}s"
              + (f" ERROR {record['error']}" if record["error"] else ""))

    cache = dict(environment.WIKI_CACHE_STATS)
    print(f"  [{arm}] cache hits={cache['hits']} misses={cache['misses']}")
    return {"questions": results, "cache": cache}


def first_diff_token(a, b):
    """First differing whitespace token between two strings."""
    ta, tb = (a or "").split(), (b or "").split()
    for i in range(max(len(ta), len(tb))):
        x = ta[i] if i < len(ta) else "<END>"
        y = tb[i] if i < len(tb) else "<END>"
        if x != y:
            return i, x, y
    return None, None, None


def step_key(s):
    return (s["action"], s["obs"])


def compare(ref, other):
    """Return (identical, first_divergent_step_index, per-step diff list)."""
    diffs = []
    n = max(len(ref["steps"]), len(other["steps"]))
    first = None
    for i in range(n):
        r = ref["steps"][i] if i < len(ref["steps"]) else None
        o = other["steps"][i] if i < len(other["steps"]) else None
        if r is None or o is None or step_key(r) != step_key(o):
            if first is None:
                first = i + 1
            diffs.append({"step": i + 1, "ref": r, "other": o})
    return (first is None), first, diffs


def which_field_leaked(r, o):
    """Attribute a divergence to action vs observation."""
    if r is None or o is None:
        return "trajectory length (one arm has no step here)"
    if r["action"] != o["action"]:
        return "real_action (agent chose differently)"
    if r["obs"] != o["obs"]:
        return "real_obs (env.page / env.obs leaked from the speculative branch)"
    return "none"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-questions", type=int, default=5)
    ap.add_argument("--out", default=os.path.join(REPO, "docs", "INVARIANT_REPORT.md"))
    ap.add_argument("--artifacts", default="/tmp/specmem/invariant")
    ap.add_argument("--retrieval-backend", default=None,
                    choices=["live", "title_exact", "bm25"],
                    help="override constants.retrieval_backend for this run")
    args = ap.parse_args()

    if args.retrieval_backend:
        constants.retrieval_backend = args.retrieval_backend

    # Handoff 4.4 requires WIKI_CACHE=1 so that Wikipedia cannot change between
    # arms. A frozen local corpus satisfies that requirement by construction and
    # more strongly -- there is no HTTP request to cache, and the cache counters
    # stay at 0/0, which the verdict's miss rule reads as clean. The guard is
    # therefore scoped to the live backend rather than dropped.
    if constants.retrieval_backend == "live" and os.environ.get("WIKI_CACHE") != "1":
        sys.exit("refusing to run: WIKI_CACHE=1 is required (handoff 4.4)")

    os.makedirs(args.artifacts, exist_ok=True)
    idxs = pilot_idxs(args.n_questions)
    prompt = build_prompt()

    print(f"pilot idxs: {idxs}")
    print(f"actor={constants.actor_model_name} spec={constants.spec_model_name} "
          f"temp={constants.temperature}/{constants.guess_temperature} "
          f"top_p={constants.top_p}/{constants.guess_top_p}")

    arms = {}
    for arm in ARMS:            # SEQ_a first: it warms the Wikipedia cache
        print(f"=== arm {arm} ===")
        arms[arm] = run_arm(arm, idxs, prompt, args.artifacts)

    raw_path = os.path.join(args.artifacts, "arms.json")
    with open(raw_path, "w") as fh:
        json.dump({"idxs": idxs, "arms": arms}, fh, indent=2)
    print(f"raw arms -> {raw_path}")

    report = build_report(idxs, arms, args)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        fh.write(report)
    print(f"report -> {args.out}")


def build_report(idxs, arms, args):
    import subprocess

    def q(arm, idx):
        return arms[arm]["questions"][idx]

    rows, divergence_blocks = [], []
    nondet_steps_by_idx, nondet_qlevel = {}, {}
    iso_pass, uniso_differs, spec_ran = True, False, False
    errors = []

    for idx in idxs:
        a, b, iso, un = q("SEQ_a", idx), q("SEQ_b", idx), q("ISO", idx), q("UNISO", idx)
        for arm, rec in (("SEQ_a", a), ("SEQ_b", b), ("ISO", iso), ("UNISO", un)):
            if rec["error"]:
                errors.append((arm, idx, rec["error"]))

        det_same, det_first, det_diffs = compare(a, b)
        nondet = {d["step"] for d in det_diffs}
        nondet_steps_by_idx[idx] = nondet
        qlevel_nondet = (a["n_steps"] != b["n_steps"]) or (a["em"] != b["em"])
        nondet_qlevel[idx] = qlevel_nondet

        iso_same_all, iso_first, iso_diffs = compare(a, iso)
        gated = [d for d in iso_diffs if d["step"] not in nondet]
        iso_ok = (not gated) and (qlevel_nondet or
                                  (a["n_steps"] == iso["n_steps"] and a["em"] == iso["em"]))
        if not iso_ok:
            iso_pass = False

        un_same, un_first, un_diffs = compare(a, un)
        if not un_same:
            uniso_differs = True

        for s in iso["sim_steps"]:
            real = next((r["obs"] for r in iso["steps"] if r["i"] == s["i"]), None)
            if real is not None and s["sim_obs"] != real:
                spec_ran = True
                break

        rows.append({
            "idx": idx, "n_steps": a["n_steps"],
            "det": "yes" if (det_same and not qlevel_nondet) else f"NO (step {det_first})",
            "iso": "yes" if iso_ok else f"NO (step {gated[0]['step'] if gated else '-'})",
            "uniso": "yes" if un_same else f"NO (step {un_first})",
            "first_div": gated[0]["step"] if gated else (un_first if not un_same else "-"),
        })

        for label, diffs in (("ISO vs SEQ_a", gated), ("UNISO vs SEQ_a", un_diffs)):
            for d in diffs[:2]:
                r, o = d["ref"], d["other"]
                i_tok, x, y = first_diff_token(
                    (r or {}).get("obs", ""), (o or {}).get("obs", ""))
                divergence_blocks.append(
                    f"#### idx {idx} — {label} — step {d['step']}\n\n"
                    f"**Field:** {which_field_leaked(r, o)}\n\n"
                    f"| | SEQ_a | {label.split()[0]} |\n|---|---|---|\n"
                    f"| real_action | `{(r or {}).get('action','<none>')}` "
                    f"| `{(o or {}).get('action','<none>')}` |\n"
                    f"| real_obs | {truncate((r or {}).get('obs','<none>'))} "
                    f"| {truncate((o or {}).get('obs','<none>'))} |\n\n"
                    + (f"First differing obs token (#{i_tok}): `{x}` vs `{y}`\n\n"
                       if i_tok is not None else "")
                )

    any_nondet = any(nondet_steps_by_idx[i] or nondet_qlevel[i] for i in idxs)
    post_seqa_misses = sum(arms[a]["cache"]["misses"] for a in ("SEQ_b", "ISO", "UNISO"))

    if errors:
        verdict, why = "FAIL", "one or more arms raised an exception; see Errors."
    elif post_seqa_misses > 0:
        verdict, why = ("INCONCLUSIVE",
                        f"{post_seqa_misses} Wikipedia cache misses after SEQ_a — "
                        "the comparison is confounded by possible Wikipedia drift.")
    elif not iso_pass:
        verdict, why = "FAIL", "ISO diverged from SEQ_a on at least one gated step."
    elif not uniso_differs:
        verdict, why = ("INCONCLUSIVE",
                        "UNISO == SEQ_a on all questions — the bug never fired, so a "
                        "green ISO is not evidence. Extend to 10 questions and re-evaluate.")
    elif not spec_ran:
        verdict, why = ("INCONCLUSIVE",
                        "ISO sim_obs never differed from real_obs — speculation did not "
                        "actually run, so ISO==SEQ_a is trivially true.")
    else:
        verdict, why = "PASS", "ISO is byte-identical to SEQ_a on every gated step."

    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                              capture_output=True, text=True).stdout.strip()[:7]
    except Exception:
        head = "unknown"

    L = []
    L.append("# INVARIANT_REPORT — online 3-arm speculation-isolation gate\n")
    L.append(f"**Verdict: {verdict}** — {why}\n")
    L.append(f"Commit `{head}` · questions {idxs} · "
             f"actor `{constants.actor_model_name}` · spec `{constants.spec_model_name}` · "
             f"temp {constants.temperature}/{constants.guess_temperature} · "
             f"top_p {constants.top_p}/{constants.guess_top_p} · "
             f"k={constants.guess_num_actions} · "
             f"retrieval_backend `{constants.retrieval_backend}`"
             + (" · WIKI_CACHE=1\n" if constants.retrieval_backend == "live"
                else " (frozen corpus; no HTTP, so the cache counters are 0/0 "
                     "and cannot confound the comparison)\n"))
    L.append("Criterion is the handoff doc §4.4, implemented verbatim in "
             "`scripts/run_invariant.py`. The compared object is the **realized** "
             "trajectory only — `(real_action, real_obs)` per step, plus `n_steps` "
             "and `em`. Speculative outputs are diagnostic, never part of the test.\n")

    L.append("## 1. Per-question\n")
    L.append("| idx | n_steps | SEQ_a==SEQ_b | ISO==SEQ_a | UNISO==SEQ_a | first divergent step |")
    L.append("|---|---|---|---|---|---|")
    for r in rows:
        L.append(f"| {r['idx']} | {r['n_steps']} | {r['det']} | {r['iso']} "
                 f"| {r['uniso']} | {r['first_div']} |")
    L.append("")

    L.append("## 2. Divergences\n")
    L.append("\n".join(divergence_blocks) if divergence_blocks
             else "No divergences in either comparison.\n")

    L.append("## 3. Wikipedia cache\n")
    L.append("| arm | hits | misses |")
    L.append("|---|---|---|")
    for arm in ARMS:
        c = arms[arm]["cache"]
        L.append(f"| {arm} | {c['hits']} | {c['misses']} |")
    L.append(f"\nMisses after SEQ_a: **{post_seqa_misses}** "
             f"({'clean' if post_seqa_misses == 0 else 'CONFOUNDED'}). "
             "SEQ_a runs first precisely to warm the cache; any later miss means a "
             "live Wikipedia fetch could have changed under the comparison.\n")

    L.append("## 4. Determinism control\n")
    if any_nondet:
        L.append("Nondeterministic steps (excluded from the gate):\n")
        for i in idxs:
            if nondet_steps_by_idx[i] or nondet_qlevel[i]:
                L.append(f"- idx {i}: steps {sorted(nondet_steps_by_idx[i]) or '—'}"
                         + (" · question-level (n_steps or em differ)" if nondet_qlevel[i] else ""))
        L.append("\nThis is a **server** issue, not a gate failure. Next diagnostics: "
                 "re-run with `--enforce-eager`, and confirm the vLLM `seed` is applied "
                 "per request. Versions in play are recorded below.\n")
    else:
        L.append("SEQ_a == SEQ_b on every step of every question. No steps excluded.\n")

    L.append("## 5. Power check\n")
    L.append(f"- UNISO != SEQ_a on >=1 question: **{'yes' if uniso_differs else 'NO'}**")
    L.append(f"- ISO sim_obs != real_obs on >=1 step (speculation ran): "
             f"**{'yes' if spec_ran else 'NO'}**\n")
    if not uniso_differs:
        L.append("Both must hold for PASS to mean anything. The contamination bug only "
                 "fires when a realized `lookup[]` follows a speculated `search[]` in the "
                 "same episode; if no episode had that shape, the gate has no power.\n")

    if errors:
        L.append("## 6. Errors\n")
        for arm, idx, e in errors:
            L.append(f"- `{arm}` idx {idx}: {e}")
        L.append("")

    L.append("## Environment\n")
    L.append(f"- vLLM **0.30.0**, torch 2.13.0+cu130 (ENV_PREP records 0.29.0 — drift)")
    L.append(f"- openai SDK 3.19.2 (ENV_PREP records 3.16.2 — drift)")
    L.append(f"- gymnasium 0.29.1 pinned, numpy 1.26.4")
    L.append(f"- raw per-arm trajectories: `{os.path.join(args.artifacts, 'arms.json')}`\n")

    L.append("## Harness fixes this gate required\n")
    L.append("- `HistoryWrapper.reset` accepted `idx` and passed `idx=None` downward, so "
             "`HotPotQAWrapper` drew a **random** question via `np.random.randint` on every "
             "reset. Fixed-index arms were impossible until this was fixed; upstream "
             "`runner.run()` never ran the seeded shuffle it computes.")
    L.append("- No Wikipedia cache existed. Added `environment.wiki_get` behind "
             "`WIKI_CACHE=1`, with hit/miss counters so §3 is checkable rather than assumed.\n")
    return "\n".join(L)


def truncate(s, n=160):
    s = (s or "").replace("\n", " ").replace("|", "\\|")
    return f"`{s[:n]}{'…' if len(s) > n else ''}`"


if __name__ == "__main__":
    main()

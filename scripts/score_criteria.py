#!/usr/bin/env python3
"""Score speculation pairs against the 13 acceptance criteria.

Reads a pairs JSONL file (schema: ``src/gates.Pair``), writes
``criteria_scores.jsonl`` -- one row per pair, keyed by ``pair_id`` -- plus a
sibling ``criteria_manifest.json`` recording the judge/embedding models, their
revisions, the threshold grids, the code version and the cache statistics.

Every LLM and embedding call goes through the append-only cache described in
``docs/JUDGE_CONTRACT.md``, so a re-run with the same inputs makes no new calls
and produces byte-identical scores.

Usage (from ``hotpotqa/``, with the servers up -- see ``scripts/serve_local.sh``):

    $PIPELINE_PY ../scripts/score_criteria.py \
        --pairs $SPEC_RUNS_DIR/pilot25_title_exact/pairs.jsonl \
        --out $SPEC_RUNS_DIR/pilot25_title_exact/criteria_scores.jsonl

    # deterministic criteria only, no server, no weights:
    python3 ../scripts/score_criteria.py --pairs pairs.jsonl \
        --out $SPEC_RUNS_DIR/scratch/criteria_scores.jsonl \
        --no-judge --no-embedder

    # the same-family circularity check for criteria 6-8:
    ... --judge-alt-model /models/models--NousResearch--Meta-Llama-3.1-8B-Instruct/... \
        --judge-alt-url http://127.0.0.1:8002/v1
"""

import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
HOTPOTQA = os.path.join(os.path.dirname(HERE), "hotpotqa")
sys.path.insert(0, HOTPOTQA)

from src import backends, durability, gates  # noqa: E402

#: Judge + embedding caches. Append-only and keyed by request content, so they
#: are simultaneously a cache and the record of what the judge was asked --
#: an output. The old default put them in the working tree
#: (hotpotqa/cache/judge); $JUDGE_CACHE_DIR from scripts/env.sh puts them on
#: persistent storage alongside the run they belong to.
DEFAULT_CACHE_DIR = os.environ.get(
    "JUDGE_CACHE_DIR",
    os.path.join(os.environ.get("SPEC_BASE",
                                os.path.expanduser("~/specmem-data")),
                 "cache", "judge"))


def git_version() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE,
                             capture_output=True, text=True, timeout=10)
        sha = out.stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=HERE,
                               capture_output=True, text=True, timeout=10)
        return sha + ("-dirty" if dirty.stdout.strip() else "")
    except Exception:
        return "unknown"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out", required=True,
                    help="criteria_scores.jsonl (appended to, never rewritten)")
    ap.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    ap.add_argument("--judge-model", default="Qwen/Qwen3-8B")
    ap.add_argument("--judge-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--judge-alt-model", default=None,
                    help="non-Qwen, similar size (same-family circularity check "
                         "for criteria 6-8); omit to record its absence")
    ap.add_argument("--judge-alt-url", default="http://127.0.0.1:8002/v1")
    ap.add_argument("--embed-model", default=backends.DEFAULT_EMBED_MODEL)
    ap.add_argument("--embed-device", default="cpu")
    ap.add_argument("--timestamp", default="2019-08-01",
                    help="<TIMESTAMP> for the Sufficient Context autorater "
                         "(default: the frozen KILT snapshot date)")
    ap.add_argument("--only", nargs="*", default=None,
                    help="criterion names to score (default: all 13)")
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--no-embedder", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)

    if args.only:
        unknown = [n for n in args.only if n not in gates.CRITERIA]
        if unknown:
            ap.error(f"unknown criteria: {unknown}")

    # The judge cache is the expensive artifact here: it is what makes a re-score
    # byte-identical without re-querying the judge. Losing it to a recycle means
    # every score has to be recomputed, so it is guarded like any other output.
    durability.require_durable_outputs(out=args.out, cache_dir=args.cache_dir)
    durability.warn_if_outside_spec_base(out=args.out, cache_dir=args.cache_dir)

    os.makedirs(args.cache_dir, exist_ok=True)
    caches = {}

    judge = judge_alt = embedder = None
    if not args.no_judge:
        caches["judge"] = backends.AppendOnlyCache(
            os.path.join(args.cache_dir, "judge_qwen.jsonl"))
        judge = backends.VLLMJudge(args.judge_model, args.judge_url,
                                   caches["judge"])
        if args.judge_alt_model:
            caches["judge_alt"] = backends.AppendOnlyCache(
                os.path.join(args.cache_dir, "judge_alt.jsonl"))
            judge_alt = backends.VLLMJudge(args.judge_alt_model,
                                           args.judge_alt_url,
                                           caches["judge_alt"])
    if not args.no_embedder:
        caches["embed"] = backends.AppendOnlyCache(
            os.path.join(args.cache_dir, "embeddings.jsonl"))
        embedder = backends.LocalEmbedder(args.embed_model, caches["embed"],
                                          device=args.embed_device)

    ctx = gates.Context(embedder=embedder, judge=judge, judge_alt=judge_alt,
                        timestamp_for_sufficient_context=args.timestamp,
                        code_version=git_version())

    pairs = gates.load_pairs(args.pairs)
    if args.limit:
        pairs = pairs[:args.limit]

    d = os.path.dirname(os.path.abspath(args.out))
    if d:
        os.makedirs(d, exist_ok=True)

    n_na = {}
    with open(args.out, "a", encoding="utf-8") as fh:
        for i, pair in enumerate(pairs, 1):
            row = gates.score_pair(pair, ctx, only=args.only)
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            fh.flush()
            for name, res in row["criteria"].items():
                if res.get("na_reason"):
                    n_na[name] = n_na.get(name, 0) + 1
            if i % 25 == 0 or i == len(pairs):
                print(f"  scored {i}/{len(pairs)}", flush=True)

    manifest = ctx.manifest()
    manifest.update({
        "pairs_file": os.path.abspath(args.pairs),
        "out_file": os.path.abspath(args.out),
        "n_pairs": len(pairs),
        "criteria_scored": args.only or list(gates.CRITERIA),
        "na_counts": n_na,
        "judge_alt_present": judge_alt is not None,
        "caches": {k: c.stats() for k, c in caches.items()},
        "registry": gates.registry_table(),
    })
    man_path = os.path.join(d or ".", "criteria_manifest.json")
    with open(man_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False, sort_keys=True)

    for c in caches.values():
        if c.collisions:
            print(f"  WARNING: {len(c.collisions)} cache key collisions with "
                  f"differing responses in {c.path} -- non-determinism, see "
                  "docs/JUDGE_CONTRACT.md", file=sys.stderr)
        c.close()

    print(f"wrote {len(pairs)} rows to {args.out}")
    print(f"manifest: {man_path}")
    if judge_alt is None and not args.no_judge:
        print("  note: no alternate judge configured -- criteria 6-8 were NOT "
              "cross-checked against a non-Qwen model (judge_alt: null)")
    if n_na:
        print(f"  NA counts: {n_na}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

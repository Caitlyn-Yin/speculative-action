#!/usr/bin/env bash
# Paper B scale-up driver — 100 questions per retrieval backend, unattended.
#
# Authorized by docs/PREREG_PAPER_B.md "Go/no-go" (all three gates passed at the
# 25-question pilot, second run; docs/PAPERB_PILOT_REPORT.md §6) and run under
# Amendment 1: both vLLM servers with --no-enable-prefix-caching.
#
#   bash scripts/run_paperb_scaleup.sh
#
# Resumable per stage: a stage whose output artifact already exists is skipped,
# so a re-invocation after a disconnect picks up where it stopped. Following
# run_grpo_chain.sh's rule, each stage trusts its *artifact*, not $?.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
source "$HERE/env.sh"

# The prereg's ladder: actor Qwen3-8B, speculator Qwen3-4B, k=3, step cap 8.
export ACTOR_MODEL="${ACTOR_MODEL:-Qwen/Qwen3-8B}"
export SPEC_MODEL="${SPEC_MODEL:-Qwen/Qwen3-4B}"
export ACTOR_GPU_FRAC="${ACTOR_GPU_FRAC:-0.60}"
export SPEC_GPU_FRAC="${SPEC_GPU_FRAC:-0.25}"

# Amendment 1 — prefix caching off on BOTH roles.
export EXTRA_ACTOR_ARGS="${EXTRA_ACTOR_ARGS:---no-enable-prefix-caching}"
export EXTRA_SPEC_ARGS="${EXTRA_SPEC_ARGS:---no-enable-prefix-caching}"

N_QUESTIONS="${N_QUESTIONS:-100}"
RUN_LABEL="${RUN_LABEL:-scale100}"
# $SPEC_RUNS_DIR, not $SPEC_BASE/paperb: checkpoint.sh only sweeps $SPEC_RUNS_DIR,
# and the 2026-10-02 recycle destroyed the previous run of this script because
# its OUT_ROOT was on the overlay.
OUT_ROOT="${OUT_ROOT:-$SPEC_RUNS_DIR}"
BACKENDS=(${BACKENDS:-title_exact bm25})

# Nothing starts until every output path is on persistent storage.
spec_require_durable "$OUT_ROOT" "$LOG_DIR" "$SPEC_RUNS_DIR" || exit 1

mkdir -p "$OUT_ROOT" "$LOG_DIR"

say() { echo "[$(date -u +%H:%M:%S)] $*"; }

# --- stage 0: servers -------------------------------------------------------
say "stage 0: servers ($ACTOR_MODEL :$ACTOR_PORT / $SPEC_MODEL :$SPEC_PORT, prefix caching off)"
bash "$HERE/serve_local.sh" both || { say "FATAL: servers did not come up"; exit 1; }
"$PIPELINE_PY" "$HERE/check_server.py" --all || { say "FATAL: check_server not green"; exit 1; }

# Record what the servers were actually started with, next to the data. JSON and
# inside a run directory, because checkpoint.sh only sweeps .jsonl/.json/.md and
# groups by the first directory under $SPEC_RUNS_DIR -- a loose .txt in the root
# would survive the recycle but never reach git, which is how evidence goes
# missing even with durable storage.
META_DIR="$OUT_ROOT/${RUN_LABEL}_meta"
mkdir -p "$META_DIR"
cat > "$META_DIR/serving.json" <<JSON
{
  "run_label": "$RUN_LABEL",
  "actor_model": "$ACTOR_MODEL",
  "actor_gpu_frac": "$ACTOR_GPU_FRAC",
  "extra_actor_args": "$EXTRA_ACTOR_ARGS",
  "spec_model": "$SPEC_MODEL",
  "spec_gpu_frac": "$SPEC_GPU_FRAC",
  "extra_spec_args": "$EXTRA_SPEC_ARGS",
  "max_model_len": "$MAX_MODEL_LEN",
  "vllm_seed": "$VLLM_SEED",
  "n_questions": "$N_QUESTIONS",
  "backends": "${BACKENDS[*]}",
  "git": "$(git -C "$REPO" rev-parse HEAD)$(git -C "$REPO" status --porcelain | head -1 | sed 's/.*/-dirty/')"
}
JSON

# --- stages 1-2 per backend -------------------------------------------------
for be in "${BACKENDS[@]}"; do
  out="$OUT_ROOT/${RUN_LABEL}_${be}"
  mkdir -p "$out"

  if [ -s "$out/pairs_raw.jsonl" ] && [ -s "$out/collect_manifest.json" ]; then
    say "stage 1 [$be]: collection already present — skipping"
  else
    say "stage 1 [$be]: collecting $N_QUESTIONS questions"
    RETRIEVAL_BACKEND="$be" SPEC_ACTIONS_FROM=speculator CAPTURE_SPEC_LOGPROBS=1 \
      "$PIPELINE_PY" "$HERE/collect_pairs.py" \
        --n-questions "$N_QUESTIONS" --out "$out" \
        --retrieval-backend "$be" --spec-actions-from speculator \
        --actor-model "$ACTOR_MODEL" --spec-model "$SPEC_MODEL" \
        --run "$RUN_LABEL" \
        > "$LOG_DIR/${RUN_LABEL}_collect_${be}.log" 2>&1
    [ -s "$out/pairs_raw.jsonl" ] || { say "FATAL: no pairs_raw.jsonl for $be"; tail -30 "$LOG_DIR/${RUN_LABEL}_collect_${be}.log"; exit 1; }
    say "stage 1 [$be]: $(wc -l < "$out/pairs_raw.jsonl") raw pairs"
  fi

  if [ -s "$out/pairs.jsonl" ] && [ -s "$out/audit_summary.json" ]; then
    say "stage 2 [$be]: audit already present — skipping"
  else
    say "stage 2 [$be]: replay audit (control + treatment per pair)"
    RETRIEVAL_BACKEND="$be" \
      "$PIPELINE_PY" "$HERE/replay_audit.py" \
        --collected "$out" --out "$out" --retrieval-backend "$be" \
        > "$LOG_DIR/${RUN_LABEL}_audit_${be}.log" 2>&1
    [ -s "$out/audit_summary.json" ] || { say "FATAL: no audit_summary.json for $be"; tail -30 "$LOG_DIR/${RUN_LABEL}_audit_${be}.log"; exit 1; }
  fi
  say "stage 2 [$be] summary: $(tr -d '\n ' < "$out/audit_summary.json" | cut -c1-400)"
done

# --- stage 3: statistics ----------------------------------------------------
# analyze_paperb.py's --out is a DIRECTORY and it writes <out>/analysis.json.
# This used to pass a "...stats.json" path as that directory and then test it
# with `[ -s ]`, which succeeds on a directory -- so a failed stage 3 would have
# looked like a completed one on the next resume.
stats_dir="$META_DIR/analysis"
stats="$stats_dir/analysis.json"
if [ -s "$stats" ]; then
  say "stage 3: stats already present — skipping"
else
  say "stage 3: cluster bootstrap over questions (H1-H4 where scorable)"
  args=()
  for be in "${BACKENDS[@]}"; do args+=(--backend "$be" "$OUT_ROOT/${RUN_LABEL}_${be}"); done
  "$PIPELINE_PY" "$HERE/analyze_paperb.py" "${args[@]}" --out "$stats_dir" \
    > "$LOG_DIR/${RUN_LABEL}_analyze.log" 2>&1
  [ -s "$stats" ] || { say "WARN: stage 3 produced nothing"; tail -30 "$LOG_DIR/${RUN_LABEL}_analyze.log"; }
fi

say "DONE. artifacts under $OUT_ROOT/${RUN_LABEL}_* (persistent storage)"
say "NEXT: bash $HERE/checkpoint.sh — a task is not finished until the artifacts are in git"

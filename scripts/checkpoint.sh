#!/usr/bin/env bash
# Carry run artifacts the last mile: persistent storage -> git -> origin.
#
#   bash scripts/checkpoint.sh                 # sweep, commit, push
#   bash scripts/checkpoint.sh --dry-run       # report only, touch nothing
#   bash scripts/checkpoint.sh --no-push       # commit locally (NOT durable)
#
# Copies every .jsonl / .json / .md file under $SPEC_RUNS_DIR smaller than
# $MAX_BYTES into the repo at results/<run_id>/<relative path>, commits, and
# pushes over the deploy key. Anything at or above the limit is LISTED with its
# path and size and left in place -- never silently skipped, because a silent
# truncation reads as "everything is checkpointed" when it is not.
#
# WHY THIS EXISTS
#
# Persistent storage stopped the 2026-10-02 class of loss (pod recycle wipes
# /tmp), but it does not stop the 2026-09 class: the GH200 held committed work
# that was never pushed, and when the host went away so did the work. JuiceFS is
# one quota on one node and shares a failure domain with everything on it.
# `git push` is the only step that puts an artifact somewhere this node cannot
# take with it. Hence WAYS_OF_WORKING.md's rule: a task ends by running this.
#
# run_id is the first path component under $SPEC_RUNS_DIR, so
#   $SPEC_RUNS_DIR/scale100_title_exact/pairs.jsonl
#     -> results/scale100_title_exact/pairs.jsonl
# A file sitting loose in the root is grouped under _root/.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
source "$HERE/env.sh"

MAX_BYTES="${MAX_BYTES:-52428800}"          # 50 MB
BRANCH="${BRANCH:-$(git -C "$REPO" rev-parse --abbrev-ref HEAD)}"
RESULTS_DIR="$REPO/results"

DRY_RUN=0
PUSH=1
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    --no-push) PUSH=0 ;;
    -h|--help) sed -n '2,27p' "$0"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

say() { echo "[checkpoint] $*"; }

# The source must itself be durable, or there is nothing to checkpoint from.
spec_require_durable "$SPEC_RUNS_DIR" || exit 1

if [ ! -d "$SPEC_RUNS_DIR" ]; then
  say "FATAL: \$SPEC_RUNS_DIR does not exist: $SPEC_RUNS_DIR"
  exit 1
fi

say "source: $SPEC_RUNS_DIR"
say "target: $RESULTS_DIR  (branch $BRANCH, limit $((MAX_BYTES / 1048576)) MB)"

# --- sweep ------------------------------------------------------------------
copied=0
copied_bytes=0
oversize=0
oversize_list=""

while IFS= read -r -d '' src; do
  size=$(stat -c %s "$src")
  rel="${src#"$SPEC_RUNS_DIR"/}"
  case "$rel" in
    */*) run_id="${rel%%/*}"; sub="${rel#*/}" ;;
    *)   run_id="_root";      sub="$rel"      ;;
  esac
  dest="$RESULTS_DIR/$run_id/$sub"

  if [ "$size" -ge "$MAX_BYTES" ]; then
    oversize=$((oversize + 1))
    oversize_list+="    $(numfmt --to=iec --suffix=B --padding=8 "$size" 2>/dev/null || echo "${size}B")  $src"$'\n'
    continue
  fi

  if [ "$DRY_RUN" -eq 1 ]; then
    [ -f "$dest" ] && cmp -s "$src" "$dest" && continue
    say "would copy  $rel  ($size B)"
  else
    mkdir -p "$(dirname "$dest")"
    # -p so the artifact keeps the mtime of the run that produced it.
    cp -p "$src" "$dest"
  fi
  copied=$((copied + 1))
  copied_bytes=$((copied_bytes + size))
done < <(find "$SPEC_RUNS_DIR" -type f \
             \( -name '*.jsonl' -o -name '*.json' -o -name '*.md' \) -print0)

say "eligible artifacts: $copied ($(numfmt --to=iec --suffix=B "$copied_bytes" 2>/dev/null || echo "$copied_bytes B"))"

# --- report what was NOT carried -------------------------------------------
if [ "$oversize" -gt 0 ]; then
  say "NOT COPIED — $oversize file(s) at or above the $((MAX_BYTES / 1048576)) MB limit:"
  printf '%s' "$oversize_list"
  say "these remain only on $SPEC_BASE (one node, one quota) — get them off-node"
  say "by hand if they matter, or summarise them into docs/."
fi

# Non-matching extensions are invisible to this sweep; say so rather than imply
# full coverage. .log is excluded by design (noisy, large, regenerable).
others=$(find "$SPEC_RUNS_DIR" -type f \
             ! -name '*.jsonl' ! -name '*.json' ! -name '*.md' | wc -l)
if [ "$others" -gt 0 ]; then
  say "note: $others file(s) under \$SPEC_RUNS_DIR are not .jsonl/.json/.md and"
  say "      were not considered (logs, parquet, sqlite, ...)."
fi

if [ "$DRY_RUN" -eq 1 ]; then
  say "dry run — nothing copied, committed or pushed"
  exit 0
fi

if [ "$copied" -eq 0 ]; then
  say "no eligible artifacts found — nothing to commit"
  exit 0
fi

# --- commit -----------------------------------------------------------------
cd "$REPO" || exit 1
git add -A results

if git diff --cached --quiet; then
  say "results/ unchanged since the last checkpoint — nothing to commit"
else
  n_files=$(git diff --cached --name-only | wc -l)
  git commit -q -F - <<MSG
checkpoint: $n_files result file(s) from \$SPEC_RUNS_DIR

Swept $SPEC_RUNS_DIR for .jsonl/.json/.md under $((MAX_BYTES / 1048576)) MB and
copied them into results/<run_id>/. Oversize files not copied: $oversize.

Run by scripts/checkpoint.sh (docs/WAYS_OF_WORKING.md, "Durability rules").

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
MSG
  say "committed $(git rev-parse --short HEAD)"
fi

# --- push -------------------------------------------------------------------
if [ "$PUSH" -eq 0 ]; then
  say "WARNING: --no-push given. The artifacts are committed LOCALLY ONLY,"
  say "which WAYS_OF_WORKING.md §1 does not count as done."
  exit 0
fi

if git push origin "$BRANCH"; then
  say "pushed: $(git rev-parse HEAD) -> origin/$BRANCH"
  say "verify: $(git ls-remote origin "refs/heads/$BRANCH")"
else
  rc=$?
  say "FATAL: push FAILED (rc=$rc). The artifacts are committed locally only."
  say "Per WAYS_OF_WORKING.md §2, report this blocker rather than treating the"
  say "task as finished. Check: ssh -T git@github.com-specmem"
  exit $rc
fi

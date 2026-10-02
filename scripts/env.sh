#!/usr/bin/env bash
# Shared paths for the speculative-action pipeline.
#
# TWO ROOTS, and the distinction is the whole point of this file:
#
#   $SPEC_BASE     PERSISTENT (JuiceFS, $HOME/specmem-data). Survived every pod
#                  recycle so far. **EVERY OUTPUT GOES HERE** -- runs/, windows,
#                  replay results, criterion scores, judge and embedding caches,
#                  logs. Nothing a run produces may live anywhere else.
#   $SPEC_SCRATCH  EPHEMERAL (container overlay, /tmp/specmem). Wiped by every
#                  pod recycle. **READ-ONLY INPUTS AND REBUILDABLE TOOLING ONLY**
#                  -- conda envs, model weights, XDG dirs, raw corpus download.
#                  Everything here is reproducible from a committed script.
#
# This split exists because every output loss on this project came from writing
# to /tmp: Steps 1-2.5 and Phases A/B (2026-09, GH200), and then the whole Paper
# B pilot + scale-up corpus (2026-10-02 recycle, which wiped /tmp/specmem while
# the scale-up was still running). `scripts/checkpoint.sh` carries the small
# artifacts the rest of the way into git; see docs/WAYS_OF_WORKING.md.
#
# The guard that enforces this is hotpotqa/src/durability.py, imported by every
# entry script. It aborts rather than letting a run write to an ephemeral mount.
#
# $SPEC_BASE is on a 64 GB JuiceFS quota that is already ~87% full (the 7.7 GiB
# frozen corpus is most of it), so outputs must stay jsonl/json-sized. The
# pre-existing ES-project cache at $HOME/.cache/huggingface is left untouched.

export SPEC_BASE="${SPEC_BASE:-$HOME/specmem-data}"
export SPEC_SCRATCH="${SPEC_SCRATCH:-/tmp/specmem}"

# Inputs and tooling: scratch. Rebuild with setup_envs.sh + download_ladder.sh.
export HF_HOME="${HF_HOME:-$SPEC_SCRATCH/hf_home}"
export HF_HUB_DISABLE_XET=1

export PIPELINE_PY="$SPEC_SCRATCH/envs/pipeline/bin/python"   # repo pipeline env
export VLLM_ENV="$SPEC_SCRATCH/envs/vllm"                     # separate serving env
export VLLM_PY="$VLLM_ENV/bin/python"

# The serving env ships libstdc++ 6.0.36 (CXXABI_1.3.15); the system one at
# /lib/x86_64-linux-gnu is older and gets resolved first, which breaks
# `import sqlite3` deep inside vllm's structured-output imports:
#   ImportError: libstdc++.so.6: version `CXXABI_1.3.15' not found
#                (required by .../libicui18n.so.78)
# Prepend the env's lib for anything run with $VLLM_PY.
export VLLM_LD_LIBRARY_PATH="$VLLM_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Trap 5 (same root cause, pipeline side). `import sqlite3` in the pipeline env
# resolves `_sqlite3` -> libicui18n.so.78 -> CXXABI_1.3.15, which the system
# libstdc++ does not provide. It only fails when something else has already
# pulled in the system libstdc++ first, which makes it *order dependent*:
# `pytest tests/test_local_wiki.py` passed while `pytest tests/` failed. Putting
# the env's lib on the global search path removes the ordering dependence.
# Harmless for the serving env, whose own lib dir is prepended ahead of this.
export LD_LIBRARY_PATH="$SPEC_SCRATCH/envs/pipeline/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

export ACTOR_MODEL="${ACTOR_MODEL:-Qwen/Qwen3-8B}"
export SPEC_MODEL="${SPEC_MODEL:-Qwen/Qwen3-0.6B}"
export ACTOR_PORT="${ACTOR_PORT:-8000}"
export SPEC_PORT="${SPEC_PORT:-8001}"

# GPU split across the two concurrent servers on one H200 (143 GB).
# Must sum to < 1.0, with headroom for CUDA context and activations.
export ACTOR_GPU_FRAC="${ACTOR_GPU_FRAC:-0.60}"
export SPEC_GPU_FRAC="${SPEC_GPU_FRAC:-0.25}"

# $HOME/.config is a root-owned *file* on this node, not a directory, so any
# library resolving an XDG path under it dies with
#   NotADirectoryError: [Errno 20] Not a directory: '/home/hyin66/.config/vllm'
# Redirect XDG onto the overlay. This also keeps caches off the JuiceFS home
# (flashinfer otherwise writes to $HOME/.cache/flashinfer). These are caches,
# not outputs, so scratch is the right home for them.
export XDG_CONFIG_HOME="${XDG_CONFIG_HOME:-$SPEC_SCRATCH/xdg/config}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$SPEC_SCRATCH/xdg/cache}"
mkdir -p "$XDG_CONFIG_HOME" "$XDG_CACHE_HOME"

# No CUDA toolkit on this node (no nvcc), so flashinfer cannot JIT-compile its
# sampling kernels:  /usr/local/cuda/bin/nvcc: not found  ->  EngineCore dies.
# Fall back to vLLM's native PyTorch sampler. Harmless for this work: all
# requests are greedy (temperature 0), so the sampler is an argmax either way.
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

export MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
export VLLM_SEED="${VLLM_SEED:-0}"

# --------------------------------------------------------------------------
# OUTPUTS -- all of these are on $SPEC_BASE (persistent) by construction.
#
# $SPEC_RUNS_DIR is the one directory scripts/checkpoint.sh sweeps: every
# .jsonl/.json/.md under it smaller than 50 MB is copied into the repo under
# results/<run_id>/ and pushed. So a new run artifact that is NOT under
# $SPEC_RUNS_DIR will survive a recycle but will never reach git -- put run
# outputs there, one subdirectory per run id.
# --------------------------------------------------------------------------
export SPEC_RUNS_DIR="${SPEC_RUNS_DIR:-$SPEC_BASE/runs}"
export LOG_DIR="${LOG_DIR:-$SPEC_BASE/logs}"

# Judge + embedding caches (docs/JUDGE_CONTRACT.md). Append-only and keyed by
# request content, so they are both a cache and the record of what the judge was
# asked -- an output, not scratch. score_criteria.py used to default these into
# the working tree (hotpotqa/cache/judge); it now reads this.
export JUDGE_CACHE_DIR="${JUDGE_CACHE_DIR:-$SPEC_BASE/cache/judge}"

# Live-Wikipedia response cache (environment.wiki_get, behind WIKI_CACHE=1).
# Recorded evidence for any run on the `live` backend, hence persistent.
export WIKI_CACHE_DIR="${WIKI_CACHE_DIR:-$SPEC_BASE/cache/wiki}"

mkdir -p "$SPEC_RUNS_DIR" "$LOG_DIR" "$JUDGE_CACHE_DIR" "$WIKI_CACHE_DIR"

# --------------------------------------------------------------------------
# Frozen local Wikipedia (KILT knowledge source, 2019-08-01).
#
# UNLIKE the model weights, the built corpus lives on **persistent** storage
# ($HOME / JuiceFS), because /tmp is wiped on every pod recycle and rebuilding
# costs a 34.8 GiB download plus ~40 min of CPU. The *raw* download is scratch
# and stays on the overlay -- it is only needed to (re)build.
#
# Layout under $WIKI_DATA_DIR (see docs/LOCAL_WIKI.md):
#   pages.zst      chunked zstd frames of page text
#   pages.sqlite   title -> (frame offset, item index)
#   bm25/          bm25s index over title + lead paragraph
#   leads.jsonl.zst  title + lead paragraph (BM25 build input, kept for rebuilds)
#   MANIFEST.json  sha256 / md5 of the source, build commands, counts, versions
# --------------------------------------------------------------------------
export WIKI_SNAPSHOT="${WIKI_SNAPSHOT:-kilt_20190801}"
export WIKI_DATA_ROOT="${WIKI_DATA_ROOT:-$SPEC_BASE/local_wiki}"
export WIKI_DATA_DIR="${WIKI_DATA_DIR:-$WIKI_DATA_ROOT/$WIKI_SNAPSHOT}"
export WIKI_RAW_DIR="${WIKI_RAW_DIR:-$SPEC_SCRATCH/kilt_raw}"
export WIKI_KS_URL="${WIKI_KS_URL:-http://dl.fbaipublicfiles.com/KILT/kilt_knowledgesource.json}"
mkdir -p "$WIKI_DATA_DIR"

# Embedding model weights for criteria 5 and 12 (BAAI/bge-base-en-v1.5, 419 MB,
# docs/CRITERIA.md). A read-only input, but it predates this split and already
# sits on persistent storage, so it stays where it is rather than being
# re-downloaded on every recycle.
export HF_EMBED_HOME="${HF_EMBED_HOME:-$SPEC_BASE/hf_embed}"

# --------------------------------------------------------------------------
# Durability guard, shell side.
#
# The Python entry scripts import hotpotqa/src/durability.py directly. Shell
# entry scripts call this, which runs the same module through the *system*
# python3 -- deliberately, so the guard also works on a freshly recycled pod
# where $PIPELINE_PY does not exist yet.
#
#   spec_require_durable "$OUT_ROOT" "$LOG_DIR" || exit 1
# --------------------------------------------------------------------------
SPEC_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export SPEC_REPO_ROOT

spec_require_durable() {
  python3 "$SPEC_REPO_ROOT/hotpotqa/src/durability.py" --check "$@" --quiet
}

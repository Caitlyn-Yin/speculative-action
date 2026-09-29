#!/usr/bin/env bash
# Shared paths for the speculative-action pipeline.
#
# Everything lives on the container overlay (/tmp/specmem, ~150 GB free), NOT
# on $HOME: $HOME is JuiceFS with a 64 GB quota and ~12 GB free, and filling it
# has corrupted artifacts before. The pre-existing ES-project HF cache at
# $HOME/.cache/huggingface is deliberately left untouched.
#
# NOTE: /tmp does not survive a pod restart. If the node is recycled, re-run
# scripts/setup_envs.sh (envs) and scripts/download_ladder.sh (weights).

export SPEC_BASE="${SPEC_BASE:-/tmp/specmem}"
export HF_HOME="${HF_HOME:-$SPEC_BASE/hf_home}"
export HF_HUB_DISABLE_XET=1

export PIPELINE_PY="$SPEC_BASE/envs/pipeline/bin/python"   # repo pipeline env
export VLLM_ENV="$SPEC_BASE/envs/vllm"                     # separate serving env
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
export LD_LIBRARY_PATH="$SPEC_BASE/envs/pipeline/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

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
# Redirect XDG onto the overlay. This also keeps caches off the 12 GB JuiceFS
# home (flashinfer otherwise writes to $HOME/.cache/flashinfer).
export XDG_CONFIG_HOME="${XDG_CONFIG_HOME:-$SPEC_BASE/xdg/config}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$SPEC_BASE/xdg/cache}"
mkdir -p "$XDG_CONFIG_HOME" "$XDG_CACHE_HOME"

# No CUDA toolkit on this node (no nvcc), so flashinfer cannot JIT-compile its
# sampling kernels:  /usr/local/cuda/bin/nvcc: not found  ->  EngineCore dies.
# Fall back to vLLM's native PyTorch sampler. Harmless for this work: all
# requests are greedy (temperature 0), so the sampler is an argmax either way.
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

export MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
export VLLM_SEED="${VLLM_SEED:-0}"
export LOG_DIR="${LOG_DIR:-$SPEC_BASE/logs}"
mkdir -p "$LOG_DIR"

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
export WIKI_DATA_ROOT="${WIKI_DATA_ROOT:-$HOME/specmem-data/local_wiki}"
export WIKI_DATA_DIR="${WIKI_DATA_DIR:-$WIKI_DATA_ROOT/$WIKI_SNAPSHOT}"
export WIKI_RAW_DIR="${WIKI_RAW_DIR:-/tmp/kilt_raw}"
export WIKI_KS_URL="${WIKI_KS_URL:-http://dl.fbaipublicfiles.com/KILT/kilt_knowledgesource.json}"
mkdir -p "$WIKI_DATA_DIR"

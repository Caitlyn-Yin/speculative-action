#!/usr/bin/env bash
# P1 — build the two environments on the container overlay.
#
#   bash scripts/setup_envs.sh            # both envs
#   bash scripts/setup_envs.sh pipeline   # repo deps only (enough for pytest)
#   bash scripts/setup_envs.sh vllm       # serving stack only
#
# Two separate envs, deliberately:
#   pipeline/ : the repo's own deps (openai SDK, gymnasium, bs4, pytest)
#   vllm/     : the serving stack (vllm + torch), kept apart so a serving
#               upgrade cannot perturb the pipeline's pinned gymnasium.
#
# Both live under $SPEC_SCRATCH (the container overlay) and NOT under
# $SPEC_BASE: they are rebuildable tooling, not outputs, and $SPEC_BASE is a
# 64 GB JuiceFS quota that is already ~87% full. Losing these to a pod recycle
# costs one `bash scripts/setup_envs.sh`, which is the whole point of the
# scratch/persistent split in scripts/env.sh. The existing ES-project cache at
# $HOME/.cache/huggingface and the envs at $HOME/micromamba are left untouched.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/env.sh"

# The offline test suite only needs the pipeline env, so it is separately
# buildable: rebuilding the serving stack pulls vllm + torch for nothing.
WHICH="${1:-both}"

export MAMBA_ROOT_PREFIX="$SPEC_SCRATCH/mamba"
mkdir -p "$MAMBA_ROOT_PREFIX" "$SPEC_SCRATCH/envs"

if [ "$WHICH" = "both" ] || [ "$WHICH" = "pipeline" ]; then
echo "=== pipeline env ==="
micromamba create -y -p "$SPEC_SCRATCH/envs/pipeline" python=3.10 -c conda-forge
"$SPEC_SCRATCH/envs/pipeline/bin/pip" install --no-input \
    openai requests beautifulsoup4 pandas matplotlib \
    "huggingface_hub[cli]" pytest

# gymnasium is PINNED. The repo relies on gym.Wrapper attribute forwarding
# (e.g. HotPotQAWrapper._get_info reads self.steps from the inner WikiEnv,
# wrappers.py:143-149; runner reads env.page/env.sim_obs). Wrapper.__getattr__
# was REMOVED in gymnasium 1.0, so 1.3.0 raises AttributeError throughout.
# 0.29.1 keeps forwarding (with a DeprecationWarning) and needs numpy<2.
"$SPEC_SCRATCH/envs/pipeline/bin/pip" install --no-input "gymnasium==0.29.1" "numpy<2"

# Frozen local Wikipedia (docs/LOCAL_WIKI.md). These were installed by hand when
# the corpus was built, so the 2026-09-29 pod recycle came back with a pipeline
# env that could not read pages.zst at all (ModuleNotFoundError: zstandard, 16
# errors in tests/test_local_wiki.py). bm25s is pinned because the index on disk
# was written by 0.3.11.
"$SPEC_SCRATCH/envs/pipeline/bin/pip" install --no-input \
    zstandard "bm25s==0.3.11" PyStemmer pyarrow

fi

if [ "$WHICH" = "both" ] || [ "$WHICH" = "vllm" ]; then
echo "=== vllm serving env ==="
micromamba create -y -p "$SPEC_SCRATCH/envs/vllm" python=3.12 -c conda-forge
"$SPEC_SCRATCH/envs/vllm/bin/pip" install --no-input vllm

fi

echo "=== versions ==="
if [ "$WHICH" = "both" ] || [ "$WHICH" = "pipeline" ]; then
"$SPEC_SCRATCH/envs/pipeline/bin/python" -c \
  "import gymnasium,numpy,openai;print('pipeline: gymnasium',gymnasium.__version__,'numpy',numpy.__version__,'openai',openai.__version__)"
fi
if [ "$WHICH" = "both" ] || [ "$WHICH" = "vllm" ]; then
LD_LIBRARY_PATH="$VLLM_LD_LIBRARY_PATH" "$SPEC_SCRATCH/envs/vllm/bin/python" -c \
  "import vllm,torch;print('serving: vllm',vllm.__version__,'torch',torch.__version__,'cuda',torch.version.cuda)"
fi
echo "done"

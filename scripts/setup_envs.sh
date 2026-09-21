#!/usr/bin/env bash
# P1 — build the two environments on the container overlay.
#
#   bash scripts/setup_envs.sh
#
# Two separate envs, deliberately:
#   pipeline/ : the repo's own deps (openai SDK, gymnasium, bs4, pytest)
#   vllm/     : the serving stack (vllm + torch), kept apart so a serving
#               upgrade cannot perturb the pipeline's pinned gymnasium.
#
# Both live under $SPEC_BASE (default /tmp/specmem) because $HOME is JuiceFS
# with ~12 GB free. The existing ES-project cache at $HOME/.cache/huggingface
# and the envs at $HOME/micromamba are left untouched.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/env.sh"

export MAMBA_ROOT_PREFIX="$SPEC_BASE/mamba"
mkdir -p "$MAMBA_ROOT_PREFIX" "$SPEC_BASE/envs"

echo "=== pipeline env ==="
micromamba create -y -p "$SPEC_BASE/envs/pipeline" python=3.10 -c conda-forge
"$SPEC_BASE/envs/pipeline/bin/pip" install --no-input \
    openai requests beautifulsoup4 pandas matplotlib \
    "huggingface_hub[cli]" pytest

# gymnasium is PINNED. The repo relies on gym.Wrapper attribute forwarding
# (e.g. HotPotQAWrapper._get_info reads self.steps from the inner WikiEnv,
# wrappers.py:143-149; runner reads env.page/env.sim_obs). Wrapper.__getattr__
# was REMOVED in gymnasium 1.0, so 1.3.0 raises AttributeError throughout.
# 0.29.1 keeps forwarding (with a DeprecationWarning) and needs numpy<2.
"$SPEC_BASE/envs/pipeline/bin/pip" install --no-input "gymnasium==0.29.1" "numpy<2"

echo "=== vllm serving env ==="
micromamba create -y -p "$SPEC_BASE/envs/vllm" python=3.12 -c conda-forge
"$SPEC_BASE/envs/vllm/bin/pip" install --no-input vllm

echo "=== versions ==="
"$SPEC_BASE/envs/pipeline/bin/python" -c \
  "import gymnasium,numpy,openai;print('pipeline: gymnasium',gymnasium.__version__,'numpy',numpy.__version__,'openai',openai.__version__)"
LD_LIBRARY_PATH="$VLLM_LD_LIBRARY_PATH" "$SPEC_BASE/envs/vllm/bin/python" -c \
  "import vllm,torch;print('serving: vllm',vllm.__version__,'torch',torch.__version__,'cuda',torch.version.cuda)"
echo "done"

#!/usr/bin/env bash
# P2 - download the actor + speculator ladder to $HF_HOME on the overlay.
#
#   bash scripts/download_ladder.sh
#
# 56 GB of read-only model weights. These go to $SPEC_SCRATCH deliberately and
# must NEVER go to $SPEC_BASE: the JuiceFS quota is 64 GB total with ~8 GB free.
# A pod recycle loses them and this script puts them back; ENV_PREP.md pins the
# five config.json sha256 hashes so a re-download is verifiable.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/env.sh"
PY=$PIPELINE_PY

MODELS="Qwen/Qwen3-0.6B Qwen/Qwen3-1.7B Qwen/Qwen3-4B Qwen/Qwen3-8B Qwen/Qwen3-14B"

for m in $MODELS; do
  echo "=== $(date -Is) downloading $m ==="
  $PY - "$m" <<'PYEOF'
import sys
from huggingface_hub import snapshot_download
repo = sys.argv[1]
p = snapshot_download(
    repo_id=repo,
    allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"],
    ignore_patterns=["*.pth", "*.bin", "*.gguf", "original/*"],
    max_workers=8,
)
print("LOCAL_PATH", repo, p)
PYEOF
  echo "=== $(date -Is) done $m rc=$? ==="
done
echo "ALL_DOWNLOADS_COMPLETE $(date -Is)"

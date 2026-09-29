#!/usr/bin/env bash
# Launch the two local vLLM servers: actor on :8000, speculator on :8001.
#
#   bash scripts/serve_local.sh            # both servers, background, wait for ready
#   bash scripts/serve_local.sh actor      # actor only
#   bash scripts/serve_local.sh spec       # speculator only
#   bash scripts/serve_local.sh stop       # stop both
#
# Greedy decoding is a per-request property (temperature=0 from
# hotpotqa/src/constants.py); the server is started with a fixed --seed so any
# non-greedy path is at least reproducible.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/env.sh"

WHICH="${1:-both}"

stop_servers() {
  pkill -f "vllm.entrypoints.openai.api_server.*--port $ACTOR_PORT" 2>/dev/null
  pkill -f "vllm.entrypoints.openai.api_server.*--port $SPEC_PORT" 2>/dev/null
  echo "stop signal sent to :$ACTOR_PORT and :$SPEC_PORT"
}

# Extra per-role flags. Used to attribute the 2026-09-29 replay nondeterminism:
# the actor was not deterministic on a repeated *identical* request
# (scripts/probe_determinism.py), and vLLM's prefix cache is the suspect, since
# a prompt whose prefix is cached is prefilled in a different chunk layout than
# one computed fresh. Set e.g.
#   EXTRA_ACTOR_ARGS="--no-enable-prefix-caching" bash scripts/serve_local.sh actor
export EXTRA_ACTOR_ARGS="${EXTRA_ACTOR_ARGS:-}"
export EXTRA_SPEC_ARGS="${EXTRA_SPEC_ARGS:-}"

start_one() {
  local role="$1" model="$2" port="$3" frac="$4"
  local log="$LOG_DIR/vllm_${role}.log"
  local extra=""
  case "$role" in
    actor) extra="$EXTRA_ACTOR_ARGS" ;;
    spec)  extra="$EXTRA_SPEC_ARGS" ;;
  esac

  if curl -sf "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1; then
    echo "[$role] already serving on :$port — leaving it alone"
    return 0
  fi

  echo "[$role] starting $model on :$port (gpu_frac=$frac) -> $log"
  HF_HOME="$HF_HOME" HF_HUB_OFFLINE=1 LD_LIBRARY_PATH="$VLLM_LD_LIBRARY_PATH" \
  XDG_CONFIG_HOME="$XDG_CONFIG_HOME" XDG_CACHE_HOME="$XDG_CACHE_HOME" \
  VLLM_USE_FLASHINFER_SAMPLER="$VLLM_USE_FLASHINFER_SAMPLER" \
  nohup "$VLLM_PY" -m vllm.entrypoints.openai.api_server \
      --model "$model" \
      --served-model-name "$model" \
      --port "$port" \
      --host 127.0.0.1 \
      --gpu-memory-utilization "$frac" \
      --max-model-len "$MAX_MODEL_LEN" \
      --seed "$VLLM_SEED" \
      --no-enable-log-requests \
      $extra \
      > "$log" 2>&1 &
  echo "[$role] pid $!"
}

wait_ready() {
  local role="$1" port="$2" timeout="${3:-900}"
  local waited=0
  until curl -sf "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1; do
    if ! pgrep -f "api_server.*--port $port" >/dev/null; then
      echo "[$role] FAILED — process gone. Tail of log:"
      tail -30 "$LOG_DIR/vllm_${role}.log"
      return 1
    fi
    if [ "$waited" -ge "$timeout" ]; then
      echo "[$role] TIMEOUT after ${timeout}s"
      return 1
    fi
    sleep 5
    waited=$((waited + 5))
  done
  echo "[$role] ready on :$port after ${waited}s"
}

case "$WHICH" in
  stop)
    stop_servers
    ;;
  actor)
    start_one actor "$ACTOR_MODEL" "$ACTOR_PORT" "$ACTOR_GPU_FRAC"
    wait_ready actor "$ACTOR_PORT"
    ;;
  spec)
    start_one spec "$SPEC_MODEL" "$SPEC_PORT" "$SPEC_GPU_FRAC"
    wait_ready spec "$SPEC_PORT"
    ;;
  both)
    start_one actor "$ACTOR_MODEL" "$ACTOR_PORT" "$ACTOR_GPU_FRAC"
    start_one spec "$SPEC_MODEL" "$SPEC_PORT" "$SPEC_GPU_FRAC"
    wait_ready actor "$ACTOR_PORT" || exit 1
    wait_ready spec "$SPEC_PORT" || exit 1
    echo "both servers ready; now run: \$PIPELINE_PY scripts/check_server.py --all"
    ;;
  *)
    echo "usage: $0 [both|actor|spec|stop]" >&2
    exit 2
    ;;
esac

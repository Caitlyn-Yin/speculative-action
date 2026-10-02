import os

openrouter_api_key = "your-openrouter-api-key"
openrouter_model_name = "openai/gpt-4"
openrouter_guess_model_name = "openai/gpt-5-nano"

# --------------------------------------------------------------------------
# Backend selection. "local" = OpenAI-compatible vLLM servers (the only
# backend permitted for research runs). "external" = the original
# Gemini/OpenAI/OpenRouter path, kept reachable behind this flag.
# --------------------------------------------------------------------------
llm_backend = "local"

# Two concurrent servers: actor on :8000, speculator on :8001.
actor_base_url = "http://127.0.0.1:8000/v1"
spec_base_url = "http://127.0.0.1:8001/v1"
actor_model_name = "Qwen/Qwen3-8B"
spec_model_name = "Qwen/Qwen3-0.6B"

# Speculation isolation. True = snapshot/restore the environment's
# observation state around the speculative branch (correct). False = upstream
# behaviour, i.e. the contaminated "spec-unisolated" arm of the online gate.
isolate_speculation = True

# --------------------------------------------------------------------------
# Who proposes the k speculated actions.
#
#   "actor"       upstream behaviour: runner.webthink generates the k
#                 candidates with self.llm, i.e. the ACTOR model, from a prompt
#                 whose observations are the speculated ones. The speculator
#                 model is then used only inside WikiEnv.guess_step to imagine
#                 the page.
#   "speculator"  the k candidates come from the speculator model
#                 (spec_model_name @ spec_base_url), which is what Speculative
#                 Actions describes and what docs/PREREG_PAPER_B.md assumes
#                 ("Actor Qwen3-8B, speculator Qwen3-4B, k=3").
#
# Default stays "actor" so every previously recorded run keeps its semantics;
# Paper B collection sets "speculator" explicitly and records it in the
# manifest.
# --------------------------------------------------------------------------
spec_actions_from = os.environ.get("SPEC_ACTIONS_FROM", "actor")

# Capture per-token log-probabilities for the speculated-action generation.
# Needed by criterion 9 (`spec_confidence`, gates.py) and by nothing else, so
# it is off unless a Paper B collection turns it on. Local backend only.
capture_spec_logprobs = os.environ.get("CAPTURE_SPEC_LOGPROBS", "0") == "1"

# --------------------------------------------------------------------------
# Retrieval backend for WikiEnv.search_step.
#
#   "live"        upstream: HTTP to en.wikipedia.org (drifts; not replayable)
#   "title_exact" frozen KILT 2019-08-01 corpus; a query that resolves to no
#                 page is a MISS -> "Could not find X. Similar: [...]"
#   "bm25"        same corpus, same everything, except a miss is silently
#                 answered with the top-1 BM25 page
#
# The two local modes differ in exactly one behaviour (local_wiki.LocalWiki
# ._on_miss), which is what makes them an environment manipulation rather than
# two different environments. Default stays "live" so that existing runs keep
# their recorded semantics; set it here or via RETRIEVAL_BACKEND.
# --------------------------------------------------------------------------
retrieval_backend = os.environ.get("RETRIEVAL_BACKEND", "live")

local_api_key = "EMPTY"          # vLLM ignores the value, the SDK requires one
local_seed = 0                   # forwarded to vLLM for reproducibility
local_request_timeout = 600
local_max_retries = 3

prompts_folder = "./prompts/"
prompt_file = "prompts_naive.json"
agent_role = "Question Answering Agent"

# --------------------------------------------------------------------------
# Where run OUTPUTS go.
#
# Upstream wrote trajectories to "./run_metrics/..." and "./trajs/", i.e.
# relative to the CWD, which meant the working tree. That is durable here only
# by accident (the repo is on JuiceFS) and it pollutes `git status` with
# per-run directories. Outputs now default under $SPEC_RUNS_DIR, which
# scripts/env.sh points at persistent storage and scripts/checkpoint.sh sweeps
# into git.
#
# Falling back to "." when the variable is unset keeps `pytest` and ad-hoc
# invocations working without sourcing env.sh; the durability guard
# (src/durability.py) is what stops that fallback from landing on /tmp.
# --------------------------------------------------------------------------
run_output_root = os.environ.get("SPEC_RUNS_DIR", ".")

random_seed = 248
num = 7405
n_steps_to_run = 8
n_samples_to_run = 20
client_error_sleep_time = 60
server_error_sleep_time = 60
trajectory_filenames = ["log.txt", "normalobs.json", "simobs.json", "metrics.json"]
guess_num_actions = 3
max_agent_retries = 1
max_guess_retries = 3

# Agent LLM settings
# Greedy precondition for bit-identity: ACTOR_TEMP = SPEC_TEMP = 0 everywhere.
max_output_tokens = 1000
top_p = 1
temperature = 0

# Guess LLM settings
max_guess_output_tokens = 100
# Budget for the speculator when it proposes ACTIONS rather than an imagined
# page (spec_actions_from = "speculator"). 100 tokens is enough for an imagined
# lead paragraph but truncates a thought + k candidate actions.
max_spec_action_tokens = 300
guess_top_p = 1
guess_temperature = 0

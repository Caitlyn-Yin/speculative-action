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

local_api_key = "EMPTY"          # vLLM ignores the value, the SDK requires one
local_seed = 0                   # forwarded to vLLM for reproducibility
local_request_timeout = 600
local_max_retries = 3

prompts_folder = "./prompts/"
prompt_file = "prompts_naive.json"
agent_role = "Question Answering Agent"

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
guess_top_p = 1
guess_temperature = 0

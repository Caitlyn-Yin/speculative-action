# ENV_PREP — local serving environment (Track P, 2026-09-21)

Node `hyin66-agent-0`: 1× H200 NVL 143 GB, 64 cores, 177 GB RAM, **no SLURM**, no sudo.
Rebuild with `scripts/setup_envs.sh` then `scripts/download_ladder.sh`.

## Storage layout

| Path | FS | Free | Use |
|---|---|---|---|
| `/tmp/specmem` | overlay | 86 GB after downloads | envs, HF_HOME, logs, XDG dirs |
| `$HOME` | JuiceFS | 12 GB / 64 GB (82 % full) | repo + docs only — **never caches** |

`scripts/env.sh` is the single source of truth for paths. The pre-existing ES-project cache
(`$HOME/.cache/huggingface`, 21 GB) and envs (`$HOME/micromamba`) were **not touched**.

> **`/tmp` does not survive a pod restart.** Envs and weights are reproducible from the two scripts;
> nothing durable is stored there.

## Environments (P1 — DONE)

| Env | Python | Key versions |
|---|---|---|
| `/tmp/specmem/envs/pipeline` | 3.10.21 | gymnasium **0.29.1** (pinned), numpy 1.26.4, openai 3.16.2, pytest 9.1.1 |
| `/tmp/specmem/envs/vllm` | 3.12 | vllm **0.29.0**, torch 2.13.0+cu130 |

**gymnasium is pinned for a reason.** The repo depends on `gym.Wrapper` attribute forwarding —
`HotPotQAWrapper._get_info` reads `self.steps`/`self.answer` from the inner `WikiEnv`
(`wrappers.py:143-149`), and the runner reads `env.page` / `env.sim_obs`. `Wrapper.__getattr__` was
**removed in gymnasium 1.0**; under 1.3.0 every such access raises `AttributeError`. Verified:
1.3.0 → `AttributeError: 'W' object has no attribute 'page'`; 0.29.1 → forwards with a
`DeprecationWarning`. New code should prefer `env.unwrapped` (as `runner._snapshot_env` does).

## Ladder (P2 — DONE)

`HF_HOME=/tmp/specmem/hf_home`, 56 GB total, all five repos ungated.

| Model | On disk | `config.json` sha256 | revision |
|---|---|---|---|
| Qwen/Qwen3-0.6B | 1.5 G | `660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd` | `c1899de289a04d12100db370d81485cdf75e47ca` |
| Qwen/Qwen3-1.7B | 3.8 G | `1ddb5b89ebc90dcb417a45c213d818577e65976454d29385c8f6140771d95197` | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` |
| Qwen/Qwen3-4B | 7.6 G | `8ba006f74fecfaaeb392872a60f4a480e7ec9860153d2e1b769ec81f9a147f8a` | `1cfa9a7208912126459214e8b04321603b3df60c` |
| Qwen/Qwen3-8B (actor) | 16 G | `f7c4eadfbbf522470667b797a3c89be2524832d2d599797248dc304fff447c30` | `b968826d9c46dd6066d109eabc6255188de91218` |
| Qwen/Qwen3-14B | 28 G | `e73c3664ca09b10a673fef0c22e8a6b456201d49bd4713c9691f775720e8857a` | `40c069824f4251a91eefaf281ebe4c544efd3e18` |

## Serving (P6 — DONE)

`bash scripts/serve_local.sh both` → actor `Qwen3-8B` on `:8000` (gpu_frac 0.60), speculator
`Qwen3-0.6B` on `:8001` (gpu_frac 0.25). Measured: actor ready in 60 s, both resident at
**85.7 GB / 143 GB**. `scripts/serve_local.sh stop` tears them down.

`scripts/check_server.py --all` result — **ALL GREEN**:

```
=== actor: Qwen/Qwen3-8B @ http://127.0.0.1:8000/v1 ===
  [ok] /v1/models -> ['Qwen/Qwen3-8B']
  [ok] LLMClient greedy reply: 'Paris'
  [ok] no <think> leakage
  [ok] deterministic across two identical calls

=== spec: Qwen/Qwen3-0.6B @ http://127.0.0.1:8001/v1 ===
  [ok] /v1/models -> ['Qwen/Qwen3-0.6B']
  [ok] LLMClient greedy reply: 'capital'
  [ok] no <think> leakage
  [ok] deterministic across two identical calls
```

The check goes *through* `hotpotqa/src/llm_client.LLMClient`, not raw HTTP, so it exercises the path
the pipeline uses. `enable_thinking=False` is passed as `chat_template_kwargs`
(`llm_client.py:_local_call`) and `_strip_thinking()` scrubs residue as a second guard.
(The 0.6B reply `'capital'` is a wrong answer, not a failed check — the assertions are mechanical:
reachability, no `<think>`, determinism.)

## Three node-specific traps, all fixed in `scripts/env.sh`

1. **`$HOME/.config` is a root-owned *file*, not a directory.** Anything resolving an XDG path under
   it dies with `NotADirectoryError: [Errno 20] Not a directory: '/home/hyin66/.config/vllm'`.
   Fixed by pointing `XDG_CONFIG_HOME`/`XDG_CACHE_HOME` at the overlay — which also keeps flashinfer's
   cache off the 12 GB home. (Same root cause makes `gh` unusable on this node.)
2. **No CUDA toolkit (`nvcc` absent).** flashinfer JIT-compiles its sampling kernels at startup and
   the engine dies: `/usr/local/cuda/bin/nvcc: not found` → `EngineCore failed to start`. Fixed with
   `VLLM_USE_FLASHINFER_SAMPLER=0`; harmless here since every request is greedy.
3. **conda vs system `libstdc++`.** The serving env ships 6.0.36 (`CXXABI_1.3.15`); the system copy is
   older and wins resolution, breaking `import sqlite3` deep inside vLLM's structured-output imports.
   Fixed by prepending `$VLLM_ENV/lib` to `LD_LIBRARY_PATH` for `$VLLM_PY`.

Also: vLLM 0.29 renamed `--disable-log-requests` → `--no-enable-log-requests`.

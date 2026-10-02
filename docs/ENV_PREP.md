# ENV_PREP — local serving environment (Track P, 2026-09-21)

Node `hyin66-agent-0`: 1× H200 NVL 143 GB, 64 cores, 177 GB RAM, **no SLURM**, no sudo.
Rebuild with `scripts/setup_envs.sh` then `scripts/download_ladder.sh`.

## Storage layout

`scripts/env.sh` is the single source of truth for paths, and it defines **two roots**. The split is
binding, not advisory — `hotpotqa/src/durability.py` aborts any run that would write an output to
the ephemeral one. Rules and rationale: `docs/WAYS_OF_WORKING.md` §2a.

| Variable | Path | FS | Survives recycle | Holds |
|---|---|---|---|---|
| `$SPEC_BASE` | `$HOME/specmem-data` | JuiceFS | **yes** | **every output** — `runs/`, logs, judge + embedding caches, the frozen corpus |
| `$SPEC_SCRATCH` | `/tmp/specmem` | overlay | no | envs, `HF_HOME` weights, XDG dirs, raw corpus download |

Outputs, all under `$SPEC_BASE`:

| Variable | Path | Holds |
|---|---|---|
| `$SPEC_RUNS_DIR` | `$SPEC_BASE/runs` | run artifacts, one subdir per run id — **the only tree `checkpoint.sh` sweeps** |
| `$LOG_DIR` | `$SPEC_BASE/logs` | vLLM server logs, stage logs |
| `$JUDGE_CACHE_DIR` | `$SPEC_BASE/cache/judge` | judge + embedding caches (`docs/JUDGE_CONTRACT.md`) |
| `$WIKI_CACHE_DIR` | `$SPEC_BASE/cache/wiki` | pinned live-Wikipedia responses (`WIKI_CACHE=1`) |
| `$WIKI_DATA_ROOT` | `$SPEC_BASE/local_wiki` | frozen KILT corpus, 7.6 GiB |
| `$HF_EMBED_HOME` | `$SPEC_BASE/hf_embed` | `bge-base-en-v1.5`, 419 MB |

> **`/tmp` does not survive a pod restart**, and it has now cost this project two data losses.
> Envs and weights are reproducible from `setup_envs.sh` + `download_ladder.sh`; **nothing that a
> run produces may be stored there.**

**`$SPEC_BASE` is tight: 64 GB quota, ~8.7 GB free** (the corpus is 7.7 GB of it). It is sized for
jsonl/json outputs, not for weights — which is exactly why the 56 GB ladder stays on scratch. Watch
it with `df -h $HOME`; if outputs start to crowd it, checkpoint and prune rather than relocating to
`/tmp`.

The pre-existing ES-project cache (`$HOME/.cache/huggingface`, 21 GB) and envs
(`$HOME/micromamba`) were **not touched**.

## Environments (P1 — DONE)

| Env | Python | Key versions |
|---|---|---|
| `$SPEC_SCRATCH/envs/pipeline` | 3.10.21 | gymnasium **0.29.1** (pinned), numpy 1.26.4, openai 3.16.2, pytest 9.1.1 |
| `$SPEC_SCRATCH/envs/vllm` | 3.12 | vllm **0.29.0**, torch 2.13.0+cu130 |

**gymnasium is pinned for a reason.** The repo depends on `gym.Wrapper` attribute forwarding —
`HotPotQAWrapper._get_info` reads `self.steps`/`self.answer` from the inner `WikiEnv`
(`wrappers.py:143-149`), and the runner reads `env.page` / `env.sim_obs`. `Wrapper.__getattr__` was
**removed in gymnasium 1.0**; under 1.3.0 every such access raises `AttributeError`. Verified:
1.3.0 → `AttributeError: 'W' object has no attribute 'page'`; 0.29.1 → forwards with a
`DeprecationWarning`. New code should prefer `env.unwrapped` (as `runner._snapshot_env` does).

## Ladder (P2 — DONE)

`HF_HOME=$SPEC_SCRATCH/hf_home`, 56 GB total, all five repos ungated.

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

---

## Frozen local Wikipedia (added 2026-09-29)

Full design, fidelity measurement and deviations: **`docs/LOCAL_WIKI.md`**. Paths only, here,
because this file is the environment record.

Persistent storage, `$HOME` / JuiceFS — survives a pod recycle, 7.6 GiB:

```
$WIKI_DATA_ROOT = $SPEC_BASE/local_wiki          # = $HOME/specmem-data/local_wiki
$WIKI_DATA_DIR  = $WIKI_DATA_ROOT/kilt_20190801
    pages.zst  5.14 GiB   pages.sqlite  639 MiB   leads.jsonl.zst  537 MiB
    bm25/      1.30 GiB   MANIFEST.json
$WIKI_DATA_ROOT/aux
    hotpot_dev_distractor.parquet     27 MB, gold supporting facts
    fidelity_live_cache/              318 pinned live responses
```

Scratch, container overlay — wiped by a recycle, re-downloadable in ~13 min:

```
$WIKI_RAW_DIR = $SPEC_SCRATCH/kilt_raw
    kilt_knowledgesource.json   34.76 GiB   build/   (staging; see LOCAL_WIKI.md §4)
```

Source of record, all three also in `MANIFEST.json`:

| | |
|---|---|
| URL | `http://dl.fbaipublicfiles.com/KILT/kilt_knowledgesource.json` |
| size | 37,318,876,722 B |
| md5 | `d1dca62aa6ba889d2e842182e3114af5` (matches the publisher's S3 metadata) |
| sha256 | `f966d6f09c4ff91656db5c56c384f136b0c495c7083c043586b8cb1033c389a5` |
| pages | 5,903,530 |

Rebuild (idempotent; a stage marked complete in `MANIFEST.json` is skipped):

```bash
source scripts/env.sh
$PIPELINE_PY -m pip install zstandard bm25s PyStemmer pyarrow
$PIPELINE_PY scripts/build_local_wiki.py --stage all      # ~27 min total
```

**Extra pipeline-env dependencies**: `zstandard` 0.25.0, `bm25s` 0.3.11, `PyStemmer`,
`pyarrow`. These are now installed by `setup_envs.sh` itself, so the manual `pip install`
above is only needed on an env built before 2026-09-29.

### Trap 4 — `$HOME/.config` is a JuiceFS mount-root artifact

Trap 1 above is now understood rather than merely worked around: `$HOME` is the **root of a
JuiceFS mount**, and JuiceFS materialises `.accesslog`, `.stats` and `.config` as root-owned
virtual files there on every mount. `~/.config` is a 0400 root-owned regular file
permanently, on every pod. Not repairable, not a cron artifact. (Recorded in
`STATE_RESUME.md` §12; repeated here because this is the environment file.)

### Trap 5 — `import sqlite3` in the *pipeline* env, order-dependently

Same family as trap 3, opposite env. `_sqlite3` in the pipeline env pulls
`libicui18n.so.78`, which needs `CXXABI_1.3.15` that the system `libstdc++` lacks:

```
ImportError: /lib/x86_64-linux-gnu/libstdc++.so.6: version `CXXABI_1.3.15' not found
             (required by .../envs/pipeline/lib/python3.10/lib-dynload/../.././libicui18n.so.78)
```

It only fires when something else has already loaded the system `libstdc++`, so it is
**order dependent**: `pytest tests/test_local_wiki.py` passed while `pytest tests/` failed
on the same tree. `scripts/env.sh` now prepends the pipeline env's `lib` to
`LD_LIBRARY_PATH` globally. Harmless for `$VLLM_PY`, which prepends its own `lib` ahead of
it.

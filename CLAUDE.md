# CLAUDE.md — orientation for the speculative-action repo

Read this first. It describes **what exists on this node right now**, what is deliberately
absent, and what the live blockers are. It is a map, not a spec — the binding process rules
live in `docs/WAYS_OF_WORKING.md`, and the authoritative state record is `docs/STATE_RESUME.md`.

---

## 1. What this repo is

Implementation of **Speculative Actions** (paper: https://arxiv.org/pdf/2510.04371): a fast
**Speculator** model predicts the agent's likely next action(s) while a slower authoritative
**Actor** computes the real one. When they agree, work pre-launched during the wait is kept —
the user sees sequential, lossless behaviour, just faster. Top-k speculation widens the hit rate.

Four independent environments, each self-contained (run commands **from inside** the env dir):

| Dir | Domain | Losslessness | Status on this node |
|---|---|---|---|
| `hotpotqa/` | ReAct multi-hop web search | lossless | **the active work area** — modified, under research |
| `chess-game/` | turn-based game play (TextArena) | lossless | pristine upstream, untouched |
| `e-commerce/` | τ-bench retail tool-calling | lossless | pristine upstream, untouched |
| `os-tuning/` | Linux scheduler knob tuning | **lossy** (last-write-wins) | pristine upstream; needs sudo + debugfs, not runnable here |

Only `hotpotqa/` has been changed by us. The other three are vendor/upstream trees and are
**out of scope** unless explicitly asked — don't grep-count across them when auditing our work.

---

## 2. `hotpotqa/` — the active environment

~1.7 kLOC of Python. Entry point `run.py`; library in `src/`.

| File | Role |
|---|---|
| `src/runner.py` (354 L) | `HotPotQARun.webthink()` — the ReAct loop. Speculation is the `simulate=True` branch (`step()`, ~L85-97). Owns the isolation snapshot/restore. |
| `src/environment.py` (175 L) | `WikiEnv`. Real Wikipedia via `search_step()` (L100). **Speculated observations via `guess_step()` (L86) ask the speculator LLM to _imagine_ the page** — no real HTTP call. `get_page_obs()` truncates to 5 sentences. |
| `src/llm_client.py` (161 L) | Two backends behind `constants.llm_backend`: `"local"` (OpenAI-compatible vLLM — the only one permitted for research runs) and `"external"` (Gemini/OpenAI/OpenRouter, lazily imported, kept reachable). Forces `enable_thinking=False` + `_strip_thinking()` for Qwen3. |
| `src/constants.py` (52 L) | All config. Greedy (`temperature=0`, `guess_temperature=0`), `guess_num_actions=3` (k=3), actor `Qwen3-8B` @ `:8000`, speculator `Qwen3-0.6B` @ `:8001`, `isolate_speculation=True`. |
| `src/wrappers.py` (278 L) | `HotPotQAWrapper` (EM/F1 + dataset), `LoggingWrapper` (writes `trajs/*.json`), `HistoryWrapper`. |
| `src/metrics.py` (199 L) | Action-match accuracy only: top-1 / top-k exact string match of speculated vs realized action, split by Search/Lookup/Finish. **This is the whole scoring layer that exists.** |
| `src/{prompts,utils,grapher}.py` | ReAct prompt templates, IO helpers, plots. |
| `tests/test_isolation_regression.py` | 5 tests, offline, scripted Wikipedia + speculator. ~0.7 s. |
| `data/`, `prompts/` | HotpotQA dev/test/train simplified, `prompts_naive.json`. |
| `run_metrics/` | **Old external-API results** (gpt-4 / gpt-5 / gemini-2.5-flash × top1/top3). Not ours, not local-model, **not comparable** to anything we produce. |

### The harness bug and its fix (the one real piece of research code we own)

Upstream ran the speculative branch against the **same `env` object** as the realized
trajectory, and `guess_step()` assigns `self.page`/`self.obs` unconditionally even under
`simulate=True`. A speculation therefore overwrote the real page, and the damage surfaced at
the next realized `lookup[...]` (which reads `self.page` via `construct_lookup_list()`).

Fixed in `034dabd`: `runner._snapshot_env` / `_restore_env` around the speculative step,
covering `page, obs, sim_obs, lookup_keyword, lookup_list, lookup_cnt`. Upstream behaviour
stays reachable via `constants.isolate_speculation = False` — that flag **is** the unisolated
arm of the online gate, not dead code.

**Two residual contamination channels are known and deliberately NOT fixed** (they change
recorded-trajectory semantics). With isolation ON, a speculative step still:
1. appends to `LoggingWrapper.traj` (`wrappers.py:258-264`) — actions/obs `1/2 → 2/3` per speculation;
2. increments `WikiEnv.steps` (`environment.py:166`) — `1 → 2`.

Neither breaks bit-identity of the realized sequence (`webthink` builds its own `running_prompt`),
but `trajs/*.json` and `info["steps"]` are polluted. **Needs a ruling before Phase C.**

---

## 3. What is ABSENT — and why that matters

The research layer was built on a GH200 host, never pushed, and is **gone**. See
`docs/LOST_WORK_MANIFEST.md` for the item-by-item inventory. Missing, with no recoverable source:

- `gates.py` — the typed deterministic battery (`channel`, `stale_intent`, `spec_fixation`, `agent_loop`)
- `rescore.py` — EM → Norm → battery → SE-v2 → overlay pipeline
- `JUDGE_CONTRACT.md`, `AUDIT_PROTOCOL.md`, `se_judge_v2` rules a–d
- Trajectory schema 0.2.0 (`obs_excerpt`, token capture), human overlays v1/v2, the 7107/7 withdrawal
- The Phase B "exactly flat ladder" result (29 windows) — fixture `runs/phase0_spec5/` never committed ⇒ **unverifiable, must not be cited**

**These are decisions, not code.** Re-deriving them produces *a* contract, not *the* contract,
and any number scored under a rewrite is not comparable to a pre-loss number. Do not silently
re-invent them — that is an open ruling for the PI (see §6).

---

## 4. Runtime state — the pod has been recycled (again, 2026-10-02)

`/tmp` does not survive a pod restart, and **it has restarted twice**. Verified 2026-10-02:

```
/tmp/specmem            → does not exist
nvidia-smi              → H200 NVL, 0 MiB / 143771 MiB used
:8000 (actor)           → DOWN
:8001 (speculator)      → DOWN
```

**The 2026-10-02 recycle also destroyed all Paper B data**, which was under `/tmp/specmem/paperb`:
both pilot runs (both backends), the probe/diagnosis JSONs, and the 100-question scale-up that was
still running. Only the numbers transcribed into `docs/PAPERB_PILOT_REPORT.md` survive. Nothing is
recoverable; the stages have to be re-run. See §4a.

So P1/P2/P6 (envs, weights, servers) are **not currently materialised**, even though their
commits are on the branch. Everything is reproducible from committed scripts:

```bash
bash scripts/setup_envs.sh      # both envs on $SPEC_SCRATCH ('pipeline' arg = repo deps only)
bash scripts/download_ladder.sh # Qwen3 0.6B/1.7B/4B/8B/14B → HF_HOME on the overlay (~56 GB)
bash scripts/serve_local.sh both  # actor 8B :8000 (gpu 0.60) + spec 0.6B :8001 (gpu 0.25)
$PIPELINE_PY scripts/check_server.py --all   # goes through LLMClient, not raw HTTP
bash scripts/serve_local.sh stop
bash scripts/checkpoint.sh        # ALWAYS last: run artifacts -> results/ -> origin
```

`scripts/env.sh` is the **single source of truth for paths**. It defines **two roots** —
`$SPEC_BASE` (`$HOME/specmem-data`, JuiceFS, persistent, **every output**) and `$SPEC_SCRATCH`
(`/tmp/specmem`, overlay, ephemeral, **read-only inputs and rebuildable tooling only**). The split
is enforced by `hotpotqa/src/durability.py`, which every entry script imports and which aborts a
run whose output path is ephemeral. Rules: `docs/WAYS_OF_WORKING.md` §2a.

It also encodes five node-specific traps — read it before debugging any startup failure:

1. `$HOME/.config` is a **root-owned regular file**, not a directory → XDG redirected to the overlay.
   (Same root cause breaks `gh`, which breaks `git push` — see §6.)
2. No CUDA toolkit (`nvcc` absent) → `VLLM_USE_FLASHINFER_SAMPLER=0`. Harmless: everything is greedy.
3. conda vs system `libstdc++` → prepend `$VLLM_ENV/lib` to `LD_LIBRARY_PATH` for `$VLLM_PY`.
4. `$HOME` is JuiceFS, 64 GB quota, **~8.7 GB free** → **never put caches or weights there**, but
   **always put outputs there** (that is `$SPEC_BASE`). The ES project's
   21 GB cache at `$HOME/.cache/huggingface` is a separate project — leave it alone.

**gymnasium is pinned to 0.29.1 on purpose.** The code relies on `gym.Wrapper.__getattr__`
forwarding (`wrappers.py:143-149` reads `self.steps`/`self.answer`; the runner reads `env.page`).
That was removed in gymnasium 1.0 and raises `AttributeError` under 1.3.0. New code should prefer
`env.unwrapped`, as `_snapshot_env` does.

Wikipedia and HuggingFace were both reachable (HTTP 200) as of the last check.

## 4a. What the 2026-10-02 recycle destroyed

Verified by `find` over `$HOME` and `/tmp` on 2026-10-02: **zero** Paper B output artifacts exist
anywhere on this node. Not "stale" — absent.

| Gone | Was at | Recoverable? |
|---|---|---|
| Pilot run 1 corpora (cache-on), both backends | `/tmp/specmem/paperb/pilot25_*` | no — re-run |
| Pilot run 2 corpora (cache-off), both backends | `/tmp/specmem/paperb/pilot25b_*` | no — re-run |
| 100-question scale-up, in flight | `/tmp/specmem/paperb/scale100_*` | no — re-run |
| `probe_*.json`, `diag_nondet_*.json` | `/tmp/specmem/paperb/` | no; numbers transcribed in `PAPERB_PILOT_REPORT.md` §4–5 |
| Invariant-gate artifacts | `/tmp/specmem/invariant` | no; verdicts transcribed in `docs/INVARIANT_REPORT*.md` |
| Envs, 56 GB weights, logs, XDG | `/tmp/specmem/{envs,hf_home,logs,xdg}` | **yes** — `setup_envs.sh`, `download_ladder.sh` |

Survived, on `$SPEC_BASE` (JuiceFS):

| Kept | Path | Size |
|---|---|---|
| Frozen KILT corpus + BM25 + MANIFEST | `$SPEC_BASE/local_wiki/kilt_20190801` | 7.7 GB |
| `aux/` (gold parquet, 318 pinned live responses) | `$SPEC_BASE/local_wiki/aux` | — |
| Embedding weights | `$SPEC_BASE/hf_embed` | 419 MB |

The judge and embedding caches were **never populated** — `hotpotqa/cache/judge/` was created on
2026-09-29 and is empty, because `score_criteria.py` has never been run on real data.

This is what `src/durability.py` and `scripts/checkpoint.sh` exist to prevent recurring.

---

## 5. Repo conventions

- Branch: `phase2-fixed-scaleup` (off `main` @ `dc938b9`). Tag `trackP-complete-2026-09-21` = `ebe9979`.
- One commit per work item, prefixed with its item id (`P3: local vLLM backend …`).
- `docs/` is the memory: `STATE_RESUME.md` (authoritative state), `ENV_PREP.md` (environment),
  `LOST_WORK_MANIFEST.md` (what's gone), `WAYS_OF_WORKING.md` (standing rules).
- **"Done means pushed."** Local commits are work-in-progress no matter how finished they look.
  Every report ends with the *pushed* hash, or says plainly that nothing was pushed and names
  the blocker. This rule exists because of the 2026-09 loss.
- Run tests from `hotpotqa/`: the package loads data by relative path (`conftest.py` chdirs).

---

## 6. Live blockers — read before planning anything

1. ~~**`git push` fails.**~~ **CLOSED 2026-09-28.** Pushing works over SSH with the repo deploy key
   (`git@github.com-specmem`); `gh`/HTTPS was abandoned, not repaired, because `$HOME/.config` is a
   root-owned JuiceFS mount-root artifact. Procedure: `docs/WAYS_OF_WORKING.md` §3.
   `origin/phase2-fixed-scaleup` and the tag are both present. Verify after any pod restart with
   `ssh -T git@github.com-specmem`.
   *Note:* that check currently answers `Hi Caitlyn-Yin!` rather than
   `Hi Caitlyn-Yin/speculative-action!`, i.e. the key is registered **account-wide, not as a repo
   deploy key**. It pushes fine, but with a wider blast radius than §3 intends.
2. **GH200 unreachable** — no mount, no SSH key, no agent, no SLURM. Third independent check
   failed. `/home/hyin66/salvage_speculative_action.sh` exists but must be run *on the GH200*
   by a human; it has never been executed anywhere.
3. **Open ruling — §6 obs-identity.** Phase D's observation-join must classify a speculated
   `search[X]` whose observation was *LLM-authored* (`environment.py:85-97`). "obs-identical"
   will essentially never hold verbatim against an imagined page, so the classifier has to be
   defined over the *realized* observation. The lost `JUDGE_CONTRACT.md` presumably fixed this.
   **Not to be re-decided unilaterally.**
4. **Open ruling — judge contract source.** Salvage from the GH200 (items 1–7 return intact) vs
   re-derive here (items 1–7 are re-decided; pre-loss numbers become incomparable).
5. **Residual contamination channels** (§2) need a ruling before Phase C.
6. Single H200, no SLURM: the two-server split is measured and works (85.7 / 143 GB), but
   everything runs foreground and three-arm gates are sequential.

---

## 7. Where the work resumes

Nothing downstream can start until the environment is re-materialised (§4) and the rulings in §6
land. Concretely, in dependency order:

1. Re-run `setup_envs.sh` → `download_ladder.sh` → `serve_local.sh both` → `check_server.py --all`.
   Cheap, fully scripted, no decisions involved.
2. Resolve the push blocker, or get an off-node destination. Until then every result is fragile
   in exactly the way that caused the last loss.
3. **Step 2 online isolation gate** (A3(ii), 3-arm, 5 questions, real servers) — blocked on the
   lost gate definition, not on code.
4. **Step 2/2.5 rewrite** (`gates.py`, SE-v2 judge, `rescore.py`, contracts) — blocked on ruling #4.
5. **Phase C scale-up + model ladder** — blocked on 1 and 3.
6. **Phase D obs-join** — blocked on a phase-2 corpus and ruling #3.

Config still carrying inherited (not chosen) values: `n_samples_to_run = 20` (the brief said 25),
`random_seed = 248`, `n_steps_to_run = 8`. These are run-protocol parameters and were left
untouched on purpose.

# STATE_RESUME — 2026-09-21

**Node:** `hyin66-agent-0` · 1× H200 NVL 143 GB · no SLURM · repo at `/home/hyin66/speculative-action`
**Branch:** `phase2-fixed-scaleup` (from `main` @ `dc938b9`)

---

## 0. Reconstruction basis — the docs did NOT survive

The Step 0 brief assumes "code + docs survived" and that `docs/` is our memory. **That premise is false
on this node and on the remote.** Evidence gathered 2026-09-21:

- `git ls-remote origin` → exactly two refs: `HEAD` and `refs/heads/main`, both `dc938b9ef747…`.
  No `phase0-recon`, `phase1-gates`, `research/salvage-2026-09-20`, no tags.
- `git log --oneline --all` → single lineage ending at `dc938b9` "update: fixes" (2026-03-09).
  `git reflog` has one entry: `clone` on 2026-09-17 22:40. No stashes, no worktrees.
- `git ls-files '*.md'` → only upstream READMEs (root, `hotpotqa/`, textarena/tau-bench env docs).
  **None of** `RECON.md`, `HARNESS_BUG_SIM_CONTAMINATION.md`, `STEP2_5_REPORT.md`,
  `JUDGE_CONTRACT.md`, `AUDIT_PROTOCOL.md`, `PHASE1_RESULTS.md` exists in any ref.
- Home-wide search for those filenames, plus `serve_local.sh`, `check_server.py`,
  `gates.py`, `rescore.py`, `research_logging.py`, `test_gate_regression*` → **zero hits**.
- No salvage bundle or tarball from the GH200 host ever landed here
  (`*.bundle`, `*.tgz` absent; nothing new in `$HOME` since 2026-09-20 but the salvage script itself).

So this document is reconstructed from the **only** surviving evidence: the upstream fork's code, plus
the prior session's state report. Every claim below is a citation into code that exists on this node,
not a recollection of the lost docs.

**Consequence:** the lost docs recorded the *decisions* (judge contract, audit protocol, schema 0.2.0
field semantics, contamination analysis, the 29-window Phase B ladder). Code can be rewritten; those
decisions cannot be recovered from code, and any rewrite will silently re-decide them.

---

## 1. What is implemented

**Nothing of the research layer.** The working tree is the pristine upstream fork. Inventory of the
research components named in the brief:

| Component | Status | Evidence |
|---|---|---|
| `gates.py` deterministic battery (stale_intent / tool_channel / terminal_channel) | **ABSENT** | no such file in any ref |
| Isolation wrapper (snapshot/restore) | **ABSENT** — and the bug it fixed is still live, see §2 | `hotpotqa/src/runner.py:203`, `hotpotqa/src/environment.py:96` |
| Trajectory schema 0.2.0 (token capture, `obs_excerpt`) | **ABSENT** | schema is upstream's ad-hoc dicts, `hotpotqa/src/wrappers.py` |
| `rescore.py` (EM → Norm → battery → SE-v2 → overlay) | **ABSENT** | no scoring pipeline beyond upstream EM/F1, `hotpotqa/src/metrics.py` |
| Human overlays (`human_audit_v1.jsonl`), judgment caches v1/v2 | **ABSENT** | no `runs/` directory at all |
| `scripts/serve_local.sh`, `check_server.py` | **ABSENT** | no `scripts/` directory |
| `tests/`, `test_gate_regression` | **ABSENT** | no `tests/` directory |

What *does* exist (upstream baseline, `hotpotqa/` only):

- `run.py` — CLI over hardcoded external model lists (`run.py:13-26`).
- `src/runner.py` — ReAct loop `webthink()` (`runner.py:153`); speculation is the `simulate=True`
  path (`runner.py:179-184`, `202-216`); k = `constants.guess_num_actions` = 3 (`constants.py:16`).
- `src/environment.py` — `WikiEnv`; real Wikipedia via `search_step()` (`environment.py:99-124`),
  speculated observations via `guess_step()` (`environment.py:85-97`).
- `src/metrics.py`, `wrappers.py`, `prompts.py`, `grapher.py`, `utils.py`.
- `data/` — hotpot dev/test/train simplified + `paper_dev.jsonl`. `prompts/prompts_naive.json`.
- `run_metrics/` — old external-API results (gpt-4, gpt-5, gemini-2.5-flash × top1/top3). Not ours,
  not local-model, not comparable.
- `trajs/4553135.json` — one stray trajectory.

---

## 2. The harness bug is still present (verifiable, not remembered)

The contamination bug the lost `HARNESS_BUG_SIM_CONTAMINATION.md` described is **reproducible in the
current code**, which independently corroborates that memory and confirms the Phase A fix is absent:

- `runner.py:203-205` runs the speculative branch against **`self.env`** — the same object the realized
  trajectory steps through at `runner.py:191`. There is no snapshot, no copy, no restore.
- `environment.py:85-97` (`guess_step`) assigns **`self.page`** (line 96) and **`self.obs`** (line 97)
  unconditionally, including when `simulate=True` — the `simulate` flag only additionally sets
  `self.sim_obs` (line 95). So a speculation overwrites the page state of the real environment.
- The damage surfaces at the next realized `lookup[...]`: `construct_lookup_list()` reads `self.page`
  (`environment.py:64-73`), which is now the *speculated* page. This is exactly why the online
  isolation gate requires a realized `lookup[]` at turn ≥ 2 — without one the unisolated arm can look
  green while still being contaminated.

**Implication for Step 2:** the unisolated arm is not merely "still switchable" — it is the *current
default*, since no isolated path exists yet. The gate cannot be run until the isolated path is written.

---

## 3. Fixture status → fixture tests SKIPPED

`runs/phase0_spec5/` is **not in git** (no `runs/` directory in any ref; no `MANIFEST.sha256`).
Per the brief's instruction:

> **Fixture tests: SKIPPED.**
> **Reason:** the `phase0_spec5` frozen fixture was never committed to this fork and no copy exists on
> this node or on `origin`; the artifacts lived only on the GH200 host, which is unreachable from here
> (no `/projects` mount, no SSH key, no agent — checked 2026-09-20 and 2026-09-21). Bit-identity
> regression against the frozen-29 corpus therefore cannot be asserted.

Forward work uses the new fixed-harness baseline anyway, so this does not block Phase C — but it does
mean **the Phase B "exactly flat ladder" result is now unverifiable** and must not be cited as an
established number. It survives only as an unreproduced claim.

---

## 4. Environment — verified today

| Check | Result |
|---|---|
| `https://en.wikipedia.org` | **HTTP 200**, 0.24 s |
| `https://en.wikipedia.org/w/index.php?search=…` (the exact URL form at `environment.py:101`) | **HTTP 200**, 0.44 s |
| `https://huggingface.co` | **HTTP 200**, 0.14 s |
| Proxy env vars | none set |

**Wikipedia is reachable → the Step 1 STOP condition is NOT triggered.**

Models — the requested ladder is *not* on the local read-only `/models` mount (69 repos; it carries
Qwen2.5, Qwen3.5, Qwen3.6, Qwen3.8 families, but **no `Qwen3-*`**). All five requested repos exist on
HF and are ungated: `Qwen3-0.6B` / `1.7B` / `4B` / `8B` / `14B`. Download ≈ 57 GB at bf16.

Disk — **`HF_HOME` must go on `/` (overlay, 152 GB free), not `$HOME`** (JuiceFS, 64 GB total,
**12 GB free, 82 % used**). An existing 21 GB HF cache already sits in `$HOME/.cache/huggingface`
from the ES project; it is a prime candidate for eviction or relocation if space gets tight.

Python — no vLLM in system python (3.10.12). Only the ES project's `micromamba/envs/{es,verl}` have
`torch`+`vllm`; neither is a spec-action env. Both envs in the brief (pipeline venv + separate serving
env) still need to be created.

---

## 5. Config drift vs the brief (must be changed before any run)

| Setting | Current code | Brief requires |
|---|---|---|
| Agent temperature | `1` (`constants.py:23`) | `ACTOR_TEMP=0` |
| Speculator temperature | `0.1` (`constants.py:28`) | `SPEC_TEMP=0` |
| LLM backend | Gemini / OpenAI / OpenRouter only (`llm_client.py:1-21`) | local vLLM only, no external APIs |
| Samples | `n_samples_to_run = 20` (`constants.py:12`) | 25 dev questions |
| k | `guess_num_actions = 3` (`constants.py:16`) | k=3 ✓ (already matches) |
| Seed | `random_seed = 248` (`constants.py:9`) | "fixed seed" — 248 is the inherited value |
| Steps/episode | `n_steps_to_run = 8` (`constants.py:11`) | unspecified |

Also note `runner.run()` imports `google.genai.errors` at `runner.py:229` — the external-API
dependency is load-bearing in control flow, not just in the client.

---

## 6. Semantic flag for Phase D (needs your ruling)

Upstream speculation does **not** issue a real Wikipedia call — `guess_step()` asks the speculator LLM
to *imagine* the page (`environment.py:85-97`), and `get_page_obs()` truncates it to the first 5
sentences (`environment.py:75-83`). So a "speculated `search[X]`" has an LLM-authored observation.

Phase D's observation-join compares a speculated `search[X]` against the real observation for the same
normalized `X`. That is well-defined here — but "obs-identical" will essentially never hold verbatim
against an imagined page, so the join's classifier must be defined over the *realized* observation
attached to the action, not over the speculated text. The lost `JUDGE_CONTRACT.md` presumably fixed
this. **I will not re-decide it unilaterally.**

---

## 7. Pending phases

| Phase | State |
|---|---|
| Step 1 recon | docs lost; superseded by this file |
| Step 2 gates (`gates.py` battery) | **must be rewritten from scratch** |
| Step 2.5 judge-v2 + battery + `JUDGE_CONTRACT` / `AUDIT_PROTOCOL` | **must be rewritten; contracts lost** |
| Phase A isolation fix + schema 0.2.0 | **must be rewritten**; bug still live (§2) |
| Phase B flat ladder (29 windows) | result unverifiable (§3); fixture gone |
| **Step 2 online isolation gate** | blocked — no isolated path to test |
| **Step 3 Phase C scale-up + ladder** | blocked — no fixed harness, no local serving |
| **Step 4 Phase D obs-join** | blocked — no phase2 corpus; §6 needs a ruling |

---

## 8. BLOCKERS

1. **All research code and all docs are gone.** Steps 2–4 depend on a fixed harness, a battery, a
   judge contract and an audit protocol that do not exist on this node. Only the GH200 host has them,
   and it is unreachable from this pod.
2. **Rebuilding is a rewrite, not a reconstruction.** Re-deriving `gates.py`, the SE-v2 judge contract
   and the audit protocol means re-deciding scoring semantics — which the brief's own discipline
   section says I must ask about first.
3. No SLURM and a single H200: the two-server split (actor :8000 + speculator :8001 with `GPU_FRAC`)
   is still feasible on 143 GB, but everything runs foreground; three-arm gates are sequential.

# STATE_RESUME — 2026-09-21

**Node:** `hyin66-agent-0` · 1× H200 NVL 143 GB · no SLURM · repo at `/home/hyin66/speculative-action`
**Branch:** `phase2-fixed-scaleup` (from `main` @ `dc938b9`)

> **Update 2026-09-21 (Track S / Track P).** Track S (salvage) **FAILED — the STOP condition in S4
> fired**; see §9. Track P (docs-independent prep) is **COMPLETE, 6/6**; see §10. Sections 0–8 below
> are the original reconstruction and remain accurate, with two corrections noted inline in §10.

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
   *(Resolved 2026-09-21 — the split is no longer hypothetical, it is measured: §10 / P6.)*

---

## 9. Salvage attempt 2 — FAILED (S4 STOP condition fired)

**Outcome: no bundle, no checksums, nothing transferred. S1–S3 and S5 were never executable.**

Per S4 this is recorded as the failure mode, and **no semantics were re-derived**: `gates.py`, the
judge, `rescore`, `JUDGE_CONTRACT`, `AUDIT_PROTOCOL` and the Step 2 gate definition remain untouched.

**Failure mode: the GH200 host is not reachable from this pod — there is no network path to attempt.**
S1 presupposes a shell on the GH200; S2 presupposes `scp`/`rsync` *from* it. Neither is available.
Checked again on 2026-09-21 (third independent check, same result):

| Probe | Result |
|---|---|
| `/projects/bhll/hyin6/code/speculative-action` | does not exist |
| `/projects`, `/project`, `/proj`, `/mnt/projects`, `/scratch`, `/work`, `/gpfs`, `/lustre`, `/nfs`, `/data` | none exist |
| non-virtual mounts | container overlay, `/home/hyin66` (JuiceFS rw), `/models` (JuiceFS **ro**), pod binds off `/dev/md0p1` — nothing from that cluster |
| private keys under `$HOME` | **none** (`~/.ssh/` holds only `authorized_keys`, `config`, `known_hosts`) |
| `SSH_AUTH_SOCK` / `ssh-add -l` | unset / "Could not open a connection to your authentication agent" |
| `~/.ssh/config` host aliases | only `worker-*` and `*.efabric-workspace.svc.cluster.local` (in-cluster EFabric peers, not a login node) |
| `known_hosts` | one hashed entry |
| SLURM (`sacct`, `sbatch`, `squeue`, `sinfo`) | absent — job **2652592 / 2662459** accounting is unreachable |
| VPN / transport clients | `ssh`, `rsync`, `scp` present but with nothing to authenticate; no `sshfs`, `globus`, `openvpn`, `tailscale` |
| `gh auth status` | fails — `$HOME/.config` is a root-owned *file*, so `gh` cannot read its config |
| bundle/tarball delivered by other means | none — no `*.bundle`/`*.tgz` anywhere, nothing new in `$HOME` since 2026-09-20 |
| `git ls-remote origin` | still only `HEAD` + `refs/heads/main`, both `dc938b9ef747…` |

**Checksums: none to record.** No artifact was produced or received, so there is nothing to verify.

**What exists instead:** `/home/hyin66/salvage_speculative_action.sh` (written 2026-09-20, ~9 KB,
**never executed anywhere**) implements S1-style collection plus commit/tag/push with a
bundle+tarball fallback. It is untested. To salvage, it must be run **on the GH200**, by a human with
a shell there.

**Per S4: switch to full rebuild next session** unless a route to the GH200 appears. Concretely, the
next session needs one of: (a) an SSH alias + key for the GH200, (b) someone running that script
there and attaching the outputs, or (c) an explicit go-ahead to re-derive the lost semantics — which
is a rewrite, and re-decides the judge contract and audit protocol.

---

## 10. Track P — prep independent of the lost docs: COMPLETE (6/6)

Full environment detail in `docs/ENV_PREP.md`. One commit per item on `phase2-fixed-scaleup`.

| Item | Status | Commit | Evidence |
|---|---|---|---|
| P1 envs | **DONE** | `e71d3ee` | pipeline py3.10.21 (gymnasium **0.29.1** pinned, numpy 1.26.4, openai 3.16.2); serving py3.12 (**vllm 0.29.0**, torch 2.13.0+cu130). `HF_HOME` on `/tmp/specmem`; ES cache untouched |
| P2 ladder | **DONE** | `83f5173` | 56 GB, 5/5 repos, sizes + `config.json` sha256 + revisions in `ENV_PREP.md` |
| P3 local backend | **DONE** | `7bc501d` | `backend={local,external}`, per-role `base_url`, lazy external SDKs, guarded `google.genai` import |
| P4 greedy | **DONE** | `5f97df2` | `temperature 1→0`, `guess_temperature 0.1→0`, `guess_top_p 0.9→1` |
| P5 isolation + test | **DONE** | `034dabd` | **5/5 tests pass in 0.70 s** — see below |
| P6 serving | **DONE** | `cdd831b` | both servers up, `check_server.py --all` → **ALL GREEN** |

### P5 regression test result

`hotpotqa/tests/test_isolation_regression.py` — **5 passed, 0.70 s**, no network and no LLM
(Wikipedia and the speculator are both scripted):

| Test | Result | What it establishes |
|---|---|---|
| `test_realized_lookup_is_identical_when_isolated` | PASS | realized `lookup[]` at turn 2 is **byte-identical** with speculation on vs off |
| `test_realized_lookup_diverges_when_unisolated` | PASS | with `isolate_speculation=False` the same comparison **diverges** — so the green above is not a blind pass |
| `test_snapshot_restores_every_declared_field` | PASS | all of `page, obs, sim_obs, lookup_keyword, lookup_list, lookup_cnt` restored |
| `test_unisolated_arm_clobbers_state[page]`/`[obs]` | PASS | pins exactly what upstream corrupts (`environment.py:96-97`) |

This is the **offline** invariant on a scripted fixture. It is *not* the A3(ii) online gate, which
still needs real servers, 5 questions and three arms — and which remains blocked on the lost gate
definition.

**Two residual contamination channels, found but deliberately NOT fixed** (they change
recorded-trajectory semantics, which Track P is not permitted to touch). With isolation ON, a
speculative step still:

1. appends to `LoggingWrapper.traj` (`wrappers.py:258-264`) — measured actions/observations
   `1/2 → 2/3` for a single speculation;
2. increments `WikiEnv.steps` (`environment.py:165`) — measured `1 → 2`.

Neither alters the realized action/observation sequence (`webthink` builds its own `running_prompt`
and loops on its own index), so the bit-identity invariant holds — but `trajs/*.json` and
`info["steps"]` are polluted. **Needs a ruling before Phase C**, since it affects what a captured
trajectory means.

### Corrections to §1 and §5 from Track P work

- §1 said the research layer is absent. Still true — but P3/P5 mean `hotpotqa/src/` is **no longer
  pristine upstream**: `llm_client.py`, `runner.py`, `environment.py`, `constants.py` now carry the
  local backend, the greedy config and the isolation fix. The *research* layer (gates, judge,
  rescore, overlays, schema 0.2.0) is still absent.
- §2 said "the Phase A fix is absent." **Now partially present**: the minimal correctness fix landed
  in `034dabd`. It is not the schema 0.2.0 wrapper, and upstream behaviour stays reachable via
  `constants.isolate_speculation = False` (the gate's unisolated arm).
- §5's config-drift table is resolved for temperature/backend rows; sample count (20 vs 25) and the
  seed remain as inherited — untouched, because they are run-protocol parameters.

### Open items carried forward (unchanged, not re-decided)

- §6 obs-identity vs speculated text — **still open**, deliberately not decided.
- Everything in §7 that depends on the lost gate/judge/battery definitions.
- The two residual contamination channels above.

---

## 11. R0 (push to origin) — FAILED, 2026-09-21

**`origin` is unchanged: `HEAD` and `refs/heads/main`, both `dc938b9ef747…`. No branch, no tag.**

```
$ git push -u origin phase2-fixed-scaleup
failed to create root command: failed to read configuration:
    open /home/hyin66/.config/gh/config.yml: not a directory
remote: No anonymous write access.
fatal: Authentication failed for 'https://github.com/Caitlyn-Yin/speculative-action.git/'
EXIT=128
```

Identical failure for `git push origin trackP-complete-2026-09-21`.

**Cause.** `~/.gitconfig` delegates GitHub auth to `gh`:
`credential.https://github.com.helper=!/home/hyin66/bin/gh auth git-credential`.
`gh` cannot start because **`$HOME/.config` is a root-owned regular file** (`-r-------- root:root`,
2903 bytes), so `~/.config/gh/config.yml` resolves through a file. No sudo ⇒ cannot be repaired here.
Same root cause as the vLLM XDG crash in §10. No `GITHUB_TOKEN`/`GH_TOKEN`, no `~/.git-credentials`,
no `gh` config elsewhere, `/run/secrets` empty.

**Tag `trackP-complete-2026-09-21` (`ebe9979`) exists locally only**, ready to push.

**Per WAYS_OF_WORKING §1, Track P is therefore NOT complete** — the commits are not durable.
The off-node transfer is blocked pending a destination from the PI.

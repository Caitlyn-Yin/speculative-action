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

> **RESOLVED 2026-09-28 — see §12.** The push blocker is gone; every commit and the tag are on
> `origin`. Track P is now complete in the sense §1 requires.

---

## 12. 2026-09-28 — push unblocked, environment rebuilt, Paper B frozen

### Push (Task A) — CLOSED

`origin` now carries `refs/heads/phase2-fixed-scaleup` and the tag
`trackP-complete-2026-09-21`. Route: **SSH with a repo key**, alias `github.com-specmem`
(`~/.ssh/id_specmem_deploy`). `gh`/HTTPS was abandoned, not repaired.

**Root cause of the §11 failure, now understood:** `$HOME` is the **root of a JuiceFS mount**, and
JuiceFS materialises `.accesslog`, `.stats` and `.config` as root-owned virtual files at its mount
root. So `~/.config` is a 0400 root-owned *regular file* on every mount, permanently — not a cron
artifact, not repairable, and its mtime merely tracks pod start. Any XDG path beneath it fails.
This is the same root cause as the vLLM `NotADirectoryError` in §10. Procedure in
`WAYS_OF_WORKING.md` §3.

Off-node backup: `git bundle --all`, verified, carried to the PI's laptop.

### Environment (Task B) — re-materialised after a pod recycle

`/tmp` was wiped by a restart between 2026-09-21 and 2026-09-28. Rebuilt from the committed
scripts; **all five `config.json` sha256 hashes and all five snapshot revisions match the values
pinned in `ENV_PREP.md`**, which is the first evidence `download_ladder.sh` actually reproduces the
recorded state. `check_server.py --all` → ALL GREEN.

**Version drift vs `ENV_PREP.md`:** vLLM **0.30.0** (doc: 0.29.0), openai SDK **3.19.2** (doc:
3.16.2). gymnasium 0.29.1 and numpy 1.26.4 held at their pins.

### Online 3-arm isolation gate — run; verdict INCONCLUSIVE

Criterion: handoff doc §4.4, implemented verbatim in `scripts/run_invariant.py`. Full result in
`docs/INVARIANT_REPORT.md`. The INCONCLUSIVE verdict is driven by the criterion's literal
cache-miss rule, **not** by an isolation defect; §6 of that report documents two wording problems
(the cache rule is circular against the power check; nondeterminism does not cascade to later
steps) and applies neither. **Two open rulings for the PI.**

The gate's power is carried by idx 1267, which is fully deterministic: UNISO reproduces the
documented contamination (`Lookup[father]` → `No more results.`) and ISO does not.

### Three harness defects found and fixed

1. `HistoryWrapper.reset` accepted `idx` and passed `idx=None` downward → `HotPotQAWrapper` drew a
   **random** question every reset. Fixed-index runs were impossible, and `runner.run()` never ran
   the seeded shuffle it computes — **`run_metrics/` was collected over random draws.** Regression
   test added (`tests/test_fixed_idx_regression.py`).
2. `runner.webthink` crashed on the isolated arm at any non-`search[]` step (`sim_obs` is `None`);
   the unisolated arm survived only by recording a **stale** observation from an earlier search.
   Now `""`. **Changes `simobs.json` semantics for non-search steps — needs a ruling before Phase C.**
3. No Wikipedia cache existed. Added `environment.wiki_get` behind `WIKI_CACHE=1` with hit/miss
   counters.

### Paper B pre-registration — FROZEN

`docs/PREREG_PAPER_B.md`, commit **`7732590b129ed059f631fde2ee6f45fee8fd910d`**, frozen before any
Paper B data collection. Amendments append to that file's `## Amendments` section as new commits;
the file is never rewritten.

### Open rulings carried forward

- §6 obs-identity vs speculated text (unchanged, still not decided).
- Judge contract source — recovered vs rewritten (unchanged).
- **New:** the two §4.4 criterion wordings (`INVARIANT_REPORT.md` §6).
- **New:** `sim_obs` semantics for non-search steps.
- The two residual contamination channels from §10.

---

## 13. 2026-09-29 — frozen local Wikipedia, two retrieval modes

Full record: **`docs/LOCAL_WIKI.md`**. Paths: `docs/ENV_PREP.md`. Per-query fidelity data:
`docs/fidelity_kilt_20190801.json`. Gate re-runs: `docs/INVARIANT_REPORT_title_exact.md`,
`docs/INVARIANT_REPORT_title_exact_5q.md`.

### What exists now

`WikiEnv.search_step` dispatches on `constants.retrieval_backend` ∈ {`live`, `title_exact`,
`bm25`} (env override `RETRIEVAL_BACKEND`; default still `live`). The two local modes read a
frozen **KILT 2019-08-01** corpus — 5,903,530 pages, 7.6 GiB on JuiceFS at
`$HOME/specmem-data/local_wiki/kilt_20190801` — and differ in **exactly one** behaviour: on a
query that resolves to no page, `title_exact` returns upstream's
`Could not find X. Similar: [...]` and `bm25` silently returns the top-1 BM25 page.
Everything else is shared code, pinned by a byte-identity test on 50 titles.

New files: `hotpotqa/src/local_wiki.py`, `hotpotqa/src/mw_title.py`,
`hotpotqa/tests/test_local_wiki.py` (29 tests), `scripts/build_local_wiki.py`,
`scripts/collect_queries.py`, `scripts/wiki_fidelity.py`.

### Results

- **Fidelity vs live, 300 queries: 94.0% hit/miss agreement** (bar 90%) — gold 99.0%,
  perturbed 90.0%, agent 93.0%; same-page agreement 92.9% on the 182 both-hit queries.
  Every disagreement category except one is *time* (renames, creations, deletions,
  disambiguation churn, the absent redirect table), not mechanism. There is **no**
  `disagree_casing` row, i.e. no evidence of a defect in the near-match ladder.
- **Offline tests: 42/42 pass**, under both `title_exact` and the default backend.
- **Online 3-arm gate under `title_exact`:** 1 question → INCONCLUSIVE (no power; the bug
  needs `idx 1267`); 5 questions → **FAIL** on a gated `real_action` divergence at
  `idx 5619` step 4. Cache confounding is gone (0 hits / 0 misses on all arms), so the rule
  that made §12's run INCONCLUSIVE can no longer fire — the next rule in line fires instead.

### Redirect data for this snapshot is NOT obtainable

`dumps.wikimedia.org/enwiki/20190801/` is 404 (mirror keeps only `20260301`+); the Internet
Archive `wikimediadownloads` collection has only `enwiki-20190120/20190201/20190220` for
2019 and no `enwiki-20190801` item; KILT ships no redirect table. The partial substitutes
present in KILT (`wikidata_info.aliases`, `anchors[].href`) were deliberately not used —
they would fabricate a resolution rule no MediaWiki version implements. **Measured cost:
7 of 300 queries (2.3%).**

### No prebuilt BM25 index exists for KILT

`pyserini`'s prebuilt catalogue has no KILT entry and its own KILT guide says to build from
scratch (~100 GB). Built with `bm25s` 0.3.11 over `title + ". " + lead paragraph`, Lucene
BM25, `(?u)\w+` tokenizer, English stopwords, Snowball stemmer: 1.30 GiB, 3.6 min.

### Open rulings added

1. **§6b of `INVARIANT_REPORT.md` is now load-bearing.** With cache confounding removed, the
   "nondeterminism does not cascade to later steps" wording is the only thing between the
   gate and a verdict. Not applied, not re-decided. **Blocks Phase C.**
2. **`search_step` does not percent-encode the search URL.** `entity.replace(" ", "+")` only,
   so any `&` in a query truncates the live search term. Affects 2 of 300 fidelity queries
   and every live run ever done on a gold title containing `&`. A behavioural change to the
   live path, so not patched.
3. **A zero-result live search is treated as an article.** `Special:Search` renders no result
   headings, so upstream's `if result_divs` test falls through and the agent receives search-page
   boilerplate as page text. Same shape as 2.
4. **Should `retrieval_backend` default to `title_exact`?** Left at `live` so recorded runs keep
   their semantics; flipping it is a protocol decision.
5. **KILT structural markers** (`Section::::`, `BULLET::::-`) survive into observations. Left in
   so page text is a pure function of the source; stripping them changes every observation.

Carried forward unchanged: §6 obs-identity, judge-contract source, `sim_obs` semantics for
non-search steps, the two residual contamination channels.

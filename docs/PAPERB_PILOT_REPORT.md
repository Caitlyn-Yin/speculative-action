# PAPERB_PILOT_REPORT — 25-question pilot per backend, and the nondeterminism it exposed

**Date:** 2026-09-29 / 30 · **Branch:** `phase2-fixed-scaleup` · **Pre-registration:**
`docs/PREREG_PAPER_B.md` (frozen `7732590`, Amendment 1 added by this work).

**Headline.** The pilot ran twice. The first run **failed go/no-go (a)** — 22.6%
(`title_exact`) and 13.6% (`bm25`) of control re-runs did not reproduce their logged
trajectory, against a `< 5%` bar. The cause is vLLM's **prefix cache**, not the harness:
with `--no-enable-prefix-caching` the same pilot gives a **0.0%** nondeterministic-pair
rate on both backends. All three go/no-go gates then pass, so the scale-up to 100
questions per backend is authorized by the prereg and is running.

Pushed commits for this work: `8032e25` (battery), `105a7bd` (collection + audit),
`6d5d6b3` (statistics + diagnosis), plus this report's commit — see §8.

---

## 1. What was rebuilt first (the pod had recycled again)

`/tmp` was wiped between 2026-09-29 01:xx and this session: no envs, no weights, both
servers down, GPU at 0 MiB. Everything came back from the committed scripts.

| Step | Result |
|---|---|
| `setup_envs.sh` | pipeline py3.10 (gymnasium 0.29.1, numpy 1.26.4, **openai 3.22.0**), serving py3.12 (**vLLM 0.30.0**, torch 2.13.0+cu130) |
| `download_ladder.sh` | 56 GB, 5/5 repos; **all five `config.json` sha256 and all five revisions match `ENV_PREP.md`** (third independent reproduction) |
| `serve_local.sh both` | actor `Qwen3-8B` :8000 (0.60), speculator **`Qwen3-4B`** :8001 (0.25) — 86.0 GB / 143 GB |
| `check_server.py --all` | ALL GREEN |
| Offline test suites | **115 pass**: `test_paperb` 20, `test_gates` 53, `test_isolation_regression` 5, `test_fixed_idx_regression` 8, `test_local_wiki` 29 |

**Version drift vs `ENV_PREP.md`:** openai SDK **3.22.0** (doc: 3.16.2, §12 recorded
3.19.2). vLLM 0.30.0 as in §12. gymnasium and numpy held at their pins.

**One reproducibility hole closed.** `test_local_wiki` came back with 16 errors —
`ModuleNotFoundError: zstandard`. The corpus libraries (`zstandard`, `bm25s==0.3.11`,
`PyStemmer`, `pyarrow`) had been installed by hand when the corpus was built and were
documented only in `LOCAL_WIKI.md` §Running. They are now installed by
`scripts/setup_envs.sh`, so a recycle no longer silently loses the ability to read
`pages.zst`.

---

## 2. What was built for Paper B

| Component | File | Role |
|---|---|---|
| Replay engine, population filter, labels | `hotpotqa/src/paperb.py` | forced action prefix → actor continuation; S1/S2/S3/harmful/delayed; control-vs-log check |
| Stage 1 collection | `scripts/collect_pairs.py` | `trajectories.jsonl` + `pairs_raw.jsonl` + manifest |
| Stage 2 audit | `scripts/replay_audit.py` | `controls.jsonl` + `pairs.jsonl` (the file the battery scores) + `audit_summary.json` |
| Stage 3 statistics | `scripts/analyze_paperb.py` | cluster bootstrap over questions, 10k resamples, H1–H4 |
| Diagnostics | `scripts/diag_nondeterminism.py`, `scripts/probe_determinism.py` | §5 |
| Tests | `hotpotqa/tests/test_paperb.py` | 20 offline tests, scripted actor + scripted corpus |

Collection is **not** a second ReAct loop: it is `runner.webthink(simulate=True)`, the
same call the isolation gate makes, and `paperb.episode_record` only reads what that
loop recorded. Only the *replay* side is a new loop, because a forced prefix is
something `webthink` cannot do; it mirrors `webthink`'s realized branch statement for
statement (same templates, same `action_lowercase` before `env.step`, same
`obs.replace('\n','')`, same `continue` on an unparseable reply, same trailing
`finish[]`). A drift between the two loops would surface as a nondeterministic pair,
which is exactly what §5 had to rule out.

### Two harness changes, both flagged, both default-off

1. **Who proposes the candidates.** Upstream `webthink` generates the k speculated
   actions with `self.llm` — **the actor** — and uses the speculator model only inside
   `WikiEnv.guess_step` to imagine a page. The prereg assumes a separate speculator
   ("Actor Qwen3-8B, speculator Qwen3-4B, k=3"). `constants.spec_actions_from` now
   selects; the default stays `"actor"` so every recorded run keeps its semantics, and
   Paper B sets `"speculator"`, recorded in every manifest. This is a deviation from
   upstream *code*, not from the prereg.
2. **Speculator token logprobs.** `constants.capture_spec_logprobs` routes the
   candidate generation through `LLMClient.call_with_logprobs`, so criterion 9
   (`spec_confidence`) has a token stream instead of going NA. Off by default.

Neither touches the realized trajectory, and the isolation regression tests still pass.

---

## 3. Pilot run 1 — FAILED go/no-go (a)

25 questions per backend (the first 25 of the seeded shuffle, so index-comparable with
every earlier run), `k=3`, step cap 8, temperature 0 both roles, frozen KILT 2019-08-01
corpus, `isolate_speculation=True`.

| Backend | pairs | deterministic | **nondet rate** | non-EM same-tool | S3 | S3 (same-tool) | S1 (same-tool search) | harmful | delayed |
|---|---|---|---|---|---|---|---|---|---|
| `title_exact` | 314 | 243 | **0.226** | 117 | 0.609 | 0.744 | 0.247 | 0.099 | 0.379 |
| `bm25` | 294 | 254 | **0.136** | 120 | 0.654 | 0.833 | 0.316 | 0.059 | 0.343 |

Gate (a) `< 5%`: **FAIL** both backends. Gate (b) `>= 50` same-tool pairs: pass
(117 / 120). Gate (c) fidelity `>= 90%`: pass (0.940, from `LOCAL_WIKI.md`, not
recomputed). The stop rule (S3 `< 2%` in both backends ⇒ probable bug) did not fire.

Structure of the failures: **every** divergence was action-level — the actor chose a
different action — and **none** was observation-only. Per backend, 28/126 and 16/119
controls diverged, with the first divergence 1–5 steps *after* the forced prefix, never
inside it.

---

## 4. Diagnosis: the loops agree with each other, not with the log

`scripts/diag_nondeterminism.py` re-runs, on one question: `webthink` with and without
speculation, and `paperb.replay` with an empty prefix (twice) and with prefixes 1..1 and
1..2 — recording every actor prompt.

* **idx 6904:** all six arms produced the **same** action sequence, and all six differed
  from the log. So the replay loop and the collection loop are not mismatched; the log
  is simply not reproducible.
* **idx 5619:** two *identical* replays (`replay_empty_a` vs `replay_empty_b`, same
  process, minutes apart) diverged at step 4 from each other, while five of six arms
  agreed.

That combination rules out a prompt mismatch and points at the server.

## 5. Attribution: vLLM prefix caching

`scripts/probe_determinism.py` removes the loop entirely: build the actor prompt for
step k by re-executing the logged prefix (a forced prefix makes no actor call, so the
prompt is a pure function of the log and the corpus), then issue that **identical**
request 10 times.

| Server config | Questions probed | Result |
|---|---|---|
| pilot config (prefix caching **on**) | 5619, 6904, 1267 | idx 1267: **2 distinct replies / 2 distinct actions** in 10 identical requests (9× `Search[An Appeal to the Coloured Citizens of the World]`, 1× `Search[Edward Garrison Walker]`) |
| `--no-enable-prefix-caching` | 5619, 6904, 1267, 3021 | **10/10 identical on all four** |

Mechanism, stated as a hypothesis consistent with the evidence rather than as something
we instrumented inside vLLM: a prompt whose prefix is already in the cache is prefilled
in a different chunk layout than one computed fresh, the logits differ in the last bits,
and a greedy argmax near a tie flips. It explains all three observations — within-session
reproducibility once the cache is warm, disagreement with a log recorded when the cache
held different blocks, and occasional flips at eviction boundaries.

Two residual facts worth recording, because they bound what was fixed:

* Under caching-on, idx 6904 and 1267 reproduced a *stable but different* action than
  the log (10/10). So the pilot log itself was unreproducible, not merely noisy.
* `--enforce-eager` was **not** tested. Prefix caching alone was sufficient to drive the
  measured rate to 0.0%, so CUDA-graph and kernel-selection effects remain untested and
  must not be claimed as ruled out.

---

## 6. Pilot run 2 — all gates pass

Same code, same questions, same corpus; both servers restarted with
`--no-enable-prefix-caching` (now reproducible via `EXTRA_ACTOR_ARGS` /
`EXTRA_SPEC_ARGS` in `scripts/serve_local.sh`).

| Backend | pairs | deterministic | **nondet rate** | non-EM same-tool | search | S3 | S3 (same-tool) | S1 (same-tool search) | S2 (same-tool) | harmful | delayed |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `title_exact` | 321 | 321 | **0.000** | 159 | 145 | 0.670 | 0.780 | 0.273 | 0.493 | 0.100 | 0.318 |
| `bm25` | 288 | 288 | **0.000** | 135 | 123 | 0.556 | 0.704 | 0.330 | 0.532 | 0.063 | 0.424 |

| Go/no-go | Bar | `title_exact` | `bm25` | Verdict |
|---|---|---|---|---|
| (a) nondeterministic-pair rate | `< 5%` | 0.000 | 0.000 | **PASS** |
| (b) non-EM same-tool pairs | `>= 50` | 159 | 135 | **PASS** |
| (c) `title_exact` fidelity vs live | `>= 90%` | 0.940 | 0.940 | **PASS** (from `LOCAL_WIKI.md`; not recomputed) |
| stop rule: S3 `< 2%` in both | — | 0.670 | 0.556 | does not fire |

Wall clock, one H200, everything sequential: collection 140 s / 128 s, audit 525 s /
460 s per backend. A replay costs ~1.7 s, so the audit is cheap relative to collection
per pair.

**These rates are not yet the paper's numbers.** They are single point estimates without
CIs (stage 3 has not been run on them), the cells are `n ≈ 300` so they *are* citable by
the prereg's `n >= 100` rule, but the pilot exists to clear the gates, and the scale-up
supersedes it.

### One thing to keep in mind when reading S3

S3 is `em_T >= em_C AND n_steps_T <= n_steps_C` — "no worse on either axis". On this
question set the actor's EM is low and many pairs leave both arms at EM 0 with equal
length, so ties count as S3 successes and the base rate is high (0.56–0.67). That is the
frozen definition and it was not changed; H1 asks only whether the rate's CI lower bound
exceeds 0, which such a base rate makes easy to satisfy. The interesting quantities are
therefore `harmful` (6–10%), `delayed` (32–42%) and the criterion precisions, not S3
alone.

---

## 7. What has NOT been done

* **Criterion battery not yet scored on this data.** `criteria_scores.jsonl` does not
  exist for either backend; H3 and H4 need it, and so does every precision/AUROC number.
  The alternate-judge circularity check (`NousResearch/Meta-Llama-3.1-8B-Instruct` on
  :8002) needs the speculator server stopped first — the GPU has ~21 GB free at 0.60 +
  0.25.
* **Stage 3 not run.** No cluster-bootstrap CI in this report is stated because none was
  computed; every number above is a point estimate.
* **2WikiMultihopQA** (the prereg's secondary dataset) untouched.
* The open rulings from `STATE_RESUME.md` §13 are all still open and none was re-decided
  here: §6b of `INVARIANT_REPORT.md`, `search_step` percent-encoding, zero-result live
  search, the `retrieval_backend` default, KILT structural markers, `sim_obs` semantics
  for non-search steps, the two residual contamination channels.
* `--enforce-eager` untested (§5).

## 8. Durability

| Artifact | Where |
|---|---|
| Code + docs | `origin/phase2-fixed-scaleup` — `8032e25`, `105a7bd`, `6d5d6b3`, and this report's commit |
| Pilot corpora (both runs), scale-up corpora | `/tmp/specmem/paperb/{pilot25_,pilot25b_,scale100_}{title_exact,bm25}` — **on the overlay, NOT durable**; reproducible from `collect_pairs.py` + `replay_audit.py` given the same servers and corpus |
| Frozen corpus | `$HOME/specmem-data/local_wiki/kilt_20190801` (JuiceFS, durable) |
| Probe/diagnosis reports | `/tmp/specmem/paperb/{probe_default,probe_noprefix,diag_nondet_title_exact}.json` — not durable; the numbers that matter are transcribed above |

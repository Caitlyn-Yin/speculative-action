# Paper B pre-registration

**Frozen 2026-09-28, before any Paper B data is collected.** Later changes go in the
`## Amendments` section as new commits. This document is never rewritten.

Working title: **Execution-grounded evaluation of acceptance criteria for speculative agents.**

## Unit

A speculation pair `p = (run, question, step i, spec j)` with `lower(spec_j) != lower(real_i)` —
i.e. **rejected** by Speculative Actions' exact match.

## Arms

Same retrieval backend, temperature 0, same prompts and parser as collection.

- **Control:** re-execute logged actions `1..i`, then the actor continues from step `i+1` to the end.
- **Treatment:** re-execute actions `1..i-1`, execute `spec_j`, record
  `(actor's thought_i, spec_j, ACTUAL observation of spec_j)` as step `i`; the actor continues from
  step `i+1` to the end. **Never use the speculator's hallucinated `sim_obs`.**
- A pair is **nondeterministic** if the control re-run differs from the logged trajectory; excluded
  from primary analysis, rate reported.

## Labels

- **S1:** normalized next actions equal (NA if either arm ends at step `i`).
- **S2:** the control's step-`(i+1)` action appears among treatment steps `i+1..i+3` (NA as S1).
- **S3 (primary):** `em_T >= em_C` AND `n_steps_T <= n_steps_C`.
- **harmful:** `em_T < em_C`. **delayed:** `n_steps_T > n_steps_C`.
- Also record delta `em`, `f1`, `steps`.

## Factors

Retrieval backend in `{title_exact, bm25}` over the same frozen Wikipedia; they differ only on
queries that miss an exact title (`title_exact` returns `"Could not find X. Similar: [top-5]"`,
`bm25` returns the top-1 page).

Dataset: HotpotQA dev (primary), 2WikiMultihopQA dev (secondary, after primary).
Actor `Qwen3-8B`, speculator `Qwen3-4B`, `k=3`, step cap 8.

## Hypotheses [falsifier]

- **H1:** S3 rate among non-EM same-tool pairs has 95% CI lower bound > 0 in at least one backend.
  *[falsifier: upper bound < 2% in both]*
- **H2:** S1 rate among non-EM same-tool **search** pairs is higher under `bm25` than `title_exact`.
  *[falsifier: difference CI includes 0 or is negative]*
- **H3:** for at least one call-level criterion, precision w.r.t. S3 differs between backends.
  *[falsifier: every such difference CI includes 0]*
- **H4 (Qwen3-8B judge only):** AUROC for S3 of the verbalized-confidence judge is lower than the
  same rubric scored by Yes/No log-prob margin.
  *[falsifier: difference CI includes 0 or is positive]*

## Statistics

Cluster bootstrap over questions, 10k resamples, 95% CIs. Cells with `n < 100` pairs are reported
but **marked not citable**. Target `>= 400` pairs per backend.

## Go/no-go

After the 25-question pilot per backend, scale to 100 questions only if:

- (a) nondeterministic-pair rate < 5%;
- (b) `>= 50` non-EM same-tool pairs per backend;
- (c) local `title_exact` fidelity `>= 90%` vs live Wikipedia.

If S3 rate < 2% in both backends at pilot, **stop and investigate as a probable bug.**

## Amendments

### Amendment 1 (2026-09-30) — serving config: vLLM prefix caching OFF

**What changed.** Both vLLM servers are now started with
`--no-enable-prefix-caching` (`EXTRA_ACTOR_ARGS` / `EXTRA_SPEC_ARGS` in
`scripts/serve_local.sh`). Nothing about the unit, the arms, the labels, the
factors or the hypotheses changes. Temperature stays 0, the prompts and parser
stay identical between collection and re-execution.

**Why.** The first 25-question pilot failed go/no-go (a) badly: **22.6%**
(`title_exact`) and **13.6%** (`bm25`) of control re-runs did not reproduce
their logged trajectory, against a `< 5%` bar. Every divergence was
action-level, never observation-level. Two measurements attributed it
(`docs/PAPERB_PILOT_REPORT.md`):

* `scripts/diag_nondeterminism.py` — the replay loop and the collection loop
  agree with *each other* and disagree with the log, and two identical replays
  can disagree at the same step. So it is not a prompt mismatch between the two
  loops.
* `scripts/probe_determinism.py` — one identical request issued 10 times:
  with prefix caching on, 1 of 10 replies differed on one question; with it off,
  4 of 4 questions returned 10/10 identical replies.

Re-running the same pilot with caching off gave a nondeterministic-pair rate of
**0.0%** on both backends (321 and 288 pairs).

**Consequence.** The cache-on pilot corpus is **not** part of Paper B data; it
is retained only as the evidence for this amendment. All Paper B data is
collected and re-executed under caching off.

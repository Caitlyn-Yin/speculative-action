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

*(none yet — append new subsections here, each in its own commit)*

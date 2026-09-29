# JUDGE_CONTRACT.md — LLM-call discipline for the criterion battery

**This is a re-derivation, not a recovery.** The pre-loss `JUDGE_CONTRACT.md` was
lost with the GH200 host (`docs/LOST_WORK_MANIFEST.md` item 5) and its content was
never restated anywhere, so nothing here can be assumed to match it. `CLAUDE.md`
§6 ruling 4 listed "salvage vs re-derive" as open; this document takes the
re-derive branch, on instruction. Consequence, stated once and applying
everywhere: **any pre-loss judge number is not comparable to a number produced
under this contract.**

Scope: every LLM call and every embedding call made by
`hotpotqa/src/gates.py` via `hotpotqa/src/backends.py` — criteria 5, 6, 7, 8, 12,
13 of `docs/CRITERIA.md`.

## 1. Determinism

- Local backend only (`constants.llm_backend = "local"`, OpenAI-compatible vLLM).
  No external LLM API is used for a research run.
- `temperature = 0`, `top_p = 1`, `seed = constants.local_seed` (0) forwarded to
  vLLM. Qwen3 thinking is disabled (`enable_thinking=False`) and stripped.
- `max_tokens` is fixed per criterion, in code, not per call site: 32 for the
  verbalized judge, 1 for every Yes/No margin, 512 for the Sufficient Context
  autorater.
- Judge models are pinned by served model name; the embedding model is pinned by
  id **and** by the HF snapshot commit sha (`LocalEmbedder.revision`). A `null`
  revision in the manifest means the weights could not be pinned and must be
  reported as such — not rounded up to "pinned".

## 2. The cache is append-only

`backends.AppendOnlyCache` writes JSONL, one record per call, and is the only path
to a model:

- **Key** = `sha256` over the canonical JSON of everything that can change the
  output: `model_id`, call kind, the full prompt, and the sampling parameters
  (`temperature`, `top_p`, `seed`, `max_tokens`, `top_logprobs`). Embedding keys
  additionally cover the revision, the pooling mode and `max_length`.
- **Records are only ever appended.** Nothing is rewritten, deleted, compacted or
  reordered, so the file is an audit log as well as a cache. Each record carries
  the full prompt and the raw response, so a score can be re-derived from the
  cache alone, without the code that produced it. Writes are flushed and
  `fsync`ed so a killed run loses at most a partial trailing line (which the
  loader skips).
- **On a duplicate key the first record wins.** A later record with the same key
  and a *different* response is kept in the file and counted in
  `cache.collisions`; `score_criteria.py` prints a warning naming the file. A
  collision means non-determinism somewhere (server, weights, sampling) and is a
  finding to report, never something to silently overwrite.
- Cache statistics (records, hits, misses, collisions) go into
  `criteria_manifest.json` for every run.

## 3. Parsing and refusals

- A judge reply that does not parse is **NA**, with `na_reason`, and never a
  silent reject: `judge_v2_verbal` needs a `Verdict:`, `sufficient_context` needs
  `{"Sufficient Context": 0|1}`. The tolerated fallbacks are enumerated in the
  criteria docstrings (a bare `yes`/`no`, a `70%`-style confidence) and each sets
  `detail.parse_note`.
- For a Yes/No margin, surface variants (case, leading space, sentencepiece
  underline) are aggregated by log-sum-exp before the margin is taken. If one
  side is missing from the top-k its log-probability is **bounded above** by the
  smallest returned one and `detail.clipped` names the side, so a margin resting
  on a bound is never mistaken for a measured one. If neither side appears, the
  result is NA.
- NA counts per criterion are reported in the manifest. An NA rate that is not
  near zero on real data invalidates that criterion's cell, and must be reported
  rather than dropped.

## 4. What a judge may see

- A **call**-level criterion is given the question, the realized history, the two
  actions. It may not be given any observation of the speculated action; a test
  asserts every call-level criterion scores identically with the observations
  stripped.
- An **obs**-level criterion is given the **executed** observation of `spec_j`.
  A speculator-imagined observation (`WikiEnv.guess_step`) is refused by code
  (`gates._require_executed_obs`), because scoring an imagined page at the obs
  level would turn an obs-level criterion into a call-level one and quietly
  reclaim the latency the obs level is supposed to cost.
- `sufficient_context` is given `obs(spec_j)` but **not** `obs(real_i)`: it is a
  sufficiency rater, and showing it the real observation would let it compare.
- No criterion may read the replay labels. They are copied into the output row
  only so the audit can join on one file.

## 5. Judge models

- Primary judge: `Qwen/Qwen3-8B` — the same family as the actor, which is the
  circularity risk.
- Circularity check: criteria 6–8 are re-run under a non-Qwen model of similar
  size (`NousResearch/Meta-Llama-3.1-8B-Instruct`, present on this node) and the
  result is written to the row's `judge_alt` block. When no alternate judge is
  configured, `judge_alt` is `null` — the absence is recorded in the data, not
  left to be inferred from silence.
- Criterion 13 is not re-run under the alternate judge: its prompt is a published
  artifact from another model generation, so a second local judge says nothing
  about circularity in *our* rubric.

## 6. Reproducing a run

Same pairs file + same manifest (models, revisions, grids, code version) + the
cache ⇒ byte-identical `criteria_scores.jsonl`, with zero new model calls. If a
re-run makes calls, either the prompt construction changed (the code version will
differ) or the cache was moved; both are reportable, neither is repaired by
re-running.

# CRITERIA.md — the 13 audited acceptance criteria

The criteria Paper B audits against the replay labels of `docs/PREREG_PAPER_B.md`
(S1, S2, S3, harmful, delayed). Implementation: `hotpotqa/src/gates.py`
(criteria, pure stdlib) + `hotpotqa/src/backends.py` (judge/embedder/cache) +
`scripts/score_criteria.py` (CLI → `criteria_scores.jsonl`). Tests:
`hotpotqa/tests/test_gates.py`, on the synthetic fixture
`hotpotqa/tests/fixtures/pairs_synthetic.jsonl` (53/53 offline, no server).

Three things this document exists to keep honest:

1. **Verified vs chosen.** Every criterion's docstring in `gates.py` carries two
   mandatory headings — `Parameters verified from the source:` and
   `Parameters chosen by us:` — and `register()` refuses to register a criterion
   that lacks either (`test_docstrings_separate_verified_from_chosen`). The
   tables below are the summary; the docstrings are the contract.
2. **Level.** `call` criteria read only the two action strings, the question and
   the realized history. `obs` criteria read the **executed** observation of
   `spec_j` — so they cannot be evaluated before the speculated call has run,
   and cannot hide the real call's latency. This is enforced in code:
   `gates._require_executed_obs` refuses any pair whose `spec_obs_source` is not
   `"executed"`, which is how `WikiEnv.guess_step`'s *imagined* page
   (`src/environment.py:86`) is kept out of the obs level
   (`test_no_obs_criterion_scores_an_imagined_observation`). A companion test
   asserts the reverse direction too: every `call` criterion scores identically
   with the observations stripped out.
3. **Provenance.** Criteria 2, 3, 6 and 7 are ours, and their pre-loss
   definitions were lost with the GH200 host (`docs/LOST_WORK_MANIFEST.md`
   items 1, 5, 6). What is implemented is a **re-derivation made on
   instruction**, not a recovery. Per `CLAUDE.md` §3, numbers scored under these
   four are **not comparable to any pre-loss number**. Criterion 4's published
   attribution also did not survive source-checking — see the discrepancy log.

---

## 1. The table

| # | Name | Level | Output | Source | Faithful? |
|---|---|---|---|---|---|
| 1 | `exact_sa` | call | binary | Speculative Actions, [2510.04371] §3; released code `hotpotqa/src/metrics.py:80-86` | yes |
| 2 | `normalized` | call | binary | ours (re-derived `NormalizedMatchGate`) | n/a |
| 3 | `battery` | call | binary + 6 flags | ours (re-derived deterministic battery) | n/a |
| 4 | `edit_distance_dsp` | call | score + {0.1…0.5} | DSP [2509.01920]; released matcher `OpenAGI/openagi_utils.py:37-39` | **no** |
| 5 | `embed_call` | call | score + {0.80, 0.85, 0.90, 0.95} | SpecBox [2607.23933] §3.3/§4 (0.80); Cost-Aware [2606.07846] §7.4 (0.95) | yes |
| 6 | `judge_v2_verbal` | call | score + {0.50…0.95} | ours (SE rubric v2) + H4 of the prereg | n/a |
| 7 | `judge_v2_logprob` | call | score + {−4…4} | ours (same rubric); log-odds form from DualSpec Eq. 6 | n/a |
| 8 | `dualspec_critic` | call | score + {−4…4} | DualSpec [2603.07416] App. A.1 (prompt, verbatim) + Eq. 6/7 | yes |
| 9 | `spec_confidence` | call | score + {0.50, 0.70, 0.80, **0.90**, 0.95} | SPORK [2607.03333] §4.2 Eq. (2), θ = 0.90 | yes (+ our arg-span variant) |
| 10 | `obs_equal` | obs | binary (+ byte-identity) | AOSpec [2608.00881] §4.2 | **no** (normalized is primary) |
| 11 | `spechop_rules` | obs | binary | SpecHop [2605.21965] App. D.4 | yes |
| 12 | `embed_obs` | obs | score + {0.80…0.95} | thresholds as in 5; the transfer to observations is ours | partly |
| 13 | `sufficient_context` | obs | binary | Sufficient Context [2411.06037]; prompt verbatim from `hljoren/sufficientcontext` README | yes |

All nine arXiv identifiers were resolved and read on 2026-09-29; the titles are
listed in §5. Section numbers below are from the arXiv HTML of `v1` in each case.

---

## 2. Verified vs chosen, criterion by criterion

### 1 `exact_sa` — call, binary
- **Verified:** case-insensitive equality of the whole action string, arguments
  included; no normalization, no threshold (`metrics.py:81-86`). Speculative
  Actions discards a speculative branch when no match exists (§3).
- **Chosen:** nothing. The criterion calls the real `Metrics.compare_action`
  when importable (it pulls in pandas via `src/utils.py`) and a verbatim
  transcription otherwise; a test asserts the two agree on every fixture pair.
- By construction of the Paper B unit this is `False` on every in-population
  pair. It is scored anyway, as the null gate and as a check that the population
  filter was applied.

### 2 `normalized` — call, binary — **ours, re-derived**
- **Verified:** nothing. The pre-loss gate is lost; this file plus the docstring
  is the whole specification.
- **Chosen:** tool channel must match; argument normalizer = NFKD diacritic
  stripping → lowercase → non-word characters to spaces → whitespace collapse →
  repeated removal of leading `the/a/an`; parenthetical disambiguators **kept**
  (`search[Mercury (planet)]` ≠ `search[Mercury]` — they retrieve different
  pages); binary, no threshold.

### 3 `battery` — call, binary + flags — **ours, re-derived**
- **Verified:** nothing; only the four pre-loss field names survived (in
  `docs/STATE_RESUME.md` and `docs/LOST_WORK_MANIFEST.md`).
- **Chosen:** all predicates. Flags are *rejection* reasons, so
  `binary = not any(flag)`:
  - `tool_channel` — different tools;
  - `terminal_channel` — exactly one side is `finish[...]`;
  - `channel` — `tool_channel or terminal_channel` (derived alias, excluded from
    the `any()` so it cannot double-count);
  - `stale_intent` — the speculated (tool, normalized arg) was already executed
    at some step ≤ i−2;
  - `spec_fixation` — it repeats step i−1 exactly;
  - `agent_loop` — the realized history already contains it ≥ 2 times.
  The i−1 / ≤ i−2 boundary, the `≥ 2` count and the normalization (same as
  criterion 2) are all ours.

### 4 `edit_distance_dsp` — call, score — **attribution corrected**
- **Verified:** DSP's *released* acceptance predicate is exact string equality —
  `judge_to_be_true(s, t): return s == t`,
  [`OpenAGI/openagi_utils.py:37-39`](https://github.com/guanyilin428/Dynamic-Speculative-Planning/blob/main/OpenAGI/openagi_utils.py)
  (default branch, fetched 2026-09-29; it is the only comparison function in the
  repo — `util.py` has none, and `planner.py` calls this one at lines 72, 82,
  105, 111, 128, 135, 249, 297, 346, 410). The paper defines divergence only as
  the approximation agent proposing "a different action" (§3) — **no distance and
  no threshold exist in DSP to recover.** The "minimum edit distance" reading is
  a third party's: DualSpec [2603.07416] §6.1 writes that "DSP … accepts a draft
  only if it matches the base action (minimum edit distance)".
- **Chosen (the documented fallback):** normalized Levenshtein over the
  `norm_text`-ed argument string, divided by the longer length; grid
  {0.1, 0.2, 0.3, 0.4, 0.5}; accept when **distance ≤ threshold** (the score is a
  distance, not a similarity). `detail.dsp_exact_equal` records DSP's faithful
  predicate on every pair, so the audit can report both.
- Registered with `faithful=False`.

### 5 `embed_call` — call, score
- **Verified:** 0.80 is SpecBox's `τ_c`, "the semantic similarity exceeds the
  threshold τ_c = 0.8 used in the reported experiments" (§4 Implementation;
  defined in §3.3's `hit(x)` predicate, which also requires the tool identity to
  match). 0.95 is Cost-Aware's Tier 2: "Semantic equivalence: equiv(i, î) ==
  True per a domain predicate. Default: normalized-embedding cosine similarity
  ≥ 0.95 for text" (§7.4). **Neither paper names an embedding model.**
- **Chosen:** the embedding model (§4 below); embedding the argument string only;
  the interior grid points 0.85 and 0.90; accept when ≥ threshold. SpecBox's
  tool-identity precondition is reported in `detail.same_tool` rather than folded
  into the score, so one score can be read with or without it.

### 6 `judge_v2_verbal` — call, score — **ours, re-derived**
- **Verified:** nothing; the rubric is ours (the pre-loss `se_judge_v2` rules a–d
  are lost and their content was never restated anywhere). Only the *comparison*
  is pre-registered: H4 predicts this scoring is weaker than criterion 7's.
- **Chosen:** the rubric text `SE_RUBRIC_V2` (rules (a) tool channel, (b)
  referent, (c) specificity, (d) retrievable content — reproduced in `gates.py`);
  two-line output `Verdict:` / `Confidence: 0-100`; the fold
  `score = conf if Yes else 1 − conf`; a missing confidence defaults to 1.0 and
  is flagged; history observations truncated to 600 chars; grid
  {0.50, 0.60, 0.70, 0.80, 0.90, 0.95}; judge Qwen3-8B, greedy, 32 max tokens.
  An unparseable verdict is NA, never a silent reject.

### 7 `judge_v2_logprob` — call, score — **ours, re-derived**
- **Verified:** the functional form only — the log-probability margin
  `log p_acc − log p_rej` as a continuous verifier score is DualSpec's Eq. (6).
- **Chosen:** the rubric (byte-identical to criterion 6's — a test asserts the
  two prompts differ only in their instruction tail, which is what makes H4 a
  test of the *scoring channel*); single-token read-out with `max_tokens=1`,
  `top_logprobs=20`; case/underline/leading-space variants of Yes and No
  aggregated by log-sum-exp before the margin; a side missing from the top-k is
  bounded above by the smallest returned logprob and `detail.clipped` records it;
  grid {−4, −2, −1, 0, 1, 2, 4}; judge Qwen3-8B.

### 8 `dualspec_critic` — call, score
- **Verified:** the prompt, **verbatim** from App. A.1 ("Verifier Prompt
  Template") — `[SYSTEM: TRAJECTORY AUDIT]`, the three REJECT rules (Stagnation,
  Ungrounded Answer, Lazy/Drift), `Answer only "Yes" or "No".`; the score
  `log p(Yes) − log p(No)` (Eq. 6); the rule accept iff `score ≥ τ` (Eq. 7); the
  critic's verdict is binary and produced by the full-capacity model (§5).
- **Chosen:** τ — DualSpec tunes it offline on held-out GAIA to a ~20%
  intervention rate (§6.1) and never publishes the number, so we sweep
  {−4, −2, −1, 0, 1, 2, 4}; the serialization of `s_t`/`z_t`/`a_t` appended below
  the prompt (App. A.1 does not publish it; `z_t = ∅`, which the appendix
  covers); 600-char observation truncation; judge Qwen3-8B.
- **Kept faithful on purpose, including two consequences we do not repair:** it
  judges *trajectory progress*, not equivalence; and it never sees `real_i`, so a
  wrong-but-healthy-looking speculation can score high. The point of including it
  is to measure that.
- Transcription note: the arXiv HTML renders the appendix's quotation marks as
  typographic `”`; those are the only characters changed (to ASCII `"`). No word,
  line break or list item is added, removed or reordered.

### 9 `spec_confidence` — call, score
- **Verified:** the statistic `c = min_{i=2..L} exp(ℓ_i)` (Eq. 2); the span is
  "the tool-name span (the first L tokens after the `"name": "` prefix)" — the
  **name only, not the arguments**; the minimum starts at `i = 2` "because the
  first token of the span (the opening quote) is always high-probability
  boilerplate"; θ = 0.90, "the operating point that maximizes F1" (88% precision,
  100% recall, F1 = 0.937, filtering 77% of probes). SPORK's own *acceptance*
  test is separate and strict — the probe is accepted "if and only if its full
  tool call (name and serialized arguments) matches the main generation exactly" —
  i.e. this is a **dispatch** signal there, and we use it only as a score.
- **Chosen:** the mapping of SPORK's JSON surface syntax onto our ReAct syntax
  (`tool[arg]`: name span = characters before the first `[`, argument span =
  characters between the outermost brackets); when the name span is a single
  token, that token is used instead of an empty minimum (SPORK's `i=2` rule
  assumes a leading quote token our syntax does not have) and
  `detail.single_token_name_span` records it; the argument-span variant and its
  minimum over *all* its tokens; grid points 0.50, 0.70, 0.80, 0.95.

### 10 `obs_equal` — obs, binary
- **Verified:** AOSpec "retains only a continuation whose predicted observation
  is byte-identical to o_t" (§4.2), with **no normalization** — canonicalizing
  inconsequential fields is named there only as something that *could* improve
  acceptance.
- **Chosen:** making normalized equality the primary output and byte identity
  secondary (`detail.byte_identical`) — a deliberate deviation, registered with
  `faithful=False`, because our observations are truncated Wikipedia prose where
  a diacritic or trailing period is not a semantic difference; the normalizer is
  `norm_text`.

### 11 `spechop_rules` — obs, binary
- **Verified (App. D.4, quoted):** normalization "lowercasing, diacritic and
  punctuation removal"; immediate rejection of "common refusal or uncertainty
  patterns (e.g., 'I don't know,' 'information unavailable')"; "if the target
  observation contains multi-digit numbers (such as years or quantities), the
  speculative observation must contain matching numeric values, otherwise it is
  automatically rejected"; "Final acceptance requires either a direct substring
  match or a high degree of lexical overlap. Specifically, after filtering out
  standard English stopwords, the speculation must achieve either **≥ 72% token
  coverage** or a **Jaccard similarity of ≥ 0.55**"; "For exceptionally short
  answers (**under 5 characters**), the verifier defaults to requiring a perfect
  token match". All four constants and the stage ordering are the paper's. The
  repo `mehrdadsaberi/spechop` contains only a README, so the appendix is the
  entire recoverable specification.
- **Chosen:** the refusal-pattern list (the appendix gives two examples and no
  list; `REFUSAL_PATTERNS`' first two entries are those examples, matched after
  normalization on both sides); the stopword list (`STOPWORDS`, a frozen literal
  — NLTK is not importable offline and a frozen list keeps the criterion
  deterministic); the *direction* of token coverage (fraction of the **target's**
  content tokens present in the candidate); "matching numeric values" read as set
  containment of every multi-digit number of the target; Jaccard over
  stopword-filtered token sets; "under 5 characters" measured on the normalized
  target; whitespace tokenization.
- Direction: candidate = `obs(spec_j)`, target = `obs(real_i)`. SpecHop's
  verifier is conservative in one direction (it must not produce false
  positives), so this is fixed, not symmetric.

### 12 `embed_obs` — obs, score
- **Verified:** the two published thresholds (as in criterion 5). Neither paper
  applies them to observations — SpecBox scores invocation signatures,
  Cost-Aware scores predicted inputs.
- **Chosen:** the transfer to the observation level (this is why it is a separate
  criterion); the embedding model; no truncation beyond the model's own context
  limit (the backend records when it truncates); 0.85 and 0.90.

### 13 `sufficient_context` — obs, binary
- **Verified:** the full prompt text, verbatim from the released repo
  (`hljoren/sufficientcontext`, `README.md`, "Sufficient Context Autorater
  Prompt"), including the one-shot Roald Dahl example and the repeated
  instruction block; the output format `{"Sufficient Context": 1|0}`; that it is
  a 1-shot autorater (README: "1-shot Gemini 1.5 Pro prompt classifies
  sufficiency", 93% accuracy). The repo ships **no code** — six files, README +
  five PNGs — so the README is the authoritative copy of the prompt.
- **Chosen:** judge Qwen3-8B instead of Gemini 1.5 Pro (local-only rule), so the
  93% figure does not transfer; `<TIMESTAMP>` → `2019-08-01`, the frozen KILT
  snapshot date; the composition of `<context>` — each history observation in
  order, labelled with its action, then the speculated step's executed
  observation last; `max_tokens=512`; taking the *last*
  `"Sufficient Context": n` match in the reply (the prompt's own example contains
  an earlier one).
- Kept faithful as a **contrast**: it asks a question-level question ("is the
  retrieved context sufficient to infer the answer"), which is not what the
  replay labels measure ("does substituting this call change the outcome"). It is
  obs-level and does **not** see `obs(real_i)`, so it cannot cheat by comparing.

---

## 3. Discrepancy log (what source-checking changed)

1. **DSP publishes no edit distance.** Its released matcher is `s == t`
   (`openagi_utils.py:37-39`). The "minimum edit distance" attribution traces to
   DualSpec §6.1, not to DSP. Criterion 4 therefore ships our normalized
   Levenshtein with the fallback grid, is registered `faithful=False`, and
   records DSP's real predicate per pair. The audit must not cite "DSP's edit
   distance threshold".
2. **Neither SpecBox nor Cost-Aware names an embedding model**, so the 0.80 and
   0.95 thresholds are model-free in their sources; ours is recorded in the
   manifest and any transfer of their numbers to our model is an assumption, not
   a replication.
3. **DualSpec's τ is unpublished** (tuned to a ~20% intervention rate on GAIA) —
   hence a grid.
4. **SPORK's θ = 0.90 is a dispatch gate, not an acceptance rule**; SPORK's
   acceptance is exact full-call match. We only score it.
5. **AOSpec is byte-identity with no normalization**; our primary output
   normalizes, so `obs_equal` is a deviation and says so.
6. **SpecHop's appendix does not publish its stopword list or its refusal
   patterns**, and its repo has no code; those two are ours inside an otherwise
   fully verified rule set.
7. **The lost contracts are re-derived, not recovered** (criteria 2, 3, 6, 7 and
   `docs/JUDGE_CONTRACT.md`). `CLAUDE.md` §6 ruling 4 listed the choice between
   salvage and re-derivation as open; this work takes the re-derivation branch on
   instruction. Consequence, restated: pre-loss numbers for these criteria are
   not comparable to post-loss ones.

## 4. Models, and the circularity check

| Role | Model | Where | Status |
|---|---|---|---|
| Judge (criteria 6, 7, 8, 13) | `Qwen/Qwen3-8B` | vLLM `:8000` | per the brief |
| Alternate judge (criteria 6, 7, 8) | `NousResearch/Meta-Llama-3.1-8B-Instruct` | `/models/models--NousResearch--Meta-Llama-3.1-8B-Instruct` (read-only node cache, complete: 4 safetensors shards + config + tokenizer) | **available** — same size, different family |
| Embeddings (criteria 5, 12) | `BAAI/bge-base-en-v1.5`, revision `a5beb1e3e68b9ab74eb54cfd186867f64f240e1a`, CLS pooling, L2-normalized, `max_length=512` | `$HOME/specmem-data/hf_embed` (419 MB, fetched 2026-09-29) | fixed and recorded |

So the same-family circularity check for criteria 6–8 **is** runnable on this
node; it is not "absent". `score_criteria.py --judge-alt-model/--judge-alt-url`
serves it on a third port, and every scored row carries a `judge_alt` block —
`null` when no alternate judge was configured, so its absence is visible in the
data rather than inferred from silence. The battery re-runs exactly criteria 6–8
under the alternate judge (`gates.CIRCULARITY_CHECK_CRITERIA`); criterion 13 is
not re-run, since its prompt is a published Gemini-era artifact and a second
local judge would not speak to circularity in *our* rubric.

Embedding sanity check on this node (bge, offline): `cos("British Railways
Board", "the british railways board") = 0.970`, `cos(…, "Beyonce") = 0.468`.

## 5. Sources, as resolved

| Cited as | Title (arXiv, v1) |
|---|---|
| 2510.04371 | Speculative Actions: A Lossless Framework for Faster Agentic Systems |
| 2509.01920 | Dynamic Speculative Agent Planning |
| 2607.23933 | SpecBox: Speculative Sandbox Scheduling for Efficient LLM Agent Serving |
| 2606.07846 | Cost-Aware Speculative Execution for LLM-Agent Workflows: An Integrated Five-Dimension Method |
| 2603.07416 | DualSpec: Accelerating Deep Research Agents via Dual-Process Action Speculation |
| 2607.03333 | SPORK: Self-Speculative Forking to Accelerate Agentic LLM Inference |
| 2608.00881 | AOSpec: Action and Observation Co-Speculation for Low-Latency Agent Serving |
| 2605.21965 | SpecHop: Continuous Speculation for Accelerating Multi-Hop Retrieval Agents |
| 2411.06037 | Sufficient Context: A New Lens on Retrieval Augmented Generation Systems |

Code sources: `github.com/guanyilin428/Dynamic-Speculative-Planning` (DSP),
`github.com/hljoren/sufficientcontext` (prompt), `github.com/mehrdadsaberi/spechop`
(README only). All fetched 2026-09-29.

## 6. Output schema

`criteria_scores.jsonl` — one row per pair, keyed by `pair_id`:

```json
{"pair_id": "...", "run": "...", "step_i": 3, "spec_index": 0,
 "spec_action": "...", "real_action": "...", "spec_obs_source": "executed",
 "labels": {"S1": false, "S2": true, "S3": true, "harmful": false, "delayed": true},
 "criteria": {
   "exact_sa":        {"name": "...", "level": "call", "output": "binary", "binary": false, "detail": {"comparator": "upstream"}},
   "battery":         {"...": "...", "flags": {"tool_channel": false, "...": false}, "detail": {"fired": []}},
   "embed_call":      {"...": "...", "score": 0.874, "decisions": {"0.8": true, "0.85": true, "0.9": false, "0.95": false}},
   "spechop_rules":   {"...": "...", "binary": true, "detail": {"rule": "lexical_overlap", "coverage": 0.8, "jaccard": 0.6}},
   "sufficient_context": {"...": "...", "na_reason": "no executed observation for spec_j"}
 },
 "judge_alt": {"model_id": "...", "criteria": {"judge_v2_verbal": {}, "judge_v2_logprob": {}, "dualspec_critic": {}}}}
```

`labels` are copied through untouched and read by no criterion — they are the
ground truth the criteria are scored *against*. A criterion that cannot be
evaluated emits `na_reason` and no `binary`/`score`; it is never silently a
reject. `criteria_manifest.json` beside it records the models and revisions, the
threshold grids, the code version (`git rev-parse HEAD`, `-dirty` when the tree
is), the NA counts, the cache statistics and a dump of the registry.

## 7. The fixture

`hotpotqa/tests/fixtures/pairs_synthetic.jsonl` — 17 hand-designed pairs, one per
criterion branch (alias/diacritics, tool channel, terminal channel, stale intent,
fixation, loop, near miss, two token-logprob streams, byte-identical obs,
normalized-equal obs, paraphrase, refusal, missing year, short target, and one
`speculator_imagined` pair that must make every obs-level criterion NA).
Regenerate with `python3 tests/fixtures/make_synthetic_pairs.py`.

**Its replay labels are invented.** They exist so the schema and the audit join
are exercised end to end. No accuracy, precision, recall or AUROC may be computed
from this fixture; those come from the re-execution audit on real data.

## 8. Running it

```bash
cd hotpotqa
python3 tests/test_gates.py                    # 53/53, offline, stdlib only
~/micromamba/envs/es/bin/python tests/test_gates.py   # also exercises upstream Metrics (needs pandas)

# deterministic criteria only — no server, no weights:
python3 ../scripts/score_criteria.py --pairs pairs.jsonl \
    --out runs/phaseB/criteria_scores.jsonl --no-judge --no-embedder

# everything (servers up per scripts/serve_local.sh; HF_HOME for bge):
HF_HOME=$HOME/specmem-data/hf_embed $PIPELINE_PY ../scripts/score_criteria.py \
    --pairs pairs.jsonl --out runs/phaseB/criteria_scores.jsonl \
    --judge-model Qwen/Qwen3-8B --judge-url http://127.0.0.1:8000/v1 \
    --judge-alt-model NousResearch/Meta-Llama-3.1-8B-Instruct \
    --judge-alt-url http://127.0.0.1:8002/v1
```

`pytest` is not installed in either environment on this node, so
`tests/test_gates.py` carries its own runner; it is written in pytest style and
will also collect normally once pytest is available.

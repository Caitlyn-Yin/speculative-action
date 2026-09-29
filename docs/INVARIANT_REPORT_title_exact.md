# INVARIANT_REPORT — online 3-arm speculation-isolation gate

**Verdict: INCONCLUSIVE** — UNISO == SEQ_a on all questions — the bug never fired, so a green ISO is not evidence. Extend to 10 questions and re-evaluate.

Commit `e75c907` · questions [7107] · actor `Qwen/Qwen3-8B` · spec `Qwen/Qwen3-0.6B` · temp 0/0 · top_p 1/1 · k=3 · retrieval_backend `title_exact` (frozen corpus; no HTTP, so the cache counters are 0/0 and cannot confound the comparison)

Criterion is the handoff doc §4.4, implemented verbatim in `scripts/run_invariant.py`. The compared object is the **realized** trajectory only — `(real_action, real_obs)` per step, plus `n_steps` and `em`. Speculative outputs are diagnostic, never part of the test.

## 1. Per-question

| idx | n_steps | SEQ_a==SEQ_b | ISO==SEQ_a | UNISO==SEQ_a | first divergent step |
|---|---|---|---|---|---|
| 7107 | 4 | yes | yes | yes | - |

## 2. Divergences

No divergences in either comparison.

## 3. Wikipedia cache

| arm | hits | misses |
|---|---|---|
| SEQ_a | 0 | 0 |
| SEQ_b | 0 | 0 |
| ISO | 0 | 0 |
| UNISO | 0 | 0 |

Misses after SEQ_a: **0** (clean). SEQ_a runs first precisely to warm the cache; any later miss means a live Wikipedia fetch could have changed under the comparison.

## 4. Determinism control

SEQ_a == SEQ_b on every step of every question. No steps excluded.

## 5. Power check

- UNISO != SEQ_a on >=1 question: **NO**
- ISO sim_obs != real_obs on >=1 step (speculation ran): **yes**

Both must hold for PASS to mean anything. The contamination bug only fires when a realized `lookup[]` follows a speculated `search[]` in the same episode; if no episode had that shape, the gate has no power.

## Environment

- vLLM **0.30.0**, torch 2.13.0+cu130 (ENV_PREP records 0.29.0 — drift)
- openai SDK 3.19.2 (ENV_PREP records 3.16.2 — drift)
- gymnasium 0.29.1 pinned, numpy 1.26.4
- raw per-arm trajectories: `/tmp/specmem/invariant_title_exact/arms.json`

## Harness fixes this gate required

- `HistoryWrapper.reset` accepted `idx` and passed `idx=None` downward, so `HotPotQAWrapper` drew a **random** question via `np.random.randint` on every reset. Fixed-index arms were impossible until this was fixed; upstream `runner.run()` never ran the seeded shuffle it computes.
- No Wikipedia cache existed. Added `environment.wiki_get` behind `WIKI_CACHE=1`, with hit/miss counters so §3 is checkable rather than assumed.

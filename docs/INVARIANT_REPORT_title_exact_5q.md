# INVARIANT_REPORT — online 3-arm speculation-isolation gate

**Verdict: FAIL** — ISO diverged from SEQ_a on at least one gated step.

Commit `e75c907` · questions [7107, 5619, 373, 6904, 1267] · actor `Qwen/Qwen3-8B` · spec `Qwen/Qwen3-0.6B` · temp 0/0 · top_p 1/1 · k=3 · retrieval_backend `title_exact` (frozen corpus; no HTTP, so the cache counters are 0/0 and cannot confound the comparison)

Criterion is the handoff doc §4.4, implemented verbatim in `scripts/run_invariant.py`. The compared object is the **realized** trajectory only — `(real_action, real_obs)` per step, plus `n_steps` and `em`. Speculative outputs are diagnostic, never part of the test.

## 1. Per-question

| idx | n_steps | SEQ_a==SEQ_b | ISO==SEQ_a | UNISO==SEQ_a | first divergent step |
|---|---|---|---|---|---|
| 7107 | 4 | yes | yes | yes | - |
| 5619 | 6 | NO (step 5) | NO (step 4) | NO (step 4) | 4 |
| 373 | 2 | yes | yes | yes | - |
| 6904 | 7 | NO (step 4) | yes | NO (step 4) | 4 |
| 1267 | 7 | NO (step 3) | NO (step -) | NO (step 3) | 3 |

## 2. Divergences

#### idx 5619 — ISO vs SEQ_a — step 4

**Field:** real_action (agent chose differently)

| | SEQ_a | ISO |
|---|---|---|
| real_action | `Search[Persona Non Grata (2005 film)]` | `Search[Persona Non Grata political leader]` |
| real_obs | `Persona non grata is a 2005 Polish drama film directed by Krzysztof Zanussi.. BULLET::::- Zbigniew Zapasiewicz - Wiktor. BULLET::::- Nikita Mikhalkov - Oleg. BU…` | `Could not find Persona Non Grata political leader. Similar: ['Persona Non Grata (2005 film)', 'Persona non grata (disambiguation)', 'List of people declared per…` |

First differing obs token (#0): `Persona` vs `Could`


#### idx 5619 — UNISO vs SEQ_a — step 4

**Field:** real_action (agent chose differently)

| | SEQ_a | UNISO |
|---|---|---|
| real_action | `Search[Persona Non Grata (2005 film)]` | `Search[Persona Non Grata political leader]` |
| real_obs | `Persona non grata is a 2005 Polish drama film directed by Krzysztof Zanussi.. BULLET::::- Zbigniew Zapasiewicz - Wiktor. BULLET::::- Nikita Mikhalkov - Oleg. BU…` | `Could not find Persona Non Grata political leader. Similar: ['Persona Non Grata (2005 film)', 'Persona non grata (disambiguation)', 'List of people declared per…` |

First differing obs token (#0): `Persona` vs `Could`


#### idx 5619 — UNISO vs SEQ_a — step 5

**Field:** real_action (agent chose differently)

| | SEQ_a | UNISO |
|---|---|---|
| real_action | `Lookup[Polish attaché]` | `Search[Persona Non Grata (2005 film)]` |
| real_obs | `(Result 1 / 1) BULLET::::- Jerzy Stuhr - Polish attaché Radca.` | `Persona non grata is a 2005 Polish drama film directed by Krzysztof Zanussi.. BULLET::::- Zbigniew Zapasiewicz - Wiktor. BULLET::::- Nikita Mikhalkov - Oleg. BU…` |

First differing obs token (#0): `(Result` vs `Persona`


#### idx 6904 — UNISO vs SEQ_a — step 4

**Field:** real_action (agent chose differently)

| | SEQ_a | UNISO |
|---|---|---|
| real_action | `Search[Woolworth Building construction date]` | `Finish[Woolworth Building]` |
| real_obs | `Could not find Woolworth Building construction date. Similar: ['F. W. Woolworth Building (Watertown, New York)', 'F. W. Woolworth Building (Wilmington, Delaware…` | `Episode finished, reward = 1 ` |

First differing obs token (#0): `Could` vs `Episode`


#### idx 6904 — UNISO vs SEQ_a — step 5

**Field:** trajectory length (one arm has no step here)

| | SEQ_a | UNISO |
|---|---|---|
| real_action | `Search[Woolworth Building construction date]` | `<none>` |
| real_obs | `Could not find Woolworth Building construction date. Similar: ['F. W. Woolworth Building (Watertown, New York)', 'F. W. Woolworth Building (Wilmington, Delaware…` | `<none>` |

First differing obs token (#0): `Could` vs `<END>`


#### idx 1267 — UNISO vs SEQ_a — step 3

**Field:** real_action (agent chose differently)

| | SEQ_a | UNISO |
|---|---|---|
| real_action | `Search[Edward F. Walker]` | `Search[Edward G. Walker]` |
| real_obs | `Edward F. Walker (January 20, 1852 – 1918) was a minister and general superintendent in the Church of the Nazarene.. Born in Steubenville, Ohio, Walker pastored…` | `Edward Garrison Walker, also known as Edwin Garrison Walker (1830–1901), was an American artisan in Boston who became an attorney in 1861; he was one of the fir…` |

First differing obs token (#1): `F.` vs `Garrison`


#### idx 1267 — UNISO vs SEQ_a — step 4

**Field:** real_action (agent chose differently)

| | SEQ_a | UNISO |
|---|---|---|
| real_action | `Search[An Appeal to the Coloured Citizens of the World]` | `Search[David Walker]` |
| real_obs | `Could not find An Appeal to the Coloured Citizens of the World. Similar: ['Colour My World', 'Colours in the Dark World Tour', 'Painted fish', 'Colour My World …` | `Could not find [David Walker]. Similar: ['David Walker', 'David Walker Barton', 'David Walker (Kentucky politician)', 'David James Walker', 'David A. Walker (ma…` |

First differing obs token (#3): `An` vs `[David`


## 3. Wikipedia cache

| arm | hits | misses |
|---|---|---|
| SEQ_a | 0 | 0 |
| SEQ_b | 0 | 0 |
| ISO | 0 | 0 |
| UNISO | 0 | 0 |

Misses after SEQ_a: **0** (clean). SEQ_a runs first precisely to warm the cache; any later miss means a live Wikipedia fetch could have changed under the comparison.

## 4. Determinism control

Nondeterministic steps (excluded from the gate):

- idx 5619: steps [5, 6, 7] · question-level (n_steps or em differ)
- idx 6904: steps [4, 5, 6, 7] · question-level (n_steps or em differ)
- idx 1267: steps [3, 4, 5, 6, 7]

This is a **server** issue, not a gate failure. Next diagnostics: re-run with `--enforce-eager`, and confirm the vLLM `seed` is applied per request. Versions in play are recorded below.

## 5. Power check

- UNISO != SEQ_a on >=1 question: **yes**
- ISO sim_obs != real_obs on >=1 step (speculation ran): **yes**

## Environment

- vLLM **0.30.0**, torch 2.13.0+cu130 (ENV_PREP records 0.29.0 — drift)
- openai SDK 3.19.2 (ENV_PREP records 3.16.2 — drift)
- gymnasium 0.29.1 pinned, numpy 1.26.4
- raw per-arm trajectories: `/tmp/specmem/invariant_te5/arms.json`

## Harness fixes this gate required

- `HistoryWrapper.reset` accepted `idx` and passed `idx=None` downward, so `HotPotQAWrapper` drew a **random** question via `np.random.randint` on every reset. Fixed-index arms were impossible until this was fixed; upstream `runner.run()` never ran the seeded shuffle it computes.
- No Wikipedia cache existed. Added `environment.wiki_get` behind `WIKI_CACHE=1`, with hit/miss counters so §3 is checkable rather than assumed.

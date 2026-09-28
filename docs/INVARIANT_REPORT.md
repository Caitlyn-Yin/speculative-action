# INVARIANT_REPORT — online 3-arm speculation-isolation gate

**Verdict: INCONCLUSIVE** — 2 Wikipedia cache misses after SEQ_a — the comparison is confounded by possible Wikipedia drift.

Commit `c13da3a` · questions [7107, 5619, 373, 6904, 1267] · actor `Qwen/Qwen3-8B` · spec `Qwen/Qwen3-0.6B` · temp 0/0 · top_p 1/1 · k=3 · WIKI_CACHE=1

Criterion is the handoff doc §4.4, implemented verbatim in `scripts/run_invariant.py`. The compared object is the **realized** trajectory only — `(real_action, real_obs)` per step, plus `n_steps` and `em`. Speculative outputs are diagnostic, never part of the test.

## 1. Per-question

| idx | n_steps | SEQ_a==SEQ_b | ISO==SEQ_a | UNISO==SEQ_a | first divergent step |
|---|---|---|---|---|---|
| 7107 | 5 | yes | yes | yes | - |
| 5619 | 7 | yes | yes | yes | - |
| 373 | 2 | yes | yes | yes | - |
| 6904 | 7 | NO (step 4) | NO (step 6) | NO (step 4) | 6 |
| 1267 | 7 | yes | yes | NO (step 3) | 3 |

## 2. Divergences

#### idx 6904 — ISO vs SEQ_a — step 6

**Field:** real_action (agent chose differently)

| | SEQ_a | ISO |
|---|---|---|
| real_action | `Search[Woolworth Building construction]` | `Search[Woolworth Building construction date]` |
| real_obs | `Could not find Woolworth Building construction. Similar: ['Woolworth Building', 'Woolworth', 'Woolworths Group', 'List of tallest buildings in the United States…` | `Could not find Woolworth Building construction date. Similar: ['Woolworth Building', 'List of tallest buildings in the United States', 'List of tallest building…` |

First differing obs token (#5): `construction.` vs `construction`


#### idx 6904 — UNISO vs SEQ_a — step 4

**Field:** real_action (agent chose differently)

| | SEQ_a | UNISO |
|---|---|---|
| real_action | `Search[Woolworth Building construction date]` | `Search[Woolworth Building]` |
| real_obs | `Could not find Woolworth Building construction date. Similar: ['Woolworth Building', 'List of tallest buildings in the United States', 'List of tallest building…` | `The Woolworth Building is a 792-foot-tall (241 m) residential building and early skyscraper at 233 Broadway in the Tribeca neighborhood of Lower Manhattan in Ne…` |

First differing obs token (#0): `Could` vs `The`


#### idx 6904 — UNISO vs SEQ_a — step 5

**Field:** real_action (agent chose differently)

| | SEQ_a | UNISO |
|---|---|---|
| real_action | `Search[Woolworth Building]` | `Search[Woolworth Building construction date]` |
| real_obs | `The Woolworth Building is a 792-foot-tall (241 m) residential building and early skyscraper at 233 Broadway in the Tribeca neighborhood of Lower Manhattan in Ne…` | `Could not find Woolworth Building construction date. Similar: ['Woolworth Building', 'List of tallest buildings in the United States', 'List of tallest building…` |

First differing obs token (#0): `The` vs `Could`


#### idx 1267 — UNISO vs SEQ_a — step 3

**Field:** real_obs (env.page / env.obs leaked from the speculative branch)

| | SEQ_a | UNISO |
|---|---|---|
| real_action | `Lookup[father]` | `Lookup[father]` |
| real_obs | `(Result 1 / 6) Though his father was enslaved, his mother was free; therefore, he was free as well (partus sequitur ventrem).` | `No more results. ` |

First differing obs token (#0): `(Result` vs `No`


#### idx 1267 — UNISO vs SEQ_a — step 5

**Field:** real_action (agent chose differently)

| | SEQ_a | UNISO |
|---|---|---|
| real_action | `Lookup[father]` | `Search[David Walker (abolitionist)]` |
| real_obs | `(Result 2 / 6) His father, who had died before his birth, had been enslaved.` | `David Walker (September 28, 1796 – August 6, 1830)[a] was an American abolitionist, writer,  and anti-slavery activist. Though his father was enslaved, his moth…` |

First differing obs token (#0): `(Result` vs `David`


## 3. Wikipedia cache

| arm | hits | misses |
|---|---|---|
| SEQ_a | 7 | 12 |
| SEQ_b | 18 | 1 |
| ISO | 19 | 0 |
| UNISO | 20 | 1 |

Misses after SEQ_a: **2** (CONFOUNDED). SEQ_a runs first precisely to warm the cache; any later miss means a live Wikipedia fetch could have changed under the comparison.

## 4. Determinism control

Nondeterministic steps (excluded from the gate):

- idx 6904: steps [4, 5, 7]

This is a **server** issue, not a gate failure. Next diagnostics: re-run with `--enforce-eager`, and confirm the vLLM `seed` is applied per request. Versions in play are recorded below.

## 5. Power check

- UNISO != SEQ_a on >=1 question: **yes**
- ISO sim_obs != real_obs on >=1 step (speculation ran): **yes**

## 6. Two issues in the criterion itself (verdict above is unchanged)

The verdict in this report is the §4.4 criterion applied **literally**, as instructed. Two
properties of that wording made it return INCONCLUSIVE for reasons unrelated to speculation
isolation. Both are reported for a ruling; neither has been applied.

### 6a. The cache-miss rule is circular, and these misses are not drift

The rule is "assert zero cache misses after SEQ_a, else the comparison is confounded by Wikipedia
drift." But a cached URL is **never re-fetched**, so drift is impossible for any URL SEQ_a already
saw. Every post-SEQ_a miss is therefore a **novel URL** — a search SEQ_a never made:

| arm | misses | searches not made by SEQ_a |
|---|---|---|
| SEQ_b | 1 | `woolworth building history` |
| ISO | 0 | — |
| UNISO | 1 | `david walker father residence` |

A novel search can only arise when an arm's trajectory diverges. But the POWER CHECK *requires*
UNISO to diverge. So the two clauses are in tension: whenever the gate has power, the divergent arm
almost certainly issues a novel search, which forces a miss, which forces INCONCLUSIVE. As written,
the gate can rarely return PASS at the same time as being meaningful.

**Suggested refinement (not applied):** count only re-fetches of already-cached URLs, which is what
"drift" actually means. Under that definition this run had **0** confounding fetches.

### 6b. Nondeterminism should cascade to all later steps

The criterion marks a step nondeterministic iff SEQ_a and SEQ_b differ *at that step*. For idx 6904
that yields `{4, 5, 7}` — leaving step 6 gated, and step 6 is the sole reason ISO is marked failing.

But the arms are autoregressive: once step 4 differs, every later step is conditioned on a different
prompt, so steps 5-7 are not comparable at all. The raw data shows exactly this — **ISO first
diverges from SEQ_a at step 4, the same step SEQ_b does**, and ISO agrees with SEQ_b there:

| step | SEQ_a | SEQ_b | ISO |
|---|---|---|---|
| 3 | `Search[…construction date]` | `Search[…construction date]` | `Search[…construction date]` |
| 4 | `Search[…construction date]` | `Search[Woolworth Building]` | `Search[Woolworth Building]` |
| 5 | `Search[Woolworth Building]` | `Search[…construction date]` | `Search[…construction date]` |
| 6 | `Search[…construction]` | `Search[…construction]` | `Search[…construction date]` |

Steps 4/5 are a **transposition**: SEQ_a orders the two searches one way, SEQ_b and ISO the other.
SEQ_b happens to re-converge at step 6 and ISO does not — a coin-flip downstream of the step-4
nondeterminism, not an isolation defect. ISO is not "failing step 6"; idx 6904 is simply
incomparable from step 4 onward.

**Suggested refinement (not applied):** exclude every step `>= first nondeterministic step`.

### What the data shows under both refinements

Applying 6a and 6b together would make the verdict **PASS**, and the power check is carried by a
question that needs neither refinement: **idx 1267 is fully deterministic** (SEQ_b and ISO are both
byte-identical to SEQ_a across all 7 steps), and UNISO diverges there at step 3 with exactly the
documented contamination signature —

| | SEQ_a / ISO | UNISO |
|---|---|---|
| real_action | `Lookup[father]` | `Lookup[father]` |
| real_obs | `(Result 1 / 6) Though his father was enslaved…` | `No more results.` |

— the speculated page overwrote `env.page`, so `construct_lookup_list()` found nothing. That is the
bug the isolation fix exists to prevent, firing on an unisolated arm and absent on the isolated one.

**This is stated as evidence, not as a verdict.** Changing either rule changes a lost contract, and
per `LOST_WORK_MANIFEST` that is the PI's call, not mine.

## Environment

- vLLM **0.30.0**, torch 2.13.0+cu130 (ENV_PREP records 0.29.0 — drift)
- openai SDK 3.19.2 (ENV_PREP records 3.16.2 — drift)
- gymnasium 0.29.1 pinned, numpy 1.26.4
- raw per-arm trajectories: `/tmp/specmem/invariant/arms.json`

## Harness fixes this gate required

- `HistoryWrapper.reset` accepted `idx` and passed `idx=None` downward, so `HotPotQAWrapper` drew a **random** question via `np.random.randint` on every reset. Fixed-index arms were impossible until this was fixed; upstream `runner.run()` never ran the seeded shuffle it computes.
- No Wikipedia cache existed. Added `environment.wiki_get` behind `WIKI_CACHE=1`, with hit/miss counters so §3 is checkable rather than assumed.
- `runner.webthink` crashed with `AttributeError: 'NoneType' object has no attribute 'replace'` on the ISO arm. `guess_step` is the only writer of `sim_obs` and runs only for `search[...]`, so on a `lookup[]`/`finish[]` step there is no speculated observation; `_restore_env` then puts `sim_obs` back to `None`. The unisolated arm did not crash only because `sim_obs` still held a **stale observation from an earlier search**, which was then recorded as this step's speculation. Now recorded as `""` = "no speculated observation for this step". This is the *diagnostic* channel and does not enter the criterion — but it does change what `simobs.json` contains for non-search steps, so **it needs a ruling before Phase C**, alongside the two residual contamination channels already listed in `CLAUDE.md`.

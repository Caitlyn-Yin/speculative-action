# LOST_WORK_MANIFEST — 2026-09-21

**Inventory only. Nothing here is re-implemented, and no lost semantics are re-derived.**

Compares `origin/main @ dc938b9` (the pristine upstream fork, the only thing on the remote) and the
Track P branch `phase2-fixed-scaleup` against the last known state of the science track
(`STEP2_5_REPORT.md` + the PI adjudications) — **as described in the briefs**, since those documents
themselves are among the lost artifacts (see `STATE_RESUME.md` §0, §9).

## Method, and its one limitation

Classification is by direct inspection of the two trees:

- `git grep -ci <term> dc938b9` for every Step-2.5 vocabulary item;
- the same against `HEAD` (Track P);
- hits scoped to `hotpotqa/` and read, not counted blindly.

Every apparent hit on `origin/main` was **incidental**: `battery` and `channel` occur only inside
HotpotQA *question text* and upstream trajectory logs (`hotpotqa/data/*.json`,
`hotpotqa/run_metrics/**/log.txt`), never in code. `overlay` and `schema_version` do not occur under
`hotpotqa/` at all — those counts came from the `chess-game/` and `e-commerce/` vendor trees, which
are out of scope. On `HEAD`, `stale_intent` and `obs_excerpt` occur **only in `docs/STATE_RESUME.md`
prose**, i.e. as descriptions of what is missing, not as implementations.

**Limitation:** the reference state is the briefs' description of `STEP2_5_REPORT.md`, not the
document. Field names, rule lettering and thresholds below are therefore *as named by the PI*. Where
a lost artifact's internal detail was never stated in a brief, it is marked **unknown content** —
that is the part a rewrite cannot reconstruct, only re-decide.

## Inventory

| # | Item | Classification | Evidence |
|---|---|---|---|
| 1 | Typed det-battery fields: `channel`, `stale_intent`, `spec_fixation`, `agent_loop` | **LOST** | 0 code hits on `dc938b9`; 0 on `HEAD` except `stale_intent` in STATE_RESUME prose. No `gates.py` in any ref |
| 2 | `adjudication_route` census (`obs_join` \| `rollout`) | **LOST** | 0 hits in any ref, either tree |
| 3 | Overlay v2, incl. the **7107/7 withdrawal** (`superseded_by`, append-only) | **LOST** | 0 hits for `superseded_by`/`overlay` under `hotpotqa/` in any ref. The withdrawal itself — a PI adjudication — exists only as the phrase "7107/7" in the brief |
| 4 | `AUDIT_PROTOCOL.md` (window-level audit rule) | **LOST** | absent from every ref; home-wide filename search returns nothing |
| 5 | `JUDGE_CONTRACT.md` | **LOST** | as above |
| 6 | `se_judge_v2` rules a–d (same-tool residue only) | **LOST** | 0 hits for `se_judge`/`judge_v2`; rule content never restated in a brief ⇒ **unknown content** |
| 7 | Schema 0.2.0 `obs_excerpt` | **LOST** | 0 code hits. Partially superseded: Track P bumps the trajectory schema (R1), but 0.2.0's field set is not reconstructed |
| 8a | Isolation invariant tests — **offline tier** | **RE-IMPLEMENTED in Track P** | `hotpotqa/tests/test_isolation_regression.py` (5 tests, commit `034dabd`). Scripted fixture, no network/LLM. **Not** claimed equivalent to the lost tier |
| 8b | Isolation invariant tests — **online tier** (A3(ii), 3-arm) | **LOST** | no gate definition, no runner, no recorded verdict; SLURM jobs 2652592/2662459 unreachable (STATE_RESUME §9) |
| — | Isolation *fix* itself (snapshot/restore) | **RE-IMPLEMENTED in Track P** | `runner.py` `_snapshot_env`/`_restore_env` + `constants.isolate_speculation` (`034dabd`). Minimal correctness fix, **not** the 0.2.0 wrapper |
| — | Phase B "exactly flat ladder" result (29 windows) | **LOST — and unverifiable** | fixture `runs/phase0_spec5/` not in git; see STATE_RESUME §3. Must not be cited as an established number |

**Nothing on this list is PRESENT on origin.** `origin/main @ dc938b9` is the untouched upstream
fork: `hotpotqa/{run.py,src/*,data/*,prompts/*,run_metrics/*,trajs/*}` and vendor trees. Every
science-track artifact is either LOST or was RE-IMPLEMENTED during Track P.

## Consequence for the rebuild decision

The re-implemented items (8a, the isolation fix) were recoverable because they are **mechanical**:
the bug is visible in upstream code, and correctness is checkable by a test that fails when the fix
is removed. Items 1–7 are not like that. They encode **decisions** — which residues count as
same-tool, what a window-level audit unit is, the four typed channels and their precedence, why
7107/7 was withdrawn. Re-deriving them produces *a* contract, not *the* contract, and any number
scored under it is not comparable to a pre-loss number.

That is the substance of the open ruling "judge contract source — recovered vs rewritten."

## Open rulings (exactly two)

1. **§6 obs-identity vs speculated text** — how Phase D's observation-join classifies a speculated
   `search[X]` whose observation was LLM-authored (`environment.py:85-97` imagines the page,
   truncated to 5 sentences by `get_page_obs`). Untouched since first raised.
2. **Judge contract source — recovered vs rewritten** — salvage from the GH200 (items 1–7 return
   intact), or re-derive on this node (items 1–7 are re-decided, and pre-loss numbers become
   incomparable). Blocked on the PI's ruling on the STATE_RESUME §9 failure mode.

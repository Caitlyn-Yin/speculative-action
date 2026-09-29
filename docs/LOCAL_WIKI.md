# LOCAL_WIKI — frozen local Wikipedia, two retrieval modes

**Status: built, measured, green.** 5,903,530 pages from the KILT 2019-08-01 knowledge
source on persistent storage; 300-query fidelity against live Wikipedia at **94.0%**
hit/miss agreement (bar 90%); 42/42 offline tests pass under both backends; the online
3-arm gate re-run on the frozen corpus.

Date: 2026-09-29. Node `hyin66-agent-0`.

---

## 1. Why

`WikiEnv.search_step` hit live `en.wikipedia.org`. Three consequences, all bad for this
programme:

1. **Drift.** A trajectory is a function of what Wikipedia said at that instant. A replay
   months later is a different experiment.
2. **The isolation gate cannot separate its factors.** Arms run minutes apart; an edit
   between two arms shows up as a trajectory divergence that has nothing to do with
   speculation isolation. `WIKI_CACHE=1` (commit `f6c4098`) patched this for URLs the
   reference arm happened to visit, but a later arm issuing a *novel* search still made a
   live request — the cache-miss rule that turned `INVARIANT_REPORT.md` INCONCLUSIVE.
3. **No environment manipulation was available.** H2/H3 need the retrieval behaviour to be
   a controlled variable, not a property of the internet.

The frozen corpus fixes 1 and 2 outright: there is no HTTP request at all, so the cache
counters are 0/0 and cannot confound anything. For 3 it provides two modes that differ in
**exactly one behaviour**.

## 2. The two modes

`constants.retrieval_backend` ∈ {`live`, `title_exact`, `bm25`} (override with
`RETRIEVAL_BACKEND`). Default stays `live` so previously recorded runs keep their semantics.

| | resolves to a page | resolves to nothing |
|---|---|---|
| `title_exact` | the page | **miss**: `Could not find {X}. Similar: [top-5 BM25 titles].` |
| `bm25` | the page | **the top-1 BM25 page**, silently |

Everything else is shared code: page text, `get_page_obs`'s five-sentence truncation,
`lookup[...]`, the disambiguation re-search, the counters, the observation strings. The
divergence lives in one method, `LocalWiki._on_miss` (`hotpotqa/src/local_wiki.py`).

That is what makes this a manipulation rather than two different environments: a
`title_exact` vs `bm25` contrast measures "does the retriever ever tell the agent no",
with page content held fixed. `tests/test_local_wiki.py` pins it — 50 sampled titles
return byte-identical `obs` **and** `page` under both modes, on the mini corpus and on the
real one.

### The one designed exception

A title whose page is a `"may refer to:"` page is turned into a **miss** by upstream's own
recursion (`environment.py` re-searches `"[" + entity + "]"`), so the miss rule applies and
the modes *do* diverge there. This is not a leak in the contract; it is the contract
propagating through upstream behaviour. Pinned by
`test_disambiguation_titles_are_the_one_designed_divergence` (`USS Calypso`). 3.9% of pages
are disambiguation pages (measured on a 1-in-2048 sample, n=2883).

## 3. Corpus: availability, size, cost

Checked from this node before downloading anything.

| | |
|---|---|
| Source | `http://dl.fbaipublicfiles.com/KILT/kilt_knowledgesource.json` |
| Reachable | yes, HTTP 200, `Server: AmazonS3` via CloudFront |
| Size | 37,318,876,722 B = **34.76 GiB** |
| Published md5 | `d1dca62aa6ba889d2e842182e3114af5` (S3 `x-amz-meta-s3cmd-attrs`) |
| Measured throughput | ~45 MB/s → **~13 min** to download |
| HF alternative | `facebook/kilt_wikipedia` exists and is public, but is a loading script over the same 34.76 GiB file, with no auto-converted parquet. Direct download is simpler and hashable. |
| **Prebuilt BM25 index** | **none exists.** `pyserini`'s prebuilt catalogue (`pyserini/prebuilt_index_info.py`, `docs/prebuilt-indexes.md`) has no KILT entry; `docs/experiments-kilt.md` tells you to build one yourself and warns it needs ~100 GB. `orionweller/kilt_passages_pyserini` on HF is a *passage dump*, not an index. We build our own (§5). |

Actual figures after the build (all in `MANIFEST.json`):

| artifact | size | time |
|---|---|---|
| raw download (scratch, `/tmp`) | 34.76 GiB | 13 min |
| sha256 + md5 verification | — | 1 min |
| `pages.zst` | 5.14 GiB | 8.5 min |
| `pages.sqlite` | 639 MiB | (in the above) |
| `leads.jsonl.zst` | 537 MiB | (in the above) |
| `bm25/` | 1.30 GiB | 3.6 min |
| **persistent total** | **7.6 GiB** | **~27 min end to end** |

sha256 of the raw source, recorded for any future rebuild:
`f966d6f09c4ff91656db5c56c384f136b0c495c7083c043586b8cb1033c389a5`.

## 4. Storage layout, and why it is where it is

`$HOME` is JuiceFS (persistent, survives pod recycles, 64 GB quota) and `/tmp` is the
container overlay (fast, ephemeral, wiped by the recycle that already happened twice during
this work). So:

```
$HOME/specmem-data/local_wiki/kilt_20190801/     # PERSISTENT, 7.6 GiB
    pages.zst        512-page independently-compressed zstd frames
    pages.sqlite     titles(norm -> pid), pages(pid -> frame offset, item index)
    leads.jsonl.zst  title + lead paragraph (the BM25 build input, kept for rebuilds)
    bm25/            bm25s index + doc_titles.json + doc_pids.json
    MANIFEST.json    sha256/md5 of the source, build commands, counts, library versions
$HOME/specmem-data/local_wiki/aux/               # PERSISTENT
    hotpot_dev_distractor.parquet    gold supporting facts (27 MB)
    fidelity_live_cache/             pinned live HTML for the fidelity study (318 responses)
/tmp/kilt_raw/                                   # SCRATCH, re-downloadable
    kilt_knowledgesource.json, build/
```

Paths are exported by `scripts/env.sh` (`WIKI_DATA_DIR`, `WIKI_RAW_DIR`, `WIKI_KS_URL`) and
recorded in `docs/ENV_PREP.md`. Nothing large is in git; `.gitignore` carries `*.zst`,
`pages.sqlite` and `kilt_knowledgesource.json` as a backstop.

**The build writes to the overlay and publishes once.** Building in place on JuiceFS was
tried and abandoned: inserting 5.9 M rows into a TEXT-primary-key table over a network
filesystem is random-page-write bound and ran at ~1 MB/s of output, projecting to ~3 h.
Two changes took it to 8.5 min: build under `/tmp/kilt_raw/build` and copy the finished
files across, and accumulate the title keys in RAM to insert them **sorted** so the B-tree
fills append-only. Both are commented at the point of the fix.

Every stage is idempotent: a stage recorded `complete` in `MANIFEST.json` is skipped unless
`--force`, and `fetch` resumes a partial download (`curl -C -`) and re-verifies the size and
md5 before proceeding.

```bash
source scripts/env.sh
$PIPELINE_PY scripts/build_local_wiki.py --stage all     # fetch, pages, bm25, verify
```

## 5. Retrieval design

### 5.1 Title resolution — the MediaWiki near-match ladder

`index.php?search=X` does **not** run a search first. `SpecialSearch` asks
`SearchEngine::getNearMatch()` for a "Go"-style near match and redirects to the article if
there is one; only when that returns nothing does the user see the result list that
upstream's `mw-search-result-heading` test detects. So the frozen corpus has to reproduce
`getNearMatch`, not a search.

`hotpotqa/src/mw_title.py` is a transcription of **MediaWiki REL1_33** (released 2019-06,
contemporaneous with the snapshot), read from
`raw.githubusercontent.com/wikimedia/mediawiki/REL1_33/`. Every function carries its
citation; the ladder itself is `includes/search/SearchNearMatcher.php:54-132`:

| order | source line | candidate |
|---|---|---|
| 1 | `SearchNearMatcher.php:79` | `Title::newFromText($term)` — the term, normalised |
| 2 | `:104` | `$lang->lc($term)` → lowercase, then ucfirst by `Title::capitalize` |
| 3 | `:110` | `$lang->ucwords($term)` → Capitalise Each Space-Separated Word |
| 4 | `:116` | `$lang->uc($term)` → ALL UPPERCASE |
| 5 | `:122` | `$lang->ucwordbreaks($term)` → capitalise after ` -(){}.,?!` too |
| 6 | `:165-169` | if the term is fully `"quoted"`, retry the whole ladder without the quotes |

Supporting ports:

* `languages/Language.php:2705-2831` — `ucfirst`, `uc`, `lc`, `lcfirst`, `ucwords`,
  `ucwordbreaks`. The PHP originals branch on `isMultibyte()` (`Language.php:2776`), which is
  a *byte-length* test, and `ucfirst` switches on `ord()` of the first **byte**
  (`Language.php:2706-2714`); both are reproduced on the encoded form, not on Python
  characters.
* `includes/title/MediaWikiTitleCodec.php:252-420` — `splitTitleString`: underscore/space
  equivalence, the Unicode whitespace and bidi-override classes, `_` trimming, the
  `^(.+?)_*:_*(.*)$` namespace/interwiki prefix split, fragment stripping, the
  illegal-character and relative-path and `~~~` and 255-byte rejections, and finally
  `Title::capitalize` (`Title.php:3110-3116`).
* `includes/DefaultSettings.php:3906` — `$wgLegalTitleChars`. The illegal ASCII set is
  therefore control characters plus `# < > [ ] { } |`, and
  `MediaWikiTitleCodec.php:467-471` additionally rejects `%XX`, `&entity;` and `&#NNN;`.

Two enwiki configuration facts are baked in, because the corpus is enwiki-only:
`$wgCapitalLinks = true` (main-namespace titles are ucfirst'd) and `en` has no language
variants, so the `$allSearchTerms` loop (`SearchNearMatcher.php:58-63`) runs once.

Observed effect on the fidelity run: of 182 local hits, 160 came from the exact candidate,
21 from `ucwords`, 2 from `ucwordbreaks`, 2 from `lc`. The ladder is not decoration — it
carries 14% of hits.

### 5.2 Redirects — **not obtainable for this snapshot**

Asked directly, and the answer is no:

* `dumps.wikimedia.org/enwiki/20190801/` → **404**. The mirror keeps only recent runs; the
  oldest available today is `20260301`.
* Internet Archive's `wikimediadownloads` collection has 18 enwiki-family items for 2019,
  and for enwiki proper only `enwiki-20190120`, `enwiki-20190201` and `enwiki-20190220`.
  `archive.org/metadata/enwiki-20190801` returns `{}` — no such item.
* The KILT knowledge source itself contains **no redirect table**: redirect pages are not
  articles and are not in the dump.

Two partial substitutes *are* present in each KILT record and were deliberately **not**
used: `wikidata_info.aliases` and `anchors[].{text,href}`. Neither is the redirect table;
using them would fabricate a resolution rule that no MediaWiki version implements, and
would be indistinguishable in the results from a faithful one. A 2019-02 redirect table
(the nearest obtainable) would introduce a six-month offset against a 2019-08 page table —
a different kind of unfaithfulness, silently.

**Measured cost of having no redirects: 7 of 300 queries (2.3%)** — the
`disagree_missing_redirect` row in §6. That is the whole price, and it is recorded rather
than hidden.

### 5.3 BM25

| | |
|---|---|
| Library | `bm25s` 0.3.11 (pure NumPy/SciPy; `pyserini` would need a JVM and a from-scratch ~100 GB Lucene build) |
| Scoring | `method="lucene"` (Lucene's BM25 variant, k1=1.5, b=0.75) |
| Documents | 5,903,530 — one per page |
| Indexed field | `wikipedia_title + ". " + lead paragraph` (KILT `text[1]`) |
| Tokenizer | `bm25s.tokenize(stopwords="en", stemmer=PyStemmer("english"))`: regex splitter `(?u)\w+`, lowercased, English stopword list, Snowball/Porter2 stemming |
| Load | `mmap=True` at runtime, so episodes share one resident copy |

Title + lead only, not full text: the miss observation lists *titles*, and the top-1
fallback wants the page whose subject matches the query, which is what the lead paragraph
identifies. Indexing full text would rank by incidental mention.

### 5.4 Page text

`local_wiki.kilt_paragraphs_to_page` reproduces upstream's HTML extraction rule given the
same paragraphs: strip each block, keep blocks with **more than two** space-separated
tokens, newline-terminate. `text[0]` is dropped — KILT stores the page title as its own
first paragraph, which has no counterpart among the rendered `<p>` elements.

`get_page_obs` and `construct_lookup_list` are untouched upstream code, so the five-sentence
truncation and the `lookup[]` semantics are identical by construction, not by imitation.

## 6. Fidelity vs live Wikipedia — 300 queries

`$PIPELINE_PY scripts/wiki_fidelity.py`. Full per-query records in
`docs/fidelity_kilt_20190801.json`. Live responses were fetched once and pinned under
`aux/fidelity_live_cache/` (318 responses); the reported run is a **100% cache-hit replay**
of them, so this table reproduces byte for byte.

Query sets, each n=100:

* **gold** — HotpotQA dev gold supporting-fact titles, from
  `hotpotqa/hotpot_qa` distractor/validation (the repo's `data/*_simplified.json` files
  carry only question/answer, and `paper_dev.jsonl` is FEVER, so the gold titles had to be
  fetched; 27 MB parquet, kept in `aux/`). Seed 248.
* **perturbed** — the same titles, one of {lowercase, drop one word, add one word} chosen
  per title (seed 249): 32 lower, 44 drop, 24 add.
* **agent** — 121 unique realized `search[...]` entities collected by
  `scripts/collect_queries.py` (12 recoverable from the committed `trajs/`, the rest from a
  33-episode live run against the real servers, since `hotpotqa/cache/wiki/` did not survive
  the recycle); the first 100 in collection order.

"Hit" is defined identically on both sides, **including** upstream's disambiguation
recursion: a query landing on a `"may refer to:"` page is re-searched as `"[query]"` and
that verdict is taken. Without this, live and local would be scored under different
definitions.

| set | n | hit/miss agreement | live hits | local hits | both hit | same page on both-hit |
|---|---|---|---|---|---|---|
| gold | 100 | **99.0%** | 98 | 97 | 97 | 92.8% |
| perturbed | 100 | **90.0%** | 45 | 37 | 36 | 91.7% |
| agent | 100 | **93.0%** | 54 | 51 | 49 | 93.9% |
| **overall** | **300** | **94.0%** | — | — | 182 | **92.9%** |

**Bar: ≥90% hit/miss agreement. Result: 94.0% overall, and ≥90% in every individual set.
PASS.**

### Categorised outcomes (300 queries)

| n | category | reading |
|---|---|---|
| 169 | `agree_hit_same_page` | both resolved to the same page |
| 100 | `agree_miss` | both reported a miss |
| 13 | `agree_hit_different_page` | both resolved, to different pages — **page renames since 2019** |
| 7 | `disagree_missing_redirect` | live followed a redirect/alias; the target page **is** in the corpus |
| 4 | `disagree_page_created_after_2019` | live target absent from the snapshot |
| 2 | `disagree_disambiguation_resolved_since_2019` | 2019 had a `"may refer to:"` page; live has an article |
| 2 | `disagree_became_disambiguation_since_2019` | the reverse |
| 2 | `disagree_live_url_unescaped` | **upstream defect, not a corpus gap** (below) |
| 1 | `disagree_page_gone_after_2019` | corpus has a page live search no longer resolves |

Examples, so the categories are checkable rather than asserted:

* `missing_redirect` — `university of washington` → live `University of Washington` (the
  corpus has the article; 2019's redirect from the lowercase-with-`of` form is what is
  missing). Also `Saddledome` → `Scotiabank Saddledome`, `at the drive-in` →
  `At the Drive-In`, `Jim's` → `Jim's Restaurants`.
* `page_created_after_2019` — `Queer as Folk (2000 TV series)`. In 2019 the article was at
  `Queer as Folk (American TV series)`; enwiki moved it afterwards. This one is instructive:
  it is an *agent-issued* query from a live run, i.e. the agent learned the 2026 title from
  the live index. Also `Persona non grata (Philippines)`, `Ralph Priso`.
* `agree_hit_different_page` — `Kiribati national basketball team` → live
  `Kiribati men's national basketball team`; `Mark Twain Riverboat` → live
  `Disney riverboats`; `2006 KNVB Cup Final` → live `2006 KNVB Cup final` (a case move).
  All are renames; the frozen corpus returns the 2019 article, which is correct for a
  2019 snapshot.
* `live_url_unescaped` — `C&amp;M Subdivision` (a HotpotQA gold title containing an HTML
  entity). `search_step` builds the URL as `entity.replace(" ", "+")` with **no
  percent-encoding**, so the `&` terminates the `search=` parameter and live Wikipedia
  searched `C` — landing on the article "C". The frozen corpus rejects `&amp;` as an
  illegal title sequence, faithfully to `MediaWikiTitleCodec.php:469`. The disagreement is
  an upstream URL-construction defect, scored in its own category so it cannot be mistaken
  for a corpus gap. **Not fixed here** — fixing it changes what live `search_step` does,
  which is a behavioural change outside this task's scope, and it is worth a ruling.

### What the disagreement profile says

Every category except `live_url_unescaped` is *time*, not *mechanism*: pages created,
deleted, renamed, or turned into disambiguation pages between 2019-08 and now, plus the
missing redirect table. That is the expected signature of a faithful 2019 snapshot, and it
is the drift the freeze exists to remove. There is **no** `disagree_casing` row — zero
queries where live resolved a title verbatim that our case ladder failed to find, which is
the one category that would indicate a defect in §5.1 rather than the passage of time.

`perturbed` is the weakest set at exactly 90.0%, and predictably so: 6 of its 10
disagreements are missing redirects, because perturbing a title is precisely how you
generate a string that only a redirect would resolve. Worth knowing before anyone builds a
perturbation-heavy experiment on this corpus.

## 7. Deviations from live `search_step`, enumerated

1. **`clean_str` is not applied.** Upstream runs every extracted block through
   `clean_str` — a `unicode-escape` → latin1 → utf-8 round-trip that repairs mojibake in
   scraped HTML. KILT text is already correct UTF-8; applying it would corrupt non-ASCII
   text (and raises on some inputs). Deliberate, documented at the function.
2. **KILT structural markers survive in page text.** `Section::::History.` and
   `BULLET::::- ...` are KILT's own markup. Most section markers are dropped for free by
   upstream's ">2 tokens" rule (`Section::::History.` is one token), but multi-word ones
   (`Section::::Early life and career.`) and all bullets survive. Bullets are the right
   analogue of upstream including `<ul>`; the markers themselves are noise. Not stripped, so
   that page text is a pure function of the frozen source. Visible in the §8 report as
   `BULLET::::- Zbigniew Zapasiewicz - Wiktor`.
3. **Disambiguation recursion terminates instead of looping.** Upstream recurses
   `search_step("[" + entity + "]")` with no depth guard. Locally, `[` and `]` are illegal
   title characters (`$wgLegalTitleChars`), so `title_exact` returns a miss and the
   recursion ends — which matches live, where the bracketed search renders a result list. In
   `bm25` mode the bracketed re-search would find a page again, so `_depth` caps the
   recursion at one level. That cap is the only structural difference from upstream.
4. **No redirects** (§5.2). Cost measured: 2.3% of queries.
5. **Namespace and interwiki prefixes are misses.** `Category:X`, `Talk:X`, `File:X`,
   `en:X` cannot be main-namespace articles, and KILT holds articles only. Faithful to
   MediaWiki, which would route them elsewhere entirely. One consequence: **1 title out of
   5,903,530** — `Topic: The Washington & Jefferson College Review` — is unreachable,
   because `Topic` is the StructuredDiscussions namespace on enwiki. Recorded in
   `MANIFEST.json` as `n_titles_unnormalisable: 1`.
6. **Zero-result live searches.** Only relevant to the fidelity harness: when live search
   finds nothing at all it renders `Special:Search` with no result headings, so upstream's
   `if result_divs` test falls through and hands the agent the search page's boilerplate as
   if it were an article. The harness scores that as a miss (detected via the
   `mw-special-Search` body class). Another upstream defect worth a ruling; not changed.
7. **`title_exact` and `bm25` diverge on disambiguation titles** (§2), by design.

Non-deviations worth stating, because they were checked rather than assumed: the miss
observation is upstream's f-string character for character
(`test_miss_observation_is_exactly_upstreams_f_string`); a miss leaves `page` and the lookup
state untouched, exactly as upstream does, which is the stale-page contamination channel the
isolation gate probes; `search_time` / `num_searches` are still updated, so
`get_time_info()` keeps working.

## 8. Re-run of the gates under `title_exact`

### Offline isolation tests — green

```
$ source scripts/env.sh && cd hotpotqa
$ RETRIEVAL_BACKEND=title_exact $PIPELINE_PY -m pytest tests/ -q
42 passed
$ $PIPELINE_PY -m pytest tests/ -q        # default backend, for comparison
42 passed
```

42 = 5 isolation regression + 8 fixed-index regression + 29 local-wiki. The isolation
regression tests script Wikipedia themselves, so they are backend-independent by
construction; running them under `title_exact` confirms the dispatch does not disturb them.

This surfaced one environment trap, now fixed in `scripts/env.sh` as trap 5: `import
sqlite3` in the pipeline env resolves `_sqlite3` → `libicui18n.so.78` → `CXXABI_1.3.15`,
which the system `libstdc++` lacks. It only failed when something else had already loaded the
system `libstdc++`, making it **order dependent** — `pytest tests/test_local_wiki.py` passed
while `pytest tests/` failed. `LD_LIBRARY_PATH` now carries the pipeline env's `lib`. Same
root cause as the vLLM `NotADirectoryError` family: this node's JuiceFS mount root and
conda/system library mismatch.

### Online 3-arm gate — run, verdict reported as the criterion produces it

```
$PIPELINE_PY scripts/run_invariant.py --n-questions 1 --retrieval-backend title_exact \
    --out docs/INVARIANT_REPORT_title_exact.md
$PIPELINE_PY scripts/run_invariant.py --n-questions 5 --retrieval-backend title_exact \
    --out docs/INVARIANT_REPORT_title_exact_5q.md
```

`run_invariant.py` gained `--retrieval-backend`, and its `WIKI_CACHE=1` guard is now scoped
to the live backend: a frozen corpus satisfies handoff §4.4's requirement *more* strongly
(there is no request to cache, and the counters stay 0/0, which the verdict's miss rule reads
as clean). Nothing else in the criterion changed.

| run | verdict | why |
|---|---|---|
| 1 question (`idx 7107`) | **INCONCLUSIVE** | `UNISO == SEQ_a`: the contamination bug never fired, so a green ISO is not evidence. Expected — `INVARIANT_REPORT.md` already records that the gate's power is carried by `idx 1267`, not 7107. A one-question run is provably powerless, which is why the 5-question run was also done. |
| 5 questions (`7107, 5619, 373, 6904, 1267`) | **FAIL** | `ISO` diverged from `SEQ_a` on a gated step (`idx 5619` step 4). |

Reading the FAIL honestly:

* **Cache confounding is gone.** 0 hits / 0 misses on all four arms. The rule that made the
  live run INCONCLUSIVE can no longer fire, which is the improvement this task was for.
* **Every gated divergence is `real_action`, never `real_obs`.** A speculation leak can only
  manifest as `real_obs` — `webthink` builds its own `running_prompt`, so a speculative step
  cannot change which action the actor emits. `real_action` divergence at temperature 0 is
  actor-server nondeterminism.
* **The determinism control sees it too, one step later.** `SEQ_a` vs `SEQ_b` (identical
  configuration, run twice) diverges at step 5 of `idx 5619`, so steps 5-7 are excluded — but
  ISO's divergence at step 4 is not excluded and drives the FAIL.
* This is **exactly** the open criterion-wording ruling already recorded in
  `INVARIANT_REPORT.md` §6b ("nondeterminism does not cascade to later steps"). It has not
  been applied or re-decided here. The verdict above is the literal criterion.
* The frozen corpus did not cause the nondeterminism: it changes which page text enters the
  prompt, so the sampling path differs and the nondeterminism surfaces at different steps.
  Under `live` the same 5 questions showed it on `idx 6904` only.

Side observation, recorded because it matters for Phase C throughput: with no HTTP,
`SEQ_a` over 5 questions took 11.8 s of wall clock in total.

## 9. Open items this created

1. **Criterion-wording ruling (already open, now load-bearing).** With cache confounding
   removed, `INVARIANT_REPORT.md` §6b is what stands between the gate and a verdict. It needs
   the PI's ruling before Phase C.
2. **Upstream does not percent-encode the search URL.** Two of 300 fidelity queries are
   affected, and any gold title containing `&` is affected in every live run ever done.
   Fixing it changes live `search_step` behaviour — a ruling, not a patch.
3. **Upstream treats a zero-result live search as an article.** Same shape.
4. **Should `retrieval_backend` default to `title_exact`?** Left at `live` so recorded runs
   keep their semantics. Flipping it is a protocol decision.
5. **KILT structural markers** (deviation 2) are visible in agent observations. Leaving them
   keeps page text a pure function of the source; stripping them would read better. A ruling,
   because it changes every observation.

## 10. Reproduce from nothing

```bash
cd /home/hyin66/speculative-action
bash scripts/setup_envs.sh                     # pipeline + vllm envs on the overlay
source scripts/env.sh
$PIPELINE_PY -m pip install zstandard bm25s PyStemmer pyarrow
$PIPELINE_PY scripts/build_local_wiki.py --stage all        # ~27 min, idempotent
cd hotpotqa && RETRIEVAL_BACKEND=title_exact $PIPELINE_PY -m pytest tests/ -q
cd .. && $PIPELINE_PY scripts/wiki_fidelity.py              # replays the pinned cache
```

Only `build_local_wiki.py --stage fetch` needs the network; if
`$WIKI_DATA_DIR/MANIFEST.json` already shows the stages complete, everything is a no-op.

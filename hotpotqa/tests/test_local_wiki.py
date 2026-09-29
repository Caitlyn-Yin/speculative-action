"""Tests for the frozen local Wikipedia (src/local_wiki.py, src/mw_title.py).

Three groups:

1. `TestNearMatch`   -- the MediaWiki case ladder, as a unit. Pure Python, no
   corpus. Every expectation is traceable to a line cited in src/mw_title.py.

2. `TestMiniCorpus`  -- a 60-page corpus built by `scripts/build_local_wiki.py`
   into a tmpdir, so the whole retrieval path (zstd frames, sqlite, bm25s) is
   exercised offline in ~10 s. This is where the contract between the two
   backends is pinned: *the only* observable difference is what a miss returns.

3. `TestRealCorpus`  -- the same byte-identity check against the real
   2019-08-01 build, skipped when it has not been built on this node.

The headline assertion the environment manipulation rests on: `title_exact` and
`bm25` return byte-identical observations for every query that resolves to an
article. If that ever fails, a title_exact/bm25 contrast is confounded by page
text and cannot be attributed to the miss behaviour.

One class of query is excluded and pinned separately: a title whose page is a
"may refer to:" page is turned into a MISS by upstream's "[X]" re-search, so the
miss rule applies to it and the modes diverge by construction. See
`test_disambiguation_titles_are_the_one_designed_divergence`.
"""

import json
import os
import subprocess
import sys

import pytest

from src import constants
from src.environment import WikiEnv
from src.mw_title import NOT_MAIN_NS, TitleError, mw_normalize_title, near_match_variants

REPO = os.path.dirname(os.path.dirname(os.path.abspath(os.path.join(__file__, ".."))))
BUILDER = os.path.join(REPO, "scripts", "build_local_wiki.py")


# ---------------------------------------------------------------------------
# 1. the near-match ladder
# ---------------------------------------------------------------------------

class TestNearMatch:

    def labels(self, term):
        return [lbl for lbl, _ in near_match_variants(term)]

    def titles(self, term):
        return [t for _, t in near_match_variants(term)]

    def test_ladder_order_matches_searchnearmatcher(self):
        # SearchNearMatcher.php:79,104,110,116,122 -- exact, lc, ucwords, uc,
        # ucwordbreaks, in that order, with duplicates dropped.
        assert self.labels("jean-luc picard") == [
            "exact", "ucwords", "uc", "ucwordbreaks"]
        assert self.titles("jean-luc picard") == [
            "Jean-luc picard", "Jean-luc Picard", "JEAN-LUC PICARD",
            "Jean-Luc Picard"]

    def test_lc_variant_is_first_letter_capitalised(self):
        # $lang->lc($term) then Title::capitalize -> ucfirst.
        assert self.titles("GO")[:2] == ["GO", "Go"]

    def test_ucwords_capitalises_each_space_separated_word(self):
        assert "New York City" in self.titles("new york city")

    def test_underscores_and_runs_of_whitespace_collapse(self):
        # MediaWikiTitleCodec.php:253,274-279
        assert mw_normalize_title("new_york   city") == "New york city"
        assert mw_normalize_title("  Go  ") == "Go"

    def test_fragment_is_stripped(self):
        # MediaWikiTitleCodec.php:365-372
        assert mw_normalize_title("Foo#Bar") == "Foo"

    def test_illegal_characters_reject_the_whole_ladder(self):
        # $wgLegalTitleChars (DefaultSettings.php:3906) excludes [ and ], and
        # SearchNearMatcher.php:80-82 returns null on a malformed exact term --
        # which is why upstream's "[entity]" disambiguation re-search is always
        # a miss against the frozen corpus, just as it is a result list live.
        for bad in ["[Go]", "a|b", "x{y}", "a<b", "100%20", "Foo&amp;"]:
            assert near_match_variants(bad) == [], bad

    def test_empty_and_hash_prefixed_terms_are_misses(self):
        # SearchNearMatcher.php:73
        assert near_match_variants("") == []
        assert near_match_variants("#REDIRECT") == []

    def test_namespace_and_interwiki_prefixes_leave_main_namespace(self):
        for term in ["Category:Foo", "Talk:Go", "File:X.png", "en:Go"]:
            assert near_match_variants(term) == [("exact", NOT_MAIN_NS)], term

    def test_colon_inside_an_article_title_is_kept(self):
        # The prefix regex is non-greedy (MediaWikiTitleCodec.php:301), so an
        # unrecognised prefix leaves the colon in the title.
        assert mw_normalize_title("Star Wars: Episode IV") == "Star Wars: Episode IV"

    def test_quoted_term_is_retried_last(self):
        # SearchNearMatcher.php:165-169 -- only after every variant failed.
        labels = self.labels('"Go"')
        assert labels[0] == "exact"
        assert labels[-2:] == ["dequote/exact", "dequote/uc"]

    def test_relative_and_tilde_titles_are_malformed(self):
        for bad in ["./x", "../x", "a/../b", "a~~~b"]:
            with pytest.raises(TitleError):
                mw_normalize_title(bad)

    def test_multibyte_first_letter_is_uppercased(self):
        assert mw_normalize_title("ödipus") == "Ödipus"


# ---------------------------------------------------------------------------
# 2. mini corpus
# ---------------------------------------------------------------------------

def _mini_records():
    """A synthetic KILT shard: 60 articles, one disambiguation page."""
    recs = []
    for i in range(58):
        title = f"Test Article {i:02d}"
        recs.append({
            "wikipedia_id": str(1000 + i),
            "wikipedia_title": title,
            "text": [
                title + "\n",
                f"{title} is a synthetic article number {i} used for testing "
                f"the frozen corpus. It mentions cartography and the year "
                f"{1900 + i}. Its third sentence names Aberdeen. Its fourth "
                f"sentence names Zanzibar. Its fifth sentence is filler. Its "
                f"sixth sentence must not appear in the observation.\n",
                "Section::::History.\n",
                f"A second paragraph for article {i} mentioning Aberdeen again.\n",
                "ok\n",  # two tokens -> dropped by the >2-word rule
            ],
        })
    recs.append({
        "wikipedia_id": "2000",
        "wikipedia_title": "Mercury",
        "text": ["Mercury\n",
                 "Mercury may refer to: the planet, the element, or the god.\n",
                 "Mercury is also a record label name.\n"],
    })
    recs.append({
        "wikipedia_id": "2001",
        "wikipedia_title": "Aberdeen",
        "text": ["Aberdeen\n",
                 "Aberdeen is a city in Scotland with a granite architecture "
                 "tradition. It sits between the rivers Dee and Don. It is the "
                 "third most populous city in Scotland. It has a harbour. It "
                 "hosts a university. This sentence is beyond the fifth.\n"],
    })
    return recs


@pytest.fixture(scope="module")
def mini_corpus(tmp_path_factory):
    raw_dir = tmp_path_factory.mktemp("raw")
    data_dir = tmp_path_factory.mktemp("data")
    recs = _mini_records()
    with open(raw_dir / "kilt_knowledgesource.json", "w", encoding="utf-8") as fh:
        for r in recs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    env = dict(os.environ)
    env.update(WIKI_DATA_DIR=str(data_dir), WIKI_RAW_DIR=str(raw_dir),
               WIKI_KS_URL="unused://mini", WIKI_SNAPSHOT="mini")
    for stage in ("pages", "bm25", "verify"):
        subprocess.run([sys.executable, BUILDER, "--stage", stage, "--verify-n", "20"],
                       check=True, env=env, cwd=REPO,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert (data_dir / "MANIFEST.json").exists()
    return str(data_dir), recs


def _backend(data_dir, mode):
    from src.local_wiki import LocalWiki
    return LocalWiki(data_dir=data_dir, mode=mode, bm25_mmap=False)


def _fresh_env():
    env = WikiEnv.__new__(WikiEnv)          # no LLM client, no network
    env.page = env.obs = None
    env.lookup_keyword = env.lookup_list = env.lookup_cnt = None
    env.search_time = 0.0
    env.num_searches = 0
    env.result_titles = []
    return env


class TestMiniCorpus:

    def test_exact_title_hit_sets_page_and_obs(self, mini_corpus):
        data_dir, recs = mini_corpus
        b = _backend(data_dir, "title_exact")
        env = _fresh_env()
        b.search_step(env, "Aberdeen")
        assert env.page.startswith("Aberdeen is a city in Scotland")
        # get_page_obs keeps five sentences and no more.
        assert env.obs.endswith("It hosts a university.")
        assert "beyond the fifth" not in env.obs
        assert env.lookup_keyword is None and env.lookup_list is None

    def test_page_text_matches_the_upstream_block_rule(self, mini_corpus):
        data_dir, recs = mini_corpus
        from src.local_wiki import kilt_paragraphs_to_page
        b = _backend(data_dir, "title_exact")
        env = _fresh_env()
        b.search_step(env, "Test Article 03")
        rec = [r for r in recs if r["wikipedia_title"] == "Test Article 03"][0]
        assert env.page == kilt_paragraphs_to_page(rec["text"])
        assert "Section::::History." not in env.page   # 1 token -> dropped
        assert "ok\n" not in env.page                  # 2 tokens -> dropped

    def test_case_variant_resolves(self, mini_corpus):
        data_dir, _ = mini_corpus
        b = _backend(data_dir, "title_exact")
        for q in ["aberdeen", "ABERDEEN", "  aberdeen  "]:
            env = _fresh_env()
            b.search_step(env, q)
            assert env.page.startswith("Aberdeen is a city"), q

    def test_title_exact_miss_uses_the_upstream_format(self, mini_corpus):
        data_dir, _ = mini_corpus
        b = _backend(data_dir, "title_exact")
        env = _fresh_env()
        env.page = "STALE PAGE TEXT\n"
        env.lookup_keyword, env.lookup_list, env.lookup_cnt = "x", ["y"], 1
        b.search_step(env, "Cartography Of Aberdeen In 1907")
        assert env.obs.startswith("Could not find Cartography Of Aberdeen In 1907. "
                                  "Similar: [")
        assert env.obs.endswith("].")
        assert env.result_titles and len(env.result_titles) <= 5
        # Upstream leaves page and lookup state alone on a miss. So do we --
        # that stale page is the contamination channel the gate probes.
        assert env.page == "STALE PAGE TEXT\n"
        assert (env.lookup_keyword, env.lookup_cnt) == ("x", 1)

    def test_miss_observation_is_exactly_upstreams_f_string(self, mini_corpus):
        data_dir, _ = mini_corpus
        b = _backend(data_dir, "title_exact")
        env = _fresh_env()
        entity = "No Such Page Whatsoever About Cartography"
        b.search_step(env, entity)
        expected = f"Could not find {entity}. Similar: {env.result_titles[:5]}."
        assert env.obs == expected

    def test_bm25_miss_returns_the_top_1_page(self, mini_corpus):
        data_dir, _ = mini_corpus
        b = _backend(data_dir, "bm25")
        env = _fresh_env()
        b.search_step(env, "granite architecture city in Scotland harbour")
        assert "Could not find" not in env.obs
        assert env.page.startswith("Aberdeen is a city in Scotland")
        assert env.lookup_keyword is None

    def test_the_two_backends_differ_only_on_misses(self, mini_corpus):
        data_dir, _ = mini_corpus
        te = _backend(data_dir, "title_exact")
        bm = _backend(data_dir, "bm25")
        miss = "granite architecture city in Scotland harbour"
        e1, e2 = _fresh_env(), _fresh_env()
        te.search_step(e1, miss)
        bm.search_step(e2, miss)
        assert e1.obs != e2.obs
        assert e1.obs.startswith("Could not find")
        assert not e2.obs.startswith("Could not find")

    def test_50_exact_titles_are_byte_identical_across_backends(self, mini_corpus):
        data_dir, recs = mini_corpus
        te = _backend(data_dir, "title_exact")
        bm = _backend(data_dir, "bm25")
        titles = [r["wikipedia_title"] for r in recs
                  if r["wikipedia_title"] != "Mercury"][:50]
        assert len(titles) == 50
        for t in titles:
            e1, e2 = _fresh_env(), _fresh_env()
            te.search_step(e1, t)
            bm.search_step(e2, t)
            assert e1.obs == e2.obs, t
            assert e1.page == e2.page, t
            assert not e1.obs.startswith("Could not find"), t

    def test_disambiguation_page_is_re_searched_like_upstream(self, mini_corpus):
        data_dir, _ = mini_corpus
        b = _backend(data_dir, "title_exact")
        env = _fresh_env()
        b.search_step(env, "Mercury")
        # environment.py re-searches "[Mercury]"; '[' is an illegal title
        # character, so the frozen corpus reports a miss -- and the miss text
        # carries the brackets, exactly as upstream's recursion does.
        assert env.obs.startswith("Could not find [Mercury]. Similar: [")
        assert "may refer to" not in env.obs

    def test_lookup_runs_off_the_local_page(self, mini_corpus):
        data_dir, _ = mini_corpus
        b = _backend(data_dir, "title_exact")
        env = _fresh_env()
        env.steps = 0
        env.answer = None
        b.search_step(env, "Test Article 07")
        obs = WikiEnv.step(env, "lookup[Aberdeen]")[0]
        assert obs.startswith("(Result 1 / 2)")
        assert "Aberdeen" in obs

    def test_search_counters_are_updated(self, mini_corpus):
        data_dir, _ = mini_corpus
        b = _backend(data_dir, "title_exact")
        env = _fresh_env()
        b.search_step(env, "Aberdeen")
        b.search_step(env, "Nothing At All Here")
        assert env.num_searches == 2
        assert env.search_time > 0

    def test_backend_dispatch_through_wikienv(self, mini_corpus, monkeypatch):
        data_dir, _ = mini_corpus
        from src import local_wiki
        monkeypatch.setitem(local_wiki._BACKENDS, ("title_exact", data_dir),
                            _backend(data_dir, "title_exact"))
        monkeypatch.setenv("WIKI_DATA_DIR", data_dir)
        monkeypatch.setattr(constants, "retrieval_backend", "title_exact")
        env = _fresh_env()
        env.steps = 0
        env.answer = None
        obs = WikiEnv.step(env, "search[Aberdeen]")[0]
        assert obs.startswith("Aberdeen is a city in Scotland")

    def test_unknown_backend_is_an_error(self, monkeypatch):
        monkeypatch.setattr(constants, "retrieval_backend", "nope")
        env = _fresh_env()
        with pytest.raises(ValueError, match="retrieval_backend"):
            WikiEnv.search_step(env, "Aberdeen")


# ---------------------------------------------------------------------------
# 3. the real 2019-08-01 build
# ---------------------------------------------------------------------------

def _real_data_dir():
    d = os.environ.get("WIKI_DATA_DIR")
    if not d or not os.path.exists(os.path.join(d, "MANIFEST.json")):
        return None
    with open(os.path.join(d, "MANIFEST.json")) as fh:
        man = json.load(fh)
    if not man["stages"].get("bm25", {}).get("complete"):
        return None
    return d


real_corpus = pytest.mark.skipif(
    _real_data_dir() is None,
    reason="frozen corpus not built on this node (see docs/LOCAL_WIKI.md)")


@pytest.fixture(scope="module")
def backends():
    d = _real_data_dir()
    from src.local_wiki import LocalWiki
    return (LocalWiki(data_dir=d, mode="title_exact"),
            LocalWiki(data_dir=d, mode="bm25"))


@real_corpus
class TestRealCorpus:

    def _sample_titles(self, store, want, skip_disambig=True):
        """`want` titles spread across the corpus by pid."""
        out = []
        n = store.n_pages
        step = max(1, n // (want * 2))
        pid = 0
        while len(out) < want and pid < n:
            title = store.title_of(pid)
            pid += step
            if title is None:
                continue
            if skip_disambig:
                from src.local_wiki import DISAMBIG_MARKER
                page = store.page_text(title)
                if page is None or DISAMBIG_MARKER in page:
                    continue
            out.append(title)
        return out

    def test_50_exact_titles_are_byte_identical_across_backends(self, backends):
        """The acceptance test: on a query that resolves to an article, the two
        backends are indistinguishable.

        Disambiguation titles are excluded, and that exclusion is not a
        loophole -- it is the one designed consequence of the miss rule, pinned
        by the next test. Everything else must match byte for byte.
        """
        te, bm = backends
        titles = self._sample_titles(te.store, 50)
        assert len(titles) == 50
        for t in titles:
            e1, e2 = _fresh_env(), _fresh_env()
            te.search_step(e1, t)
            bm.search_step(e2, t)
            assert e1.obs == e2.obs, t
            assert e1.page == e2.page, t
            assert not e1.obs.startswith("Could not find"), t

    def test_disambiguation_titles_are_the_one_designed_divergence(self, backends):
        """A disambiguation title becomes a MISS via upstream's "[X]" re-search,
        so the miss rule applies and the modes diverge -- by construction.

        "USS Calypso" is a 2019-08-01 set-index page.
        """
        te, bm = backends
        e1, e2 = _fresh_env(), _fresh_env()
        te.search_step(e1, "USS Calypso")
        bm.search_step(e2, "USS Calypso")
        assert e1.obs.startswith("Could not find [USS Calypso].")
        assert not e2.obs.startswith("Could not find")

    def test_known_hotpotqa_titles_resolve(self, backends):
        te, _ = backends
        for t in ["Scott Derrickson", "Ed Wood", "Aberdeen",
                  # the 2019-08-01 title; enwiki renamed it to
                  # "Queer as Folk (2000 TV series)" after the snapshot, which
                  # is exactly the drift the frozen corpus removes.
                  "Queer as Folk (American TV series)"]:
            env = _fresh_env()
            te.search_step(env, t)
            assert not env.obs.startswith("Could not find"), t

    def test_case_variants_resolve_on_the_real_corpus(self, backends):
        te, _ = backends
        for q in ["scott derrickson", "SCOTT DERRICKSON", "scott Derrickson"]:
            env = _fresh_env()
            te.search_step(env, q)
            assert not env.obs.startswith("Could not find"), q

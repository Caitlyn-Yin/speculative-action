"""Frozen local Wikipedia behind the `WikiEnv.search_step` interface.

Motivation. `WikiEnv.search_step` hits live `en.wikipedia.org`. That makes every
trajectory a function of what Wikipedia happened to say at that moment, so a
replay months later is not the same experiment, and the 3-arm isolation gate
cannot tell "the arms differ" from "the encyclopedia changed between arms".
This module replaces the network with a frozen 2019-08-01 snapshot (KILT) and
offers two retrieval modes that differ in **exactly one behaviour**:

  title_exact  a query that resolves to no page is a MISS: the agent gets
               "Could not find X. Similar: [top-5 BM25 titles]." -- the same
               shape live Wikipedia's result list produces.
  bm25         a query that resolves to no page is silently answered with the
               top-1 BM25 page. The agent never sees a miss.

Everything else -- page text, `get_page_obs` truncation, `lookup[...]`,
the disambiguation re-search, the observation strings -- is shared, so a
title_exact/bm25 contrast isolates "does the retriever ever say no".

`retrieval_backend` selects between these and `live` (constants.py). The class
is a *collaborator* of `WikiEnv`, not a subclass: `search_step(env, entity)`
mutates the same attributes upstream's does, in the same cases, so
`runner._snapshot_env` and `HotPotQAWrapper` need no change.

See docs/LOCAL_WIKI.md for the design, the fidelity measurement against live
Wikipedia, and the enumerated deviations.
"""

import json
import os
import sqlite3
import threading
import time

from .mw_title import NOT_MAIN_NS, TitleError, mw_normalize_title, near_match_variants

# ---------------------------------------------------------------------------
# on-disk layout (also consumed by scripts/build_local_wiki.py)
# ---------------------------------------------------------------------------
PAGES_FILE = "pages.zst"
SQLITE_FILE = "pages.sqlite"
LEADS_FILE = "leads.jsonl.zst"
BM25_DIR = "bm25"
MANIFEST_FILE = "MANIFEST.json"

#: pages per independently-compressed zstd frame. 512 keeps a frame around
#: 1.5 MB uncompressed -- small enough that decompressing one to read one page
#: costs ~2 ms, large enough that zstd sees cross-document redundancy (a
#: per-page frame is ~45% of plaintext, a 512-page frame ~29%).
CHUNK_PAGES = 512

#: marker for "this page exists in the corpus but the retriever should treat it
#: as a disambiguation page", matching upstream's test in environment.py.
DISAMBIG_MARKER = "may refer to:"


# ---------------------------------------------------------------------------
# text construction -- shared by the builder and the tests
# ---------------------------------------------------------------------------

def kilt_paragraphs_to_page(paras):
    """Build the `WikiEnv.page` string from a KILT record's `text` list.

    Mirrors `environment.WikiEnv.search_step`'s HTML path
    (`hotpotqa/src/environment.py`, the `else:` branch):

        page = [p.get_text().strip() for p in soup.find_all("p") + soup.find_all("ul")]
        ...
        for p in page:
            if len(p.split(" ")) > 2:
                self.page += clean_str(p)
                if not p.endswith("\\n"):
                    self.page += "\\n"

    i.e. strip each block, keep those with more than two space-separated
    tokens, and newline-terminate. `text[0]` is dropped: KILT stores the page
    title as its own first paragraph, which has no counterpart in the rendered
    HTML's `<p>` elements.

    `clean_str` is deliberately NOT applied -- see docs/LOCAL_WIKI.md
    "Deviations": it is a `unicode-escape`/latin1 round-trip that repairs
    mojibake in scraped HTML and corrupts already-correct UTF-8.
    """
    out = []
    for p in paras[1:]:
        p = p.strip()
        if len(p.split(" ")) > 2:
            out.append(p)
            out.append("\n")
    return "".join(out)


def lead_paragraph(paras):
    """The lead paragraph of a KILT record (`text[1]`), or ''.

    `text[0]` is the title line; `text[1]` is the first body paragraph, which is
    what `get_page_obs` would surface and what the BM25 index is built over.
    """
    if len(paras) > 1:
        return paras[1].strip()
    return ""


def norm_key(title):
    """The page-table key for a title: its MediaWiki-normalised form, or None."""
    try:
        k = mw_normalize_title(title)
    except TitleError:
        return None
    if k is NOT_MAIN_NS:
        return None
    return k


def open_pages_db(data_dir):
    path = os.path.join(data_dir, SQLITE_FILE)
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True, check_same_thread=False)
    return con


# ---------------------------------------------------------------------------
# page store
# ---------------------------------------------------------------------------

class LocalWikiStore:
    """Title -> page text over the chunked zstd store."""

    def __init__(self, data_dir):
        import zstandard as zstd

        self.data_dir = data_dir
        self.con = open_pages_db(data_dir)
        self._dctx = zstd.ZstdDecompressor()
        self._pages_path = os.path.join(data_dir, PAGES_FILE)
        self._fh = open(self._pages_path, "rb")
        self._lock = threading.Lock()
        self._frame_cache = {}          # frame_off -> decoded list of page texts
        self._frame_cache_order = []
        self._frame_cache_max = 8
        row = self.con.execute("SELECT v FROM meta WHERE k='n_pages'").fetchone()
        self.n_pages = int(row[0]) if row else 0

    # -- lookup -------------------------------------------------------------
    def resolve(self, key):
        """Normalised title -> pid, or None."""
        row = self.con.execute("SELECT pid FROM titles WHERE norm=?", (key,)).fetchone()
        return row[0] if row else None

    def title_of(self, pid):
        row = self.con.execute("SELECT title FROM pages WHERE pid=?", (pid,)).fetchone()
        return row[0] if row else None

    def page_text_by_pid(self, pid):
        row = self.con.execute(
            "SELECT frame_off, frame_len, item_idx FROM pages WHERE pid=?",
            (pid,)).fetchone()
        if row is None:
            return None
        off, ln, idx = row
        return self._frame(off, ln)[idx]

    def page_text(self, title):
        key = norm_key(title)
        if key is None:
            return None
        pid = self.resolve(key)
        if pid is None:
            return None
        return self.page_text_by_pid(pid)

    # -- frames -------------------------------------------------------------
    def _frame(self, off, ln):
        with self._lock:
            hit = self._frame_cache.get(off)
            if hit is not None:
                return hit
            self._fh.seek(off)
            blob = self._fh.read(ln)
            items = json.loads(self._dctx.decompress(blob).decode("utf-8"))
            self._frame_cache[off] = items
            self._frame_cache_order.append(off)
            if len(self._frame_cache_order) > self._frame_cache_max:
                self._frame_cache.pop(self._frame_cache_order.pop(0), None)
            return items


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------

class LocalBm25:
    """bm25s index over `title + '. ' + lead paragraph`."""

    def __init__(self, data_dir, mmap=True):
        import bm25s

        self._bm25s = bm25s
        d = os.path.join(data_dir, BM25_DIR)
        self.retriever = bm25s.BM25.load(d, mmap=mmap, load_corpus=False)
        with open(os.path.join(d, "doc_titles.json"), encoding="utf-8") as fh:
            self.titles = json.load(fh)["titles"]
        with open(os.path.join(d, "doc_pids.json")) as fh:
            self.pids = json.load(fh)["pids"]
        man = os.path.join(data_dir, MANIFEST_FILE)
        self.stemmer = None
        if os.path.exists(man):
            with open(man) as fh:
                tok = json.load(fh)["stages"].get("bm25", {}).get("tokenizer", "")
            if "PyStemmer" in tok:
                import Stemmer
                self.stemmer = Stemmer.Stemmer("english")

    def top_k(self, query, k=5):
        """[(title, pid, score)] for the k best documents, best first."""
        tokens = self._bm25s.tokenize([query], stopwords="en", stemmer=self.stemmer,
                                      show_progress=False)
        try:
            idx, scores = self.retriever.retrieve(tokens, k=k, show_progress=False,
                                                  n_threads=1)
        except ValueError:
            # bm25s raises when the query has no in-vocabulary token.
            return []
        out = []
        for j in range(idx.shape[1]):
            i = int(idx[0, j])
            out.append((self.titles[i], self.pids[i], float(scores[0, j])))
        return out


# ---------------------------------------------------------------------------
# the backend
# ---------------------------------------------------------------------------

class LocalWiki:
    """Frozen-corpus replacement for `WikiEnv.search_step`.

    `mode` is "title_exact" or "bm25"; they share every code path except
    :meth:`_on_miss`.
    """

    def __init__(self, data_dir=None, mode="title_exact", bm25_mmap=True):
        if mode not in ("title_exact", "bm25"):
            raise ValueError(f"unknown local retrieval mode {mode!r}")
        self.mode = mode
        self.data_dir = data_dir or os.environ.get("WIKI_DATA_DIR") or ""
        if not self.data_dir:
            raise RuntimeError(
                "WIKI_DATA_DIR is unset; `source scripts/env.sh` or pass data_dir")
        self.store = LocalWikiStore(self.data_dir)
        self._bm25 = None
        self._bm25_mmap = bm25_mmap
        self.snapshot = "unknown"
        man = os.path.join(self.data_dir, MANIFEST_FILE)
        if os.path.exists(man):
            with open(man) as fh:
                self.snapshot = json.load(fh).get("snapshot", "unknown")

    @property
    def bm25(self):
        if self._bm25 is None:
            self._bm25 = LocalBm25(self.data_dir, mmap=self._bm25_mmap)
        return self._bm25

    # -- resolution ---------------------------------------------------------
    def resolve(self, entity):
        """Run the MediaWiki near-match ladder against the frozen page table.

        Returns ``(pid, title, variant_label)`` or ``(None, None, reason)``
        where reason is one of ``no_candidate`` (malformed / empty / '#'-prefixed
        term), ``not_main_ns`` (namespace or interwiki prefix), ``no_match``.
        """
        cands = near_match_variants(entity)
        if not cands:
            return None, None, "no_candidate"
        if cands[0][1] is NOT_MAIN_NS:
            return None, None, "not_main_ns"
        for label, cand in cands:
            pid = self.store.resolve(cand)
            if pid is not None:
                return pid, self.store.title_of(pid), label
        return None, None, "no_match"

    # -- the WikiEnv-compatible entry point ---------------------------------
    def search_step(self, env, entity, _depth=0):
        """Set `env.page` / `env.obs` / `env.result_titles` / lookup state.

        Mirrors `WikiEnv.search_step` case for case:

        * hit  -> `page`, `obs = get_page_obs(page)`, and
          `lookup_keyword = lookup_list = lookup_cnt = None`.
        * a page whose text contains "may refer to:" -> re-search the
          bracketed term, exactly as upstream does.
        * miss -> `result_titles` and
          `obs = f"Could not find {entity}. Similar: {result_titles[:5]}."`
          Upstream leaves `page` and the lookup state untouched on a miss; so
          does this. (That is how a stale page survives a failed search, which
          is precisely the contamination channel the isolation gate probes.)
        """
        t0 = time.time()
        try:
            pid, title, why = self.resolve(entity)
            if pid is None:
                self._on_miss(env, entity, why)
                return

            page = self.store.page_text_by_pid(pid)

            # environment.py: `if any("may refer to:" in p for p in page)` is
            # tested over the extracted blocks before they are joined; the
            # joined text contains the same substring iff some block does.
            if DISAMBIG_MARKER in page and _depth == 0:
                self.search_step(env, "[" + entity + "]", _depth=1)
                return

            env.page = page
            env.obs = env.get_page_obs(env.page)
            env.lookup_keyword = env.lookup_list = env.lookup_cnt = None
        finally:
            env.search_time += time.time() - t0
            env.num_searches += 1

    def _on_miss(self, env, entity, why):
        """The ONE behaviour the two modes do not share."""
        if self.mode == "bm25":
            hits = self.bm25.top_k(entity, k=1)
            if hits:
                pid = hits[0][1]
                env.page = self.store.page_text_by_pid(pid)
                env.obs = env.get_page_obs(env.page)
                env.lookup_keyword = env.lookup_list = env.lookup_cnt = None
                return
            # No in-vocabulary token at all: there is no page to fall back to,
            # so bm25 degrades to the title_exact miss observation.
        titles = [t for t, _pid, _s in self.bm25.top_k(entity, k=5)]
        env.result_titles = titles
        env.obs = f"Could not find {entity}. Similar: {env.result_titles[:5]}."


# ---------------------------------------------------------------------------
# process-wide cache: the sqlite handle and the mmapped BM25 index are
# expensive to open and safe to share across episodes.
# ---------------------------------------------------------------------------
_BACKENDS = {}
_BACKENDS_LOCK = threading.Lock()


def get_local_wiki(mode, data_dir=None):
    key = (mode, data_dir or os.environ.get("WIKI_DATA_DIR", ""))
    with _BACKENDS_LOCK:
        if key not in _BACKENDS:
            _BACKENDS[key] = LocalWiki(data_dir=data_dir, mode=mode)
        return _BACKENDS[key]

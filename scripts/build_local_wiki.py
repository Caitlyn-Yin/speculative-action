#!/usr/bin/env python
"""Build the frozen local Wikipedia from the KILT knowledge source.

    source scripts/env.sh
    $PIPELINE_PY scripts/build_local_wiki.py --stage all

Stages, each independently idempotent (a stage whose outputs are already
recorded complete in MANIFEST.json is skipped unless --force):

  fetch    download kilt_knowledgesource.json to $WIKI_RAW_DIR (scratch, /tmp)
           and verify it against the md5 published in the S3 object metadata
  pages    stream the jsonl once, writing
              pages.zst      independently-compressed zstd frames, 512 pages each
              pages.sqlite   title -> (frame offset, frame length, item index)
              leads.jsonl.zst  title + lead paragraph, the BM25 build input
  bm25     build the bm25s index over title + lead paragraph -> bm25/
  verify   re-read a sample of pages through the runtime reader

Everything but `fetch` writes to $WIKI_DATA_DIR on persistent storage. The raw
34.8 GiB download is scratch: it is only needed to rebuild, and the MANIFEST
records its hashes and URL so the rebuild is checkable.

Page text is built to be byte-identical with what `WikiEnv.search_step` would
have produced from the rendered HTML, given the same paragraphs -- see
docs/LOCAL_WIKI.md for the one place that is impossible (clean_str) and why.
"""

import argparse
import hashlib
import io
import json
import os
import sqlite3
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "hotpotqa"))

from src import durability  # noqa: E402
from src.local_wiki import (  # noqa: E402
    CHUNK_PAGES, PAGES_FILE, SQLITE_FILE, LEADS_FILE, BM25_DIR, MANIFEST_FILE,
    kilt_paragraphs_to_page, lead_paragraph, norm_key, open_pages_db,
)

KS_MD5_PUBLISHED = "d1dca62aa6ba889d2e842182e3114af5"   # x-amz-meta-s3cmd-attrs
KS_SIZE_PUBLISHED = 37318876722


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_manifest(data_dir):
    path = os.path.join(data_dir, MANIFEST_FILE)
    if os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)
    return {"snapshot": os.environ.get("WIKI_SNAPSHOT", "kilt_20190801"), "stages": {}}


def save_manifest(data_dir, man):
    path = os.path.join(data_dir, MANIFEST_FILE)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(man, fh, indent=2, sort_keys=True)
        fh.write("\n")
    os.replace(tmp, path)


def file_hashes(path, algos=("sha256", "md5"), bufsize=1 << 24):
    hs = {a: hashlib.new(a) for a in algos}
    n = 0
    with open(path, "rb") as fh:
        while True:
            b = fh.read(bufsize)
            if not b:
                break
            n += len(b)
            for h in hs.values():
                h.update(b)
    return n, {a: h.hexdigest() for a, h in hs.items()}


# ---------------------------------------------------------------------------
# fetch
# ---------------------------------------------------------------------------

def stage_fetch(args, man):
    raw = os.path.join(os.environ["WIKI_RAW_DIR"], "kilt_knowledgesource.json")
    url = os.environ["WIKI_KS_URL"]
    os.makedirs(os.environ["WIKI_RAW_DIR"], exist_ok=True)

    if not (os.path.exists(raw) and os.path.getsize(raw) == KS_SIZE_PUBLISHED):
        log(f"downloading {url} -> {raw} ({KS_SIZE_PUBLISHED/2**30:.2f} GiB)")
        subprocess.check_call(["curl", "-sSL", "--retry", "5", "--retry-delay", "5",
                               "-C", "-", "-o", raw, url])
    size = os.path.getsize(raw)
    if size != KS_SIZE_PUBLISHED:
        raise SystemExit(f"size mismatch: {size} != {KS_SIZE_PUBLISHED}")

    prev = man["stages"].get("fetch", {})
    if prev.get("sha256") and not args.force:
        log("fetch: hashes already in MANIFEST, skipping re-hash")
        return
    log("hashing the raw knowledge source (sha256 + md5, ~35 GiB)")
    n, hs = file_hashes(raw)
    if hs["md5"] != KS_MD5_PUBLISHED:
        raise SystemExit(f"md5 mismatch: {hs['md5']} != {KS_MD5_PUBLISHED} (published)")
    man["stages"]["fetch"] = {
        "source_url": url,
        "bytes": n,
        "sha256": hs["sha256"],
        "md5": hs["md5"],
        "md5_published": KS_MD5_PUBLISHED,
        "md5_matches_published": True,
        "raw_path": raw,
        "raw_path_is_scratch": True,
        "build_command": f"curl -sSL -C - -o {raw} {url}",
        "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    log(f"fetch ok: sha256={hs['sha256']}")


# ---------------------------------------------------------------------------
# pages
# ---------------------------------------------------------------------------

# The build writes to a work directory on the container overlay and *publishes*
# the finished artifacts to $WIKI_DATA_DIR. Building in place on JuiceFS is not
# viable: inserting 5.9 M rows into a TEXT-primary-key table over a network
# filesystem is random-page-write bound and was measured at ~1 MB/s of output
# (vs ~80 MB/s of input on the overlay). Two consequences, both deliberate:
#   * titles are accumulated in RAM and inserted in sorted order, so the B-tree
#     is filled sequentially instead of randomly;
#   * the sqlite file and the zstd store are copied over once, at the end.
def work_dir(args):
    d = args.work_dir or os.path.join(os.environ["WIKI_RAW_DIR"], "build")
    os.makedirs(d, exist_ok=True)
    return d


def publish(src, dst):
    """Copy a finished artifact onto persistent storage, atomically."""
    import shutil
    tmp = dst + ".tmp"
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)
    return os.path.getsize(dst)


DDL = """
PRAGMA journal_mode = OFF;
PRAGMA synchronous = OFF;
PRAGMA cache_size = -2000000;
CREATE TABLE IF NOT EXISTS pages (
    pid          INTEGER PRIMARY KEY,
    title        TEXT NOT NULL,
    wikipedia_id TEXT,
    frame_off    INTEGER NOT NULL,
    frame_len    INTEGER NOT NULL,
    item_idx     INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS titles (
    norm TEXT PRIMARY KEY,
    pid  INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT);
"""


def stage_pages(args, man):
    import zstandard as zstd

    data_dir = os.environ["WIKI_DATA_DIR"]
    wd = work_dir(args)
    raw = os.path.join(os.environ["WIKI_RAW_DIR"], "kilt_knowledgesource.json")

    if man["stages"].get("pages", {}).get("complete") and not args.force:
        log("pages: complete in MANIFEST, skipping")
        return
    if not os.path.exists(raw):
        raise SystemExit("pages stage needs the raw download; run --stage fetch first")

    w_pages = os.path.join(wd, PAGES_FILE)
    w_db = os.path.join(wd, SQLITE_FILE)
    w_leads = os.path.join(wd, LEADS_FILE)
    for p in (w_pages, w_db, w_leads):
        if os.path.exists(p):
            os.remove(p)

    cctx = zstd.ZstdCompressor(level=args.zstd_level, threads=args.zstd_threads)
    lead_cctx = zstd.ZstdCompressor(level=6)

    con = sqlite3.connect(w_db)
    con.executescript(DDL)

    pages_fh = open(w_pages, "wb")
    leads_raw = open(w_leads, "wb")
    leads_fh = lead_cctx.stream_writer(leads_raw)

    batch = []              # page texts for the frame being filled
    batch_rows = []         # (title, wikipedia_id, pid) aligned with batch
    page_rows = []
    title_rows = []         # (norm, pid), sorted and inserted at the end
    pid = 0
    offset = 0
    n_bad_title = 0
    t0 = time.time()

    def flush_frame():
        nonlocal offset, batch, batch_rows
        if not batch:
            return
        frame = cctx.compress(json.dumps(batch, ensure_ascii=False).encode("utf-8"))
        pages_fh.write(frame)
        for i, (title, wid, p) in enumerate(batch_rows):
            page_rows.append((p, title, wid, offset, len(frame), i))
        offset += len(frame)
        batch = []
        batch_rows = []

    def flush_pages():
        if page_rows:
            con.executemany(
                "INSERT INTO pages(pid,title,wikipedia_id,frame_off,frame_len,item_idx)"
                " VALUES (?,?,?,?,?,?)", page_rows)
            page_rows.clear()

    with open(raw, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            title = rec["wikipedia_title"]
            paras = rec.get("text") or []

            batch.append(kilt_paragraphs_to_page(paras))
            batch_rows.append((title, str(rec.get("wikipedia_id") or ""), pid))

            key = norm_key(title)
            if key is None:
                n_bad_title += 1
            else:
                title_rows.append((key, pid))

            leads_fh.write(json.dumps(
                {"pid": pid, "title": title, "lead": lead_paragraph(paras)},
                ensure_ascii=False).encode("utf-8") + b"\n")

            pid += 1
            if len(batch) >= CHUNK_PAGES:
                flush_frame()
            if len(page_rows) >= 200_000:
                flush_pages()
                el = time.time() - t0
                log(f"  {pid:,} pages, {offset/2**30:.2f} GiB compressed, "
                    f"{pid/el:,.0f} pages/s")

    flush_frame()
    flush_pages()
    con.commit()
    leads_fh.close()
    leads_raw.close()
    pages_fh.close()

    # Sorted bulk insert: filling a TEXT primary key in key order keeps the
    # B-tree append-only. Unsorted, this step alone dominated the build.
    log(f"inserting {len(title_rows):,} title keys in sorted order")
    title_rows.sort()
    n_dup = 0
    seen_prev = None
    dedup = []
    for k, p in title_rows:
        if k == seen_prev:
            n_dup += 1
            continue
        seen_prev = k
        dedup.append((k, p))
    con.executemany("INSERT INTO titles(norm,pid) VALUES (?,?)", dedup)
    con.executemany("INSERT OR REPLACE INTO meta(k,v) VALUES (?,?)", [
        ("n_pages", str(pid)),
        ("chunk_pages", str(CHUNK_PAGES)),
        ("snapshot", os.environ.get("WIKI_SNAPSHOT", "kilt_20190801")),
    ])
    con.commit()
    con.execute("VACUUM")
    con.close()

    log("publishing to persistent storage")
    n_pages_bytes = publish(w_pages, os.path.join(data_dir, PAGES_FILE))
    n_db_bytes = publish(w_db, os.path.join(data_dir, SQLITE_FILE))
    n_leads_bytes = publish(w_leads, os.path.join(data_dir, LEADS_FILE))

    man["stages"]["pages"] = {
        "complete": True,
        "n_pages": pid,
        "n_titles_indexed": len(dedup),
        "n_titles_unnormalisable": n_bad_title,
        "n_titles_duplicate_after_normalisation": n_dup,
        "chunk_pages": CHUNK_PAGES,
        "zstd_level": args.zstd_level,
        "pages_zst_bytes": n_pages_bytes,
        "sqlite_bytes": n_db_bytes,
        "leads_bytes": n_leads_bytes,
        "work_dir": wd,
        "build_command": "$PIPELINE_PY scripts/build_local_wiki.py --stage pages",
        "seconds": round(time.time() - t0, 1),
        "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    log(f"pages ok: {pid:,} pages, {n_pages_bytes/2**30:.2f} GiB, "
        f"{time.time()-t0:.0f}s")


# ---------------------------------------------------------------------------
# bm25
# ---------------------------------------------------------------------------

def stage_bm25(args, man):
    import bm25s
    import zstandard as zstd

    data_dir = os.environ["WIKI_DATA_DIR"]
    leads_path = os.path.join(data_dir, LEADS_FILE)
    out_dir = os.path.join(data_dir, BM25_DIR)

    if man["stages"].get("bm25", {}).get("complete") and not args.force:
        log("bm25: complete in MANIFEST, skipping")
        return

    t0 = time.time()
    log("reading title + lead paragraph")
    docs = []
    pids = []
    titles = []
    dctx = zstd.ZstdDecompressor()
    with open(leads_path, "rb") as raw, dctx.stream_reader(raw) as zr:
        for line in io.TextIOWrapper(zr, encoding="utf-8"):
            rec = json.loads(line)
            pids.append(rec["pid"])
            titles.append(rec["title"])
            docs.append(rec["title"] + ". " + rec["lead"])
    log(f"  {len(docs):,} documents")

    stemmer = None
    stemmer_name = "none"
    if not args.no_stemmer:
        try:
            import Stemmer
            stemmer = Stemmer.Stemmer("english")
            stemmer_name = "PyStemmer/english(snowball)"
        except ImportError:
            log("  PyStemmer unavailable -> unstemmed index")

    log("tokenising")
    tokens = bm25s.tokenize(docs, stopwords="en", stemmer=stemmer,
                            show_progress=False)
    del docs
    log("indexing")
    import shutil
    retriever = bm25s.BM25(method=args.bm25_method)
    retriever.index(tokens, show_progress=False)
    # Same reason as stage_pages: write to the overlay, publish once.
    w_out = os.path.join(work_dir(args), BM25_DIR)
    if os.path.exists(w_out):
        shutil.rmtree(w_out)
    retriever.save(w_out)
    with open(os.path.join(w_out, "doc_pids.json"), "w") as fh:
        json.dump({"pids": pids}, fh)
    with open(os.path.join(w_out, "doc_titles.json"), "w", encoding="utf-8") as fh:
        json.dump({"titles": titles}, fh, ensure_ascii=False)
    log("publishing the index to persistent storage")
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    shutil.copytree(w_out, out_dir)

    total = sum(os.path.getsize(os.path.join(out_dir, f)) for f in os.listdir(out_dir))
    man["stages"]["bm25"] = {
        "complete": True,
        "library": "bm25s " + getattr(bm25s, "__version__", "unknown"),
        "method": args.bm25_method,
        "tokenizer": "bm25s.tokenize(stopwords='en', stemmer=%s), "
                     "regex splitter r'(?u)\\\\w+'" % stemmer_name,
        "fields": "wikipedia_title + '. ' + lead paragraph (KILT text[1])",
        "n_docs": len(pids),
        "index_bytes": total,
        "build_command": "$PIPELINE_PY scripts/build_local_wiki.py --stage bm25",
        "seconds": round(time.time() - t0, 1),
        "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    log(f"bm25 ok: {total/2**30:.2f} GiB in {time.time()-t0:.0f}s")


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------

def stage_verify(args, man):
    from src.local_wiki import LocalWikiStore

    store = LocalWikiStore(os.environ["WIKI_DATA_DIR"])
    con = open_pages_db(os.environ["WIKI_DATA_DIR"])
    n = con.execute("SELECT v FROM meta WHERE k='n_pages'").fetchone()[0]
    n = int(n)
    step = max(1, n // args.verify_n)
    checked = empty = 0
    for pid in range(0, n, step):
        row = con.execute("SELECT title FROM pages WHERE pid=?", (pid,)).fetchone()
        if row is None:
            raise SystemExit(f"pid {pid} missing from pages")
        title = row[0]
        page = store.page_text(title)
        if page is None:
            raise SystemExit(f"title {title!r} did not resolve through the store")
        checked += 1
        if not page.strip():
            empty += 1
    log(f"verify ok: {checked} sampled titles resolved, {empty} with empty text")
    man["stages"]["verify"] = {
        "sampled": checked, "empty_text": empty,
        "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["all", "fetch", "pages", "bm25", "verify"])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--zstd-level", type=int, default=6)
    ap.add_argument("--zstd-threads", type=int, default=0)
    ap.add_argument("--bm25-method", default="lucene")
    ap.add_argument("--no-stemmer", action="store_true")
    ap.add_argument("--verify-n", type=int, default=200)
    ap.add_argument("--work-dir", default=None,
                    help="scratch build dir on fast local disk "
                         "(default $WIKI_RAW_DIR/build)")
    args = ap.parse_args()

    for var in ("WIKI_DATA_DIR", "WIKI_RAW_DIR", "WIKI_KS_URL"):
        if var not in os.environ:
            raise SystemExit(f"{var} unset -- `source scripts/env.sh` first")

    # The built corpus is the one output here, and it costs a 34.8 GiB download
    # plus ~27 min of CPU to rebuild -- hence persistent. $WIKI_RAW_DIR is
    # deliberately NOT guarded: the raw download is scratch by design, needed
    # only while building.
    durability.require_durable_outputs(
        wiki_data_dir=os.environ["WIKI_DATA_DIR"])

    os.makedirs(os.environ["WIKI_DATA_DIR"], exist_ok=True)

    man = load_manifest(os.environ["WIKI_DATA_DIR"])
    stages = ["fetch", "pages", "bm25", "verify"] if args.stage == "all" else [args.stage]
    for st in stages:
        log(f"=== stage {st} ===")
        globals()["stage_" + st](args, man)
        save_manifest(os.environ["WIKI_DATA_DIR"], man)
    log("done")


if __name__ == "__main__":
    main()

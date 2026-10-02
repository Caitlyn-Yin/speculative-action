#!/usr/bin/env python
"""Measure the frozen local corpus against live Wikipedia on 300 queries.

    source scripts/env.sh
    $PIPELINE_PY scripts/wiki_fidelity.py

Three query sets of 100, chosen because they fail differently:

  gold       HotpotQA dev gold supporting-fact titles. Clean, canonical, and
             the thing the benchmark's answers actually depend on -- a miss here
             is a benchmark-breaking miss.
  perturbed  the same titles, lowercased / one word dropped / one word added.
             Probes the near-match case ladder and the BM25 fallback.
  agent      queries a real agent issued (hotpotqa/cache/wiki/queries.jsonl,
             see scripts/collect_queries.py). The realistic distribution: often
             over-specified, sometimes not a title at all.

Reported per set and overall:
  * hit/miss agreement between live and `title_exact` -- the acceptance bar is
    >= 90%;
  * same-page agreement on the queries both resolve;
  * categorised disagreements.

"Hit" is defined identically on both sides, including upstream's
disambiguation re-search: a query that lands on a "may refer to:" page is
re-searched as "[query]", which is a miss. Without that, live and local would
be scored under different definitions.

Live responses go through `environment.wiki_get` with WIKI_CACHE=1 and a cache
directory on persistent storage, so the measurement is replayable byte for byte
after this pod is recycled.
"""

import argparse
import json
import os
import random
import sys
import time
from collections import Counter, OrderedDict

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
HOTPOTQA = os.path.join(REPO, "hotpotqa")
sys.path.insert(0, HOTPOTQA)
os.chdir(HOTPOTQA)

from bs4 import BeautifulSoup                        # noqa: E402

from src import durability                            # noqa: E402
from src import environment                           # noqa: E402
from src.environment import clean_str, wiki_get       # noqa: E402
from src.local_wiki import DISAMBIG_MARKER, LocalWiki  # noqa: E402
from src.mw_title import NOT_MAIN_NS, TitleError, mw_normalize_title  # noqa: E402

GOLD_PARQUET_URL = ("https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/"
                    "refs%2Fconvert%2Fparquet/distractor/validation/0000.parquet")

FILLER_WORDS = ["history", "biography", "overview", "wikipedia", "details",
                "summary", "career", "origins"]


# ---------------------------------------------------------------------------
# query sets
# ---------------------------------------------------------------------------

def gold_titles(aux_dir, n, rng):
    import pandas as pd
    path = os.path.join(aux_dir, "hotpot_dev_distractor.parquet")
    if not os.path.exists(path):
        import subprocess
        os.makedirs(aux_dir, exist_ok=True)
        print(f"downloading gold supporting facts -> {path}")
        subprocess.check_call(["curl", "-sSL", "-o", path + ".tmp", GOLD_PARQUET_URL])
        os.replace(path + ".tmp", path)
    df = pd.read_parquet(path, columns=["question", "supporting_facts"])
    titles = OrderedDict()
    for sf in df["supporting_facts"]:
        for t in sf["title"]:
            titles.setdefault(str(t), None)
    allt = list(titles)
    rng.shuffle(allt)
    return allt[:n]


def perturb(title, rng):
    """lowercase / drop one word / add one word -- cycled deterministically."""
    words = title.split()
    kind = rng.choice(["lower", "drop", "add"])
    if kind == "drop" and len(words) > 1:
        i = rng.randrange(len(words))
        return " ".join(words[:i] + words[i + 1:]), "drop"
    if kind == "add":
        return title + " " + rng.choice(FILLER_WORDS), "add"
    return title.lower(), "lower"


def agent_queries(n):
    path = os.path.join(HOTPOTQA, "cache", "wiki", "queries.jsonl")
    if not os.path.exists(path):
        raise SystemExit(f"{path} missing -- run scripts/collect_queries.py first")
    seen = OrderedDict()
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                seen.setdefault(json.loads(line)["query"], None)
    out = list(seen)[:n]
    if len(out) < n:
        raise SystemExit(f"only {len(out)} unique agent queries available, need {n}; "
                         "run scripts/collect_queries.py with more --n-questions")
    return out


# ---------------------------------------------------------------------------
# probes
# ---------------------------------------------------------------------------

HEADERS = {"User-Agent": "React/1.0"}


def live_probe(entity, sleep=0.25, _depth=0):
    """What upstream's `search_step` would have concluded, live, right now.

    Returns {hit, title, similar, disambig, recursed}.
    """
    url = ("https://en.wikipedia.org/w/index.php?search="
           + entity.replace(" ", "+"))
    text = wiki_get(url, HEADERS)
    if sleep:
        time.sleep(sleep)
    soup = BeautifulSoup(text, features="html.parser")
    result_divs = soup.find_all("div", {"class": "mw-search-result-heading"})
    if result_divs:
        return {"hit": False, "title": None,
                "similar": [clean_str(d.get_text().strip()) for d in result_divs][:5],
                "disambig": False, "recursed": _depth > 0, "zero_results": False}

    # Zero-result search: Special:Search renders no result headings at all, so
    # upstream's `if result_divs` test falls through to the article branch and
    # hands the agent the search page's boilerplate. That is a miss in every
    # sense that matters here, and is scored as one.
    body = soup.find("body")
    body_cls = " ".join(body.get("class", [])) if body else ""
    if "mw-special-Search" in body_cls:
        return {"hit": False, "title": None, "similar": [], "disambig": False,
                "recursed": _depth > 0, "zero_results": True}

    blocks = [p.get_text().strip()
              for p in soup.find_all("p") + soup.find_all("ul")]
    if any(DISAMBIG_MARKER in p for p in blocks) and _depth == 0:
        # environment.py re-searches the bracketed term; take its verdict.
        out = live_probe("[" + entity + "]", sleep=sleep, _depth=1)
        out["disambig"] = True
        return out

    h1 = soup.find(id="firstHeading")
    return {"hit": True,
            "title": clean_str(h1.get_text().strip()) if h1 else None,
            "similar": [], "disambig": False, "recursed": _depth > 0,
            "zero_results": False}


def local_probe(backend, entity):
    pid, title, why = backend.resolve(entity)
    if pid is None:
        return {"hit": False, "title": None, "why": why, "disambig": False,
                "variant": None}
    page = backend.store.page_text_by_pid(pid)
    if DISAMBIG_MARKER in page:
        # Same recursion as upstream: "[entity]" has illegal title characters,
        # so the frozen corpus always answers it with a miss.
        return {"hit": False, "title": title, "why": "disambiguation",
                "disambig": True, "variant": why}
    return {"hit": True, "title": title, "why": "hit", "disambig": False,
            "variant": why}


def norm_or_none(title):
    if not title:
        return None
    try:
        t = mw_normalize_title(title)
    except TitleError:
        return None
    return None if t is NOT_MAIN_NS else t


# ---------------------------------------------------------------------------
# classification
# ---------------------------------------------------------------------------

def classify(backend, query, live, local):
    """Category for one query. Agreement rows get 'agree_*'."""
    live_title_norm = norm_or_none(live["title"])
    in_corpus = (live_title_norm is not None
                 and backend.store.resolve(live_title_norm) is not None)

    if live["hit"] and local["hit"]:
        if norm_or_none(local["title"]) == live_title_norm:
            return "agree_hit_same_page"
        return "agree_hit_different_page"
    if not live["hit"] and not local["hit"]:
        return "agree_miss"

    if live["hit"] and not local["hit"]:
        q_norm = norm_or_none(query)
        if "&" in query:
            # `search_step` builds the URL with `entity.replace(" ", "+")` and no
            # percent-encoding, so an '&' in the entity terminates the `search=`
            # parameter: live searched a PREFIX of the query, not the query.
            # Upstream's own defect, not a corpus gap; scored separately so it
            # cannot be mistaken for one.
            return "disagree_live_url_unescaped"
        if local["disambig"]:
            # live resolves the ambiguous term to an article; the 2019-08-01
            # snapshot still had a "may refer to:" page at that title.
            return "disagree_disambiguation_resolved_since_2019"
        if in_corpus and live_title_norm != q_norm:
            return "disagree_missing_redirect"
        if in_corpus and live_title_norm == q_norm:
            return "disagree_casing"
        if live_title_norm is None:
            return "disagree_other"
        return "disagree_page_created_after_2019"

    # local hit, live miss
    if live.get("disambig"):
        return "disagree_became_disambiguation_since_2019"
    if live.get("zero_results"):
        return "disagree_live_zero_results"
    return "disagree_page_gone_after_2019"


CATEGORY_HELP = {
    "agree_hit_same_page": "both resolved, same page",
    "agree_hit_different_page": "both resolved, different page",
    "agree_miss": "both reported a miss",
    "disagree_missing_redirect": "live followed a redirect/alias the snapshot has "
                                 "no redirect table for; the target page IS in the corpus",
    "disagree_casing": "live resolved the query title verbatim and the corpus has "
                       "it, so the case ladder should have found it (a defect)",
    "disagree_page_created_after_2019": "live target absent from the 2019-08-01 corpus",
    "disagree_page_gone_after_2019": "corpus has a page live search no longer resolves "
                                     "(deleted, merged or renamed since 2019)",
    "disagree_disambiguation_resolved_since_2019": "the 2019 title was a "
                                                   '"may refer to:" page; live is now an article',
    "disagree_became_disambiguation_since_2019": "the title is an article in the "
                                                 "snapshot and a disambiguation page live",
    "disagree_live_zero_results": "live search returned nothing at all; the "
                                  "snapshot has the page",
    "disagree_live_url_unescaped": "upstream builds the search URL without "
                                   "percent-encoding, so an '&' in the query truncated "
                                   "the live search term (upstream defect, not a corpus gap)",
    "disagree_other": "none of the above",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-gold", type=int, default=100)
    ap.add_argument("--n-perturbed", type=int, default=100)
    ap.add_argument("--n-agent", type=int, default=100)
    ap.add_argument("--seed", type=int, default=248)
    ap.add_argument("--sleep", type=float, default=0.25)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    # The pinned live cache under aux/ is what makes this a replayable
    # measurement rather than a fresh scrape of today's Wikipedia.
    durability.require_durable_outputs(
        report=args.out or os.path.join(REPO, "docs",
                                        "fidelity_kilt_20190801.json"),
        live_cache=os.path.join(os.environ["WIKI_DATA_ROOT"], "aux"))

    data_dir = os.environ["WIKI_DATA_DIR"]
    aux_dir = os.path.join(os.environ["WIKI_DATA_ROOT"], "aux")

    # Pin the live side so the measurement replays byte for byte.
    environment.WIKI_CACHE_ENABLED = True
    environment.WIKI_CACHE_DIR = os.path.join(aux_dir, "fidelity_live_cache")
    environment.reset_wiki_cache_stats()
    os.makedirs(environment.WIKI_CACHE_DIR, exist_ok=True)

    backend = LocalWiki(data_dir=data_dir, mode="title_exact")

    rng = random.Random(args.seed)
    gold = gold_titles(aux_dir, args.n_gold, rng)
    prng = random.Random(args.seed + 1)
    perturbed = []
    for t in gold[:args.n_perturbed]:
        q, kind = perturb(t, prng)
        perturbed.append((q, t, kind))
    agent = agent_queries(args.n_agent)

    sets = [("gold", [(q, None) for q in gold]),
            ("perturbed", [(q, src) for q, src, _k in perturbed]),
            ("agent", [(q, None) for q in agent])]

    rows = []
    for set_name, queries in sets:
        print(f"=== {set_name}: {len(queries)} queries ===", flush=True)
        for i, (q, src) in enumerate(queries, 1):
            live = live_probe(q, sleep=args.sleep)
            local = local_probe(backend, q)
            cat = classify(backend, q, live, local)
            rows.append({"set": set_name, "query": q, "source_title": src,
                         "live_hit": live["hit"], "live_title": live["title"],
                         "live_disambig": live["disambig"],
                         "live_zero_results": live.get("zero_results", False),
                         "local_hit": local["hit"], "local_title": local["title"],
                         "local_why": local["why"], "local_variant": local["variant"],
                         "category": cat})
            if i % 25 == 0:
                print(f"  {i}/{len(queries)}", flush=True)

    # ---- summary ----------------------------------------------------------
    summary = {}
    for set_name in ["gold", "perturbed", "agent"]:
        sub = [r for r in rows if r["set"] == set_name]
        agree = [r for r in sub if r["live_hit"] == r["local_hit"]]
        both = [r for r in sub if r["live_hit"] and r["local_hit"]]
        same = [r for r in both if r["category"] == "agree_hit_same_page"]
        summary[set_name] = {
            "n": len(sub),
            "hit_miss_agreement": len(agree) / len(sub) if sub else 0.0,
            "live_hits": sum(1 for r in sub if r["live_hit"]),
            "local_hits": sum(1 for r in sub if r["local_hit"]),
            "both_hit": len(both),
            "same_page_agreement_on_both_hit": len(same) / len(both) if both else None,
            "categories": dict(Counter(r["category"] for r in sub)),
        }
    agree_all = [r for r in rows if r["live_hit"] == r["local_hit"]]
    both_all = [r for r in rows if r["live_hit"] and r["local_hit"]]
    same_all = [r for r in both_all if r["category"] == "agree_hit_same_page"]
    summary["overall"] = {
        "n": len(rows),
        "hit_miss_agreement": len(agree_all) / len(rows),
        "both_hit": len(both_all),
        "same_page_agreement_on_both_hit":
            len(same_all) / len(both_all) if both_all else None,
        "categories": dict(Counter(r["category"] for r in rows)),
        "live_cache": dict(environment.WIKI_CACHE_STATS),
        "live_cache_dir": environment.WIKI_CACHE_DIR,
    }
    bar = 0.90
    summary["bar"] = bar
    summary["verdict"] = ("PASS" if summary["overall"]["hit_miss_agreement"] >= bar
                          else "FAIL")

    out = args.out or os.path.join(REPO, "docs", "fidelity_kilt_20190801.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"summary": summary, "rows": rows,
                   "perturbation_kinds": {q: k for q, _s, k in perturbed},
                   "args": vars(args),
                   "category_help": CATEGORY_HELP}, fh, indent=1, ensure_ascii=False)
        fh.write("\n")

    print()
    print(f"{'set':<11}{'n':>5}{'hit/miss agree':>16}{'both hit':>10}{'same page':>11}")
    for k in ["gold", "perturbed", "agent", "overall"]:
        s = summary[k]
        sp = s["same_page_agreement_on_both_hit"]
        print(f"{k:<11}{s['n']:>5}{s['hit_miss_agreement']*100:>15.1f}%"
              f"{s['both_hit']:>10}"
              + (f"{sp*100:>10.1f}%" if sp is not None else f"{'-':>11}"))
    print()
    for cat, n in sorted(summary["overall"]["categories"].items(),
                         key=lambda kv: -kv[1]):
        print(f"  {n:>4}  {cat}")
    print(f"\nverdict: {summary['verdict']} (bar {bar:.0%}) -> {out}")
    return 0 if summary["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())

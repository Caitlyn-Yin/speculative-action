"""Judge and embedding backends for the criterion battery (``src/gates.py``).

Three things live here, all kept out of ``gates.py`` so that the criteria stay
importable with nothing but the standard library:

* :class:`AppendOnlyCache` -- the append-only JSONL cache every LLM and
  embedding call goes through, per ``docs/JUDGE_CONTRACT.md``;
* :class:`VLLMJudge` -- Qwen3-8B (or any other local model) behind the
  OpenAI-compatible vLLM server, exposing ``complete`` and ``yes_no_margin``;
* :class:`LocalEmbedder` -- one fixed local sentence embedding model.

Heavy dependencies (``openai``, ``torch``, ``transformers``) are imported
lazily, so importing this module on a node with neither installed is fine and
only *using* the backend fails.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Append-only cache
# ---------------------------------------------------------------------------

CACHE_FORMAT_VERSION = 1


def cache_key(payload: Dict[str, Any]) -> str:
    """sha256 over the canonical JSON of everything that changes the output."""
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class AppendOnlyCache:
    """Append-only JSONL cache keyed by a hash of (model, prompt, parameters).

    Contract (``docs/JUDGE_CONTRACT.md``):

    * records are only ever appended; nothing is rewritten or deleted, so the
      file is an audit log as well as a cache;
    * a record carries the full prompt, so a cached score can be re-derived
      without the code that produced it;
    * on a duplicate key the **first** record wins and the later one is kept in
      the file and counted in ``self.collisions``; a differing response for an
      identical key means non-determinism somewhere and must be reported, not
      silently overwritten.
    """

    def __init__(self, path: str, read_only: bool = False):
        self.path = path
        self.read_only = read_only
        self.entries: Dict[str, Dict[str, Any]] = {}
        self.collisions: List[str] = []
        self.hits = 0
        self.misses = 0
        self._fh = None
        if path and os.path.exists(path):
            self._load()

    def _load(self) -> None:
        with open(self.path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue          # a truncated tail from a killed run
                key = rec.get("key")
                if key is None:
                    continue
                if key in self.entries:
                    if self.entries[key].get("response") != rec.get("response"):
                        self.collisions.append(key)
                    continue          # first record wins
                self.entries[key] = rec

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        rec = self.entries.get(key)
        if rec is None:
            self.misses += 1
        else:
            self.hits += 1
        return rec

    def put(self, key: str, record: Dict[str, Any]) -> None:
        record = dict(record, key=key, cache_format_version=CACHE_FORMAT_VERSION,
                     recorded_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        if key not in self.entries:
            self.entries[key] = record
        if self.read_only or not self.path:
            return
        d = os.path.dirname(self.path)
        if d:
            os.makedirs(d, exist_ok=True)
        if self._fh is None:
            self._fh = open(self.path, "a", encoding="utf-8")
        self._fh.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        self._fh.flush()
        os.fsync(self._fh.fileno())

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def stats(self) -> Dict[str, Any]:
        return {"path": self.path, "records": len(self.entries),
                "hits": self.hits, "misses": self.misses,
                "collisions": len(self.collisions)}


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------

def verdict_token(tok: str) -> Optional[str]:
    """``"Yes"``, ``" yes"``, ``"▁YES"`` -> ``"yes"``; anything else -> ``None``.

    Ours: which surface variants count as a verdict token.  The leading-space
    and sentencepiece-underline forms are folded in because a tokenizer's choice
    of word-boundary marker is not evidence about the judge's verdict.
    """
    t = tok.replace("▁", " ").strip().lower()
    return t if t in ("yes", "no") else None


def _logsumexp(xs: Sequence[float]) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    m = max(xs)
    return m + math.log(sum(math.exp(x - m) for x in xs))


def yes_no_margin_from_top(top: Dict[str, float]) -> Tuple[Optional[float], Dict[str, Any]]:
    """``log p(Yes) - log p(No)`` from one position's ``top_logprobs`` dict.

    A side missing from the top-k is bounded above by the smallest returned
    log-probability -- the true value cannot exceed it -- and ``clipped`` records
    which side was bounded, so a margin resting on a bound is never mistaken for
    a measured one.
    """
    if not top:
        return None, {"clipped": None, "reason": "no top_logprobs"}
    # every surface variant that normalizes to "yes"/"no" contributes its own
    # mass, so the aggregation must not collapse them into a dict first
    yes = _logsumexp([v for k, v in top.items() if verdict_token(k) == "yes"])
    no = _logsumexp([v for k, v in top.items() if verdict_token(k) == "no"])
    floor = min(top.values())
    clipped = []
    if yes is None:
        yes = floor
        clipped.append("yes")
    if no is None:
        no = floor
        clipped.append("no")
    if len(clipped) == 2:
        return None, {"clipped": clipped, "reason": "neither Yes nor No in top-k",
                      "top": dict(sorted(top.items(), key=lambda kv: -kv[1])[:5])}
    return yes - no, {"logp_yes": yes, "logp_no": no,
                      "clipped": clipped or None, "floor": floor}


class VLLMJudge:
    """A local judge model behind the OpenAI-compatible vLLM server.

    ``model_id`` is the served model name (e.g. ``Qwen/Qwen3-8B``).  Every call
    goes through ``AppendOnlyCache``; ``revision`` is whatever the caller
    records about the served weights (vLLM does not report a commit sha, so this
    is the operator's note, not a verified hash).
    """

    def __init__(self, model_id: str, base_url: str, cache: AppendOnlyCache,
                 temperature: float = 0.0, top_p: float = 1.0,
                 seed: Optional[int] = 0, top_logprobs: int = 20,
                 revision: Optional[str] = None, max_tokens: int = 512):
        self.model_id = model_id
        self.base_url = base_url
        self.cache = cache
        self.temperature = temperature
        self.top_p = top_p
        self.seed = seed
        self.top_logprobs = top_logprobs
        self.revision = revision
        self.default_max_tokens = max_tokens
        self._client = None

    # -- plumbing ----------------------------------------------------------

    def _llm(self):
        if self._client is None:
            from .llm_client import LLMClient
            self._client = LLMClient(model_name=self.model_id,
                                     temperature=self.temperature,
                                     max_tokens=self.default_max_tokens,
                                     top_p=self.top_p,
                                     backend="local",
                                     base_url=self.base_url)
        return self._client

    def _params(self, **kw) -> Dict[str, Any]:
        return {"temperature": self.temperature, "top_p": self.top_p,
                "seed": self.seed, **kw}

    def _cached(self, kind: str, prompt: str, params: Dict[str, Any], fn):
        key = cache_key({"model_id": self.model_id, "kind": kind,
                         "prompt": prompt, "params": params})
        rec = self.cache.get(key)
        if rec is not None:
            return rec["response"]
        response = fn()
        self.cache.put(key, {"model_id": self.model_id, "kind": kind,
                             "prompt": prompt, "params": params,
                             "response": response})
        return response

    # -- API used by gates.Context ----------------------------------------

    def complete(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        mt = max_tokens or self.default_max_tokens
        params = self._params(max_tokens=mt)

        def call():
            return {"text": self._llm()._local_call(prompt, None)}

        return self._cached("complete", prompt, params, call)["text"]

    def yes_no_margin(self, prompt: str) -> Tuple[Optional[float], Dict[str, Any]]:
        params = self._params(max_tokens=1, top_logprobs=self.top_logprobs)

        def call():
            out = self._llm().call_with_logprobs(
                prompt, max_tokens=1, top_logprobs=self.top_logprobs)
            first = out["tokens"][0] if out.get("tokens") else {}
            return {"text": out.get("text", ""),
                    "first_token": first.get("token"),
                    "top": first.get("top", {})}

        resp = self._cached("yes_no_margin", prompt, params, call)
        margin, info = yes_no_margin_from_top(resp.get("top") or {})
        info["first_token"] = resp.get("first_token")
        return margin, info


# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------

#: The one fixed embedding model.  bge-* uses CLS pooling per its model card;
#: anything else falls back to mean pooling.  Both are L2-normalized, which is
#: what makes the dot product a cosine and the published 0.80 / 0.95 thresholds
#: comparable at all.
DEFAULT_EMBED_MODEL = "BAAI/bge-base-en-v1.5"


class LocalEmbedder:
    """Sentence embeddings from one fixed local model, with a cache.

    ``revision`` is resolved to the local snapshot directory's commit hash when
    the model comes from an HF cache, so the manifest pins the exact weights.
    """

    def __init__(self, model_id: str = DEFAULT_EMBED_MODEL,
                 cache: Optional[AppendOnlyCache] = None,
                 max_length: int = 512, device: str = "cpu",
                 local_files_only: bool = True):
        self.model_id = model_id
        self.cache = cache
        self.max_length = max_length
        self.device = device
        self.local_files_only = local_files_only
        self.revision: Optional[str] = None
        self._tok = None
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        import torch  # noqa: F401
        from transformers import AutoModel, AutoTokenizer
        kw = {"local_files_only": self.local_files_only}
        self._tok = AutoTokenizer.from_pretrained(self.model_id, **kw)
        self._model = AutoModel.from_pretrained(self.model_id, **kw)
        self._model.eval()
        self._model.to(self.device)
        self.revision = self._resolve_revision()

    @staticmethod
    def _sha_from_snapshot_path(path: str) -> Optional[str]:
        parts = (path or "").rstrip("/").split("/")
        if "snapshots" in parts:
            i = parts.index("snapshots")
            if i + 1 < len(parts):
                return parts[i + 1]
        return None

    def _resolve_revision(self) -> Optional[str]:
        """The HF cache snapshot directory name, i.e. the weights' commit sha.

        Tried in two ways -- the config's ``_name_or_path`` (set when the model
        was loaded from a snapshot directory) and the hub cache lookup for
        ``config.json``.  ``None`` means the weights could not be pinned; the
        manifest then says so rather than implying a revision.
        """
        sha = self._sha_from_snapshot_path(
            getattr(self._model.config, "_name_or_path", "") or "")
        if sha:
            return sha
        try:
            from huggingface_hub import try_to_load_from_cache
            hit = try_to_load_from_cache(self.model_id, "config.json")
            if isinstance(hit, str):
                return self._sha_from_snapshot_path(hit)
        except Exception:
            pass
        return None

    @property
    def pooling(self) -> str:
        return "cls" if "bge" in self.model_id.lower() else "mean"

    def _embed_uncached(self, texts: Sequence[str]) -> List[List[float]]:
        self._load()
        import torch
        batch = self._tok(list(texts), padding=True, truncation=True,
                          max_length=self.max_length, return_tensors="pt")
        batch = {k: v.to(self.device) for k, v in batch.items()}
        with torch.no_grad():
            out = self._model(**batch).last_hidden_state
        if self.pooling == "cls":
            vecs = out[:, 0]
        else:
            mask = batch["attention_mask"].unsqueeze(-1).float()
            vecs = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        vecs = torch.nn.functional.normalize(vecs, p=2, dim=1)
        return [[float(x) for x in row] for row in vecs.cpu()]

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        if self.cache is None:
            return self._embed_uncached(texts)
        self._load()      # so `revision` is part of the key
        out: List[Optional[List[float]]] = [None] * len(texts)
        todo: List[int] = []
        keys: List[str] = []
        for i, t in enumerate(texts):
            key = cache_key({"model_id": self.model_id, "kind": "embed",
                             "revision": self.revision, "pooling": self.pooling,
                             "max_length": self.max_length, "text": t})
            keys.append(key)
            rec = self.cache.get(key)
            if rec is None:
                todo.append(i)
            else:
                out[i] = rec["response"]["vector"]
        if todo:
            fresh = self._embed_uncached([texts[i] for i in todo])
            for i, vec in zip(todo, fresh):
                out[i] = vec
                self.cache.put(keys[i], {
                    "model_id": self.model_id, "kind": "embed",
                    "revision": self.revision, "pooling": self.pooling,
                    "max_length": self.max_length,
                    "truncated": len(self._tok.tokenize(texts[i])) > self.max_length,
                    "prompt": texts[i],
                    "response": {"vector": vec}})
        return [v for v in out]  # type: ignore[misc]

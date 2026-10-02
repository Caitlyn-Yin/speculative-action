"""Acceptance criteria for speculative actions -- the audited criterion battery.

This module implements the 13 published/derived acceptance criteria that Paper B
audits against the replay labels defined in ``docs/PREREG_PAPER_B.md`` (S1, S2,
S3, harmful, delayed).  Every criterion is a registered :class:`Criterion` and
records, in machine-readable form:

* ``name``        -- stable key used in ``criteria_scores.jsonl``;
* ``source``      -- arXiv id + section, or a repo file:line, or "ours";
* ``level``       -- ``"call"`` or ``"obs"``.  ``"obs"`` criteria need the
  *executed* observation of ``spec_j``, so they cannot be evaluated before the
  speculated call has actually run: they can never hide the real call's latency.
  This is enforced in code (see :func:`_require_executed_obs`), not just
  documented, because a speculator-imagined observation
  (``environment.guess_step``) would silently turn an obs-level criterion into a
  free call-level one;
* ``output``      -- ``"binary"`` (a single accept/reject) or ``"score"`` (a real
  value plus a threshold grid, so the audit can sweep operating points).

Each criterion's docstring has two mandatory headings -- ``Parameters verified
from the source:`` and ``Parameters chosen by us:`` -- and ``test_gates.py``
asserts both are present.  Nothing in this module may pretend that a number we
picked came from a paper.

Provenance warning
------------------
Criteria 2, 3, 6 and 7 are *ours*, and their pre-loss definitions
(``NormalizedMatchGate``, the deterministic battery, ``se_judge_v2`` rules a-d)
were lost with the GH200 host -- see ``docs/LOST_WORK_MANIFEST.md`` items 1, 5,
6.  What is implemented here is a **re-derivation**, made on instruction, not a
recovery.  Per ``CLAUDE.md`` §3 this means numbers scored under these four
criteria are **not comparable** to any pre-loss number; the re-derived text is
frozen in ``docs/CRITERIA.md`` so that at least this version is auditable.

The module is pure standard library.  Embedding and judge backends are injected
through :class:`Context` (see ``src/backends.py``); the criteria that need them
return an ``na_reason`` when they are absent, so a stdlib-only run still scores
the eight deterministic criteria.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Threshold grids.  Every value that appears in a grid is either taken from a
# source (annotated) or chosen by us (annotated).  The grids are part of the
# recorded contract: the audit sweeps them, it does not invent new ones.
# ---------------------------------------------------------------------------

#: normalized Levenshtein *distance* grid for `edit_distance`; ours (DSP
#: publishes no distance and no threshold -- see the criterion docstring).
EDIT_DISTANCE_GRID: Tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5)

#: cosine grid for `embed_call` / `embed_obs`.  0.80 = SpecBox arXiv:2607.23933
#: §3.3/§4 (tau_c = 0.8); 0.95 = Cost-Aware arXiv:2606.07846 §7.4 (Tier 2);
#: 0.85 and 0.90 are ours, interpolated to give the sweep two interior points.
EMBED_GRID: Tuple[float, ...] = (0.80, 0.85, 0.90, 0.95)

#: verbalized-confidence grid; ours.
VERBAL_CONF_GRID: Tuple[float, ...] = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95)

#: log-odds margin grid for the two log-prob judges; ours.  DualSpec
#: (arXiv:2603.07416 §5/Eq. 7) uses a single tau tuned offline to a ~20%
#: intervention rate and never publishes its value, so a grid is the only
#: faithful option.
LOGPROB_MARGIN_GRID: Tuple[float, ...] = (-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0)

#: min-token-probability grid for `spec_confidence`.  0.90 = SPORK
#: arXiv:2607.03333 §4.2 (theta = 0.90); the other four are ours.
SPEC_CONF_GRID: Tuple[float, ...] = (0.50, 0.70, 0.80, 0.90, 0.95)

#: SpecHop arXiv:2605.21965 App. D.4 constants, verbatim from the appendix.
SPECHOP_TOKEN_COVERAGE = 0.72      # ">= 72% token coverage"
SPECHOP_JACCARD = 0.55             # "a Jaccard similarity of >= 0.55"
SPECHOP_SHORT_TARGET_CHARS = 5     # "For exceptionally short answers (under 5 characters)"

#: SpecHop gives refusal/uncertainty patterns only as "e.g., 'I don't know,'
#: 'information unavailable'".  The list below is ours; the two examples the
#: appendix names are the first two entries.  Patterns are written in their
#: surface form and matched after ``norm_text`` on both sides, so "I don't know"
#: matches "i don t know" -- punctuation removal splits the contraction.
REFUSAL_PATTERNS: Tuple[str, ...] = (
    "I don't know",
    "information unavailable",
    "I do not know",
    "not sure",
    "unsure",
    "cannot determine",
    "cannot be determined",
    "unable to determine",
    "unable to answer",
    "no information",
    "not available",
    "unknown",
    "could not find",
    "no relevant information",
    "as an AI",
)

#: "standard English stopwords" (SpecHop App. D.4) -- the appendix does not
#: publish its list, so this one is ours: the NLTK English stopword list is not
#: importable offline, and a frozen literal keeps the criterion deterministic.
STOPWORDS: frozenset = frozenset("""
a about above after again against all am an and any are aren as at be because
been before being below between both but by can cannot could couldn did didn do
does doesn doing don down during each few for from further had hadn has hasn
have haven having he her here hers herself him himself his how i if in into is
isn it its itself just me more most mustn my myself no nor not now of off on
once only or other ought our ours ourselves out over own re s same shan she
should shouldn so some such t than that the their theirs them themselves then
there these they this those through to too under until up ve very was wasn we
were weren what when where which while who whom why will with won would wouldn
you your yours yourself yourselves
""".split())

#: Alternative refusal/uncertainty pattern list for criterion 11, registered as
#: the ``refusal_minimal`` variant.  Where ``REFUSAL_PATTERNS`` above is ours
#: (16 entries, of which the appendix names 2), this is the *only* list strictly
#: recoverable from SpecHop App. D.4: its two published examples and nothing
#: else.  It is the informative alternative precisely because it is a subset --
#: comparing the two bounds how much of criterion 11's rejection behaviour comes
#: from our 14 additions rather than from the source.
REFUSAL_PATTERNS_MINIMAL: Tuple[str, ...] = (
    "I don't know",
    "information unavailable",
)

#: NLTK's English stopword list, verbatim from `nltk.corpus.stopwords
#: .words("english")` under nltk 3.10.3 (198 entries, sorted;
#: sha256 of the space-joined sorted list:
#: 97f4fd27ecb1ef242e68e83c16b1f7a903d78a42eb719f6c1e7f40d313e97443).
#: Frozen as a literal so the criterion stays deterministic and offline --
#: nltk is not a dependency of the pipeline env and its corpus download is
#: a network call we must not make mid-audit.
STOPWORDS_NLTK_RAW: Tuple[str, ...] = tuple("""
    a about above after again against ain all am an and any are aren
    aren't as at be because been before being below between both but by
    can couldn couldn't d did didn didn't do does doesn doesn't doing don
    don't down during each few for from further had hadn hadn't has hasn
    hasn't have haven haven't having he he'd he'll he's her here hers
    herself him himself his how i i'd i'll i'm i've if in into is isn
    isn't it it'd it'll it's its itself just ll m ma me mightn mightn't
    more most mustn mustn't my myself needn needn't no nor not now o of
    off on once only or other our ours ourselves out over own re s same
    shan shan't she she'd she'll she's should should've shouldn shouldn't
    so some such t than that that'll the their theirs them themselves then
    there these they they'd they'll they're they've this those through to
    too under until up ve very was wasn wasn't we we'd we'll we're we've
    were weren weren't what when where which while who whom why will with
    won won't wouldn wouldn't y you you'd you'll you're you've your yours
    yourself yourselves
""".split())

#: scikit-learn's ENGLISH_STOP_WORDS, verbatim from
#: `sklearn.feature_extraction.text.ENGLISH_STOP_WORDS` under sklearn 1.7.2
#: (318 entries, sorted; sha256 of the space-joined sorted list:
#: e570e9b41eab43e963c44d1d8b7ad441d084fa84f1104e01c9e8b41ad43feb89).
#: This is the Glasgow IR list; sklearn's own docs call it a known-poor
#: general-purpose list, which is exactly why it is useful as a bound.
STOPWORDS_SKLEARN_RAW: Tuple[str, ...] = tuple("""
    a about above across after afterwards again against all almost alone
    along already also although always am among amongst amoungst amount an
    and another any anyhow anyone anything anyway anywhere are around as
    at back be became because become becomes becoming been before
    beforehand behind being below beside besides between beyond bill both
    bottom but by call can cannot cant co con could couldnt cry de
    describe detail do done down due during each eg eight either eleven
    else elsewhere empty enough etc even ever every everyone everything
    everywhere except few fifteen fifty fill find fire first five for
    former formerly forty found four from front full further get give go
    had has hasnt have he hence her here hereafter hereby herein hereupon
    hers herself him himself his how however hundred i ie if in inc indeed
    interest into is it its itself keep last latter latterly least less
    ltd made many may me meanwhile might mill mine more moreover most
    mostly move much must my myself name namely neither never nevertheless
    next nine no nobody none noone nor not nothing now nowhere of off
    often on once one only onto or other others otherwise our ours
    ourselves out over own part per perhaps please put rather re same see
    seem seemed seeming seems serious several she should show side since
    sincere six sixty so some somehow someone something sometime sometimes
    somewhere still such system take ten than that the their them
    themselves then thence there thereafter thereby therefore therein
    thereupon these they thick thin third this those though three through
    throughout thru thus to together too top toward towards twelve twenty
    two un under until up upon us very via was we well were what whatever
    when whence whenever where whereafter whereas whereby wherein
    whereupon wherever whether which while whither who whoever whole whom
    whose why will with within without would yet you your yours yourself
    yourselves
""".split())


#: Articles stripped by our normalizer (criterion 2).  Ours.
LEADING_ARTICLES: Tuple[str, ...] = ("the ", "a ", "an ")


# ---------------------------------------------------------------------------
# Pair schema
# ---------------------------------------------------------------------------

@dataclass
class Turn:
    """One realized step of the recorded trajectory."""
    thought: str = ""
    action: str = ""
    obs: str = ""


@dataclass
class Pair:
    """A speculation pair, the unit of Paper B (``docs/PREREG_PAPER_B.md``).

    ``p = (run, question, step i, spec j)`` with ``lower(spec_j) != lower(real_i)``.

    Attributes
    ----------
    spec_obs, spec_obs_source
        ``spec_obs`` must be the observation the environment **actually**
        returned for ``spec_j`` during the treatment re-execution.
        ``spec_obs_source`` must then be ``"executed"``.  The speculator's
        imagined page (``WikiEnv.guess_step``, ``src/environment.py:86``) is
        recorded as ``"speculator_imagined"`` and is *refused* by every
        obs-level criterion: the prereg's treatment arm says "Never use the
        speculator's hallucinated ``sim_obs``", and an obs-level criterion fed
        an imagined observation would no longer be paying the real call's
        latency.  ``"unavailable"`` is the third legal value.
    spec_tokens
        Per-token records for the speculator's generation of ``spec_j``, as
        returned by ``LLMClient.call_with_logprobs`` -- ``[{"token": str,
        "logprob": float, "top": {tok: logprob}}, ...]``.  Needed by
        ``spec_confidence`` only.
    labels
        Replay labels from the re-execution audit (``S1``, ``S2``, ``S3``,
        ``harmful``, ``delayed``, deltas).  Never read by any criterion -- they
        are the ground truth the criteria are scored against, and are copied
        into the output row only so the audit can join on one file.
    """

    pair_id: str
    question: str
    real_action: str
    spec_action: str
    step_i: int = 0
    spec_index: int = 0
    run: str = ""
    history: List[Turn] = field(default_factory=list)
    real_obs: Optional[str] = None
    spec_obs: Optional[str] = None
    spec_obs_source: str = "unavailable"
    spec_tokens: Optional[List[Dict[str, Any]]] = None
    labels: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    # -- serialization ------------------------------------------------------

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Pair":
        d = dict(d)
        hist = [Turn(**t) if isinstance(t, dict) else Turn(*t)
                for t in d.pop("history", [])]
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        extra = {k: v for k, v in d.items() if k not in known}
        d = {k: v for k, v in d.items() if k in known}
        pair = cls(history=hist, **d)
        if extra:
            pair.meta.update(extra)
        return pair

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

@dataclass
class CriterionResult:
    name: str
    level: str
    output: str
    binary: Optional[bool] = None
    score: Optional[float] = None
    decisions: Dict[str, bool] = field(default_factory=dict)
    flags: Dict[str, bool] = field(default_factory=dict)
    na_reason: Optional[str] = None
    detail: Dict[str, Any] = field(default_factory=dict)
    #: Named alternative parameterizations of the SAME criterion, scored on the
    #: same pair in the same pass.  Each value is
    #: ``{"binary"|"score", "decisions", "detail"}``.  Variants exist so a
    #: sensitivity analysis is possible without re-querying any judge, and so
    #: that the set of alternatives is fixed in the registry BEFORE the audit
    #: rather than chosen after seeing which one looks better.
    variants: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v not in (None, {}, [])} \
            | {"name": self.name, "level": self.level, "output": self.output}


@dataclass
class Criterion:
    name: str
    level: str                      # "call" | "obs"
    output: str                     # "binary" | "score"
    source: str                     # arXiv id + section / repo file:line / "ours"
    fn: Callable[["Pair", "Context"], CriterionResult]
    thresholds: Tuple[float, ...] = ()
    requires: Tuple[str, ...] = ()  # "embedder" | "judge" | "spec_tokens" | "spec_obs" | "real_obs"
    faithful: bool = True           # False = we knowingly deviate from the source
    notes: str = ""
    #: Names of the alternative parameterizations this criterion reports under
    #: ``CriterionResult.variants``.  Declared here so ``registry_table()`` --
    #: and therefore ``criteria_manifest.json`` and docs/CRITERIA.md -- is the
    #: pre-audit record of which variants exist.
    variants: Tuple[str, ...] = ()

    @property
    def doc(self) -> str:
        return self.fn.__doc__ or ""

    def describe(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "level": self.level,
            "output": self.output,
            "source": self.source,
            "thresholds": list(self.thresholds),
            "requires": list(self.requires),
            "faithful_to_source": self.faithful,
            "notes": self.notes,
            "variants": list(self.variants),
            "docstring": self.doc,
        }


CRITERIA: Dict[str, Criterion] = {}
_ORDER: List[str] = []


def register(name: str, *, level: str, output: str, source: str,
             thresholds: Sequence[float] = (), requires: Sequence[str] = (),
             faithful: bool = True, notes: str = "",
             variants: Sequence[str] = ()):
    """Register a criterion.  Enforces the docstring contract at import time."""

    def deco(fn):
        doc = fn.__doc__ or ""
        for heading in ("Parameters verified from the source:",
                        "Parameters chosen by us:"):
            if heading not in doc:
                raise AssertionError(
                    f"criterion {name!r} docstring is missing {heading!r}")
        if level not in ("call", "obs"):
            raise AssertionError(f"criterion {name!r}: bad level {level!r}")
        if output not in ("binary", "score"):
            raise AssertionError(f"criterion {name!r}: bad output {output!r}")
        if output == "score" and not thresholds:
            raise AssertionError(f"criterion {name!r}: score output needs a grid")
        CRITERIA[name] = Criterion(
            name=name, level=level, output=output, source=source, fn=fn,
            thresholds=tuple(thresholds), requires=tuple(requires),
            faithful=faithful, notes=notes, variants=tuple(variants))
        _ORDER.append(name)
        return fn

    return deco


# ---------------------------------------------------------------------------
# Context (injected backends + recorded configuration)
# ---------------------------------------------------------------------------

@dataclass
class Context:
    """Backends and configuration for one scoring run.

    ``embedder`` must expose ``model_id``, ``revision`` and
    ``embed(list[str]) -> list[list[float]]``.  ``judge`` / ``judge_alt`` must
    expose ``model_id``, ``complete(prompt, max_tokens)`` and
    ``yes_no_margin(prompt)``; see ``src/backends.py``.  ``judge_alt`` is the
    non-Qwen same-size circularity check for criteria 6-8 (Llama-3.1-8B-Instruct
    is present on this node at ``/models``); when it is absent the scored row
    records ``judge_alt: null`` so the absence is visible in the data.
    """

    embedder: Any = None
    judge: Any = None
    judge_alt: Any = None
    timestamp_for_sufficient_context: str = "2019-08-01"
    code_version: str = ""

    def manifest(self) -> Dict[str, Any]:
        def ident(b):
            if b is None:
                return None
            return {"model_id": getattr(b, "model_id", "?"),
                    "revision": getattr(b, "revision", None)}
        return {
            "embedder": ident(self.embedder),
            "judge": ident(self.judge),
            "judge_alt": ident(self.judge_alt),
            "timestamp_for_sufficient_context": self.timestamp_for_sufficient_context,
            "code_version": self.code_version,
            "grids": {
                "edit_distance": list(EDIT_DISTANCE_GRID),
                "embed": list(EMBED_GRID),
                "verbal_confidence": list(VERBAL_CONF_GRID),
                "logprob_margin": list(LOGPROB_MARGIN_GRID),
                "spec_confidence": list(SPEC_CONF_GRID),
            },
        }


# ---------------------------------------------------------------------------
# Shared text helpers
# ---------------------------------------------------------------------------

_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_WS_RE = re.compile(r"\s+")


def strip_diacritics(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s)
                   if not unicodedata.combining(c))


def norm_text(s: str) -> str:
    """Lowercase, strip diacritics, drop punctuation, collapse whitespace.

    This is the normalization SpecHop App. D.4 specifies ("lowercasing,
    diacritic and punctuation removal"); the exact character classes are ours.
    """
    if s is None:
        return ""
    s = strip_diacritics(str(s)).lower()
    s = _PUNCT_RE.sub(" ", s)
    return _WS_RE.sub(" ", s).strip()


def parse_action(action: str) -> Tuple[str, str]:
    """Split a ReAct action ``tool[arg]`` into ``(tool_lower, arg)``.

    Mirrors upstream's ``Metrics.get_action_name`` (``src/metrics.py:88-95``),
    which takes everything before the first ``[`` as the tool name; the argument
    is the text up to the last ``]``.  A malformed action yields ``("", raw)``.
    """
    if action is None:
        return "", ""
    raw = str(action).strip()
    i = raw.find("[")
    if i < 0:
        return "", raw
    tool = raw[:i].strip().lower()
    arg = raw[i + 1:]
    if arg.endswith("]"):
        arg = arg[:-1]
    return tool, arg.strip()


def norm_arg(arg: str) -> str:
    """Our argument normalizer (criterion 2): ``norm_text`` + article stripping.

    Parenthetical qualifiers are *kept*: ``search[Mercury (planet)]`` and
    ``search[Mercury]`` retrieve different pages, so collapsing them would make
    the gate accept a different entity.
    """
    s = norm_text(arg)
    changed = True
    while changed:
        changed = False
        for art in LEADING_ARTICLES:
            if s.startswith(art):
                s = s[len(art):]
                changed = True
    return s.strip()


def tokens(s: str) -> List[str]:
    return norm_text(s).split()


def content_tokens(s: str, stopwords: frozenset = None) -> List[str]:
    return [t for t in tokens(s)
            if t not in (STOPWORDS if stopwords is None else stopwords)]


def normalize_stopword_list(words: Sequence[str]) -> frozenset:
    """Project a published stopword list onto *our* token space.

    Necessary, not cosmetic. ``content_tokens`` filters tokens produced by
    ``tokens(norm_text(...))``, which strips punctuation -- so NLTK's
    apostrophe entries (``"aren't"``, ``"he's"``, ``"should've"``: 56 of its
    198) could never match a token as published, and the list would silently
    behave as a 142-word list with 56 dead entries. Normalizing each entry the
    same way the text is normalized, and splitting it into tokens, is what makes
    the comparison between lists mean what it says.

    Note the consequence, which is recorded rather than hidden: this *adds*
    short fragments (``aren't`` -> ``aren``, ``t``), so the normalized NLTK set
    is not a subset of the published one. Our own ``STOPWORDS`` literal was
    already written in this normalized space (it contains ``aren``, ``don``,
    ``t``, ``ve``), which is why it needs no projection.
    """
    out: set = set()
    for w in words:
        out.update(tokens(norm_text(w)))
    return frozenset(out)


#: The two published alternatives, projected onto our token space.  Registered
#: as criterion 11's ``stopwords_nltk`` / ``stopwords_sklearn`` variants.
STOPWORDS_NLTK: frozenset = normalize_stopword_list(STOPWORDS_NLTK_RAW)
STOPWORDS_SKLEARN: frozenset = normalize_stopword_list(STOPWORDS_SKLEARN_RAW)

#: Named stopword lists for criterion 11.  ``ours`` is the primary (the one
#: SpecHop's appendix does not publish); the other two are the variants.
STOPWORD_LISTS: Dict[str, frozenset] = {
    "ours": STOPWORDS,
    "nltk": STOPWORDS_NLTK,
    "sklearn": STOPWORDS_SKLEARN,
}

#: Named refusal-pattern lists for criterion 11.
REFUSAL_LISTS: Dict[str, Tuple[str, ...]] = {
    "ours": REFUSAL_PATTERNS,
    "minimal": REFUSAL_PATTERNS_MINIMAL,
}


def levenshtein(a: str, b: str) -> int:
    """Classic Levenshtein edit distance (unit cost ins/del/sub)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                           prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def normalized_levenshtein(a: str, b: str) -> float:
    """Edit distance divided by the longer string's length; 0.0 for two empties."""
    m = max(len(a), len(b))
    return 0.0 if m == 0 else levenshtein(a, b) / m


def cosine(u: Sequence[float], v: Sequence[float]) -> float:
    num = sum(x * y for x, y in zip(u, v))
    du = math.sqrt(sum(x * x for x in u))
    dv = math.sqrt(sum(y * y for y in v))
    if du == 0.0 or dv == 0.0:
        return 0.0
    return num / (du * dv)


def has_multi_digit_numbers(s: str) -> List[str]:
    """Multi-digit numbers ("years or quantities", SpecHop App. D.4)."""
    return re.findall(r"\d{2,}", s or "")


def _grid_decisions(score: float, grid: Sequence[float],
                    accept_when: str) -> Dict[str, bool]:
    """Turn one score into an accept/reject at every threshold in the grid."""
    out = {}
    for t in grid:
        key = f"{t:g}"
        out[key] = (score >= t) if accept_when == "ge" else (score <= t)
    return out


def _na(crit_name: str, reason: str) -> CriterionResult:
    c = CRITERIA[crit_name]
    return CriterionResult(name=crit_name, level=c.level, output=c.output,
                           na_reason=reason)


def _require_executed_obs(pair: Pair) -> Optional[str]:
    """Guard for every obs-level criterion.

    Returns an ``na_reason`` unless both observations are present *and*
    ``spec_obs`` came from a real execution.  This is the code-level expression
    of the prereg's treatment rule and of ``CLAUDE.md`` §6 ruling 3: an
    obs-level criterion scored against ``guess_step``'s imagined page would be a
    call-level criterion wearing an obs-level label.
    """
    if pair.real_obs is None:
        return "no real observation recorded"
    if pair.spec_obs is None:
        return "no executed observation for spec_j"
    if pair.spec_obs_source != "executed":
        return (f"spec_obs_source={pair.spec_obs_source!r}: obs-level criteria "
                "refuse speculator-imagined observations")
    return None


# ===========================================================================
# 1. exact_sa
# ===========================================================================

def _upstream_compare_action(action1: str, action2: str) -> int:
    """Verbatim transcription of upstream ``Metrics.compare_action`` (non-sparse).

    ``src/metrics.py:80-86``::

        action1 = action1.lower()
        action2 = action2.lower()
        ...
        return 1 if action1 == action2 else 0

    Used only when ``src.metrics`` cannot be imported (it pulls in pandas via
    ``src/utils.py``).  ``test_gates.py`` asserts the two agree on every fixture
    pair whenever the real import succeeds.
    """
    return 1 if action1.lower() == action2.lower() else 0


def upstream_compare_action(action1: str, action2: str) -> Tuple[int, str]:
    """Call the real upstream comparator when importable; else the transcription."""
    try:
        from .metrics import Metrics  # noqa: WPS433 (heavy import: pandas)
    except Exception:  # pragma: no cover - depends on the interpreter's env
        return _upstream_compare_action(action1, action2), "transcribed"
    return Metrics.compare_action(action1, action2, sparse=False), "upstream"


@register("exact_sa", level="call", output="binary",
          source="Speculative Actions, arXiv:2510.04371 §3 (lossless match / "
                 "discard on mismatch); operationalized by the released "
                 "implementation at hotpotqa/src/metrics.py:80-86",
          notes="the rejection rule that defines the Paper B population")
def exact_sa(pair: Pair, ctx: Context) -> CriterionResult:
    """Upstream exact match: ``lower(spec_j) == lower(real_i)``.

    Speculative Actions keeps a speculative branch only when the speculated
    action matches the realized one, and discards it otherwise; in the released
    HotpotQA harness that test is ``Metrics.compare_action``, which lowercases
    both strings and compares them for equality.  By construction of the Paper B
    unit (``lower(spec_j) != lower(real_i)``) this criterion is ``False`` on
    every in-population pair -- it is scored anyway, as the audit's null gate and
    as a guard that the population filter was applied.

    Parameters verified from the source:
      * comparison is case-insensitive string equality over the whole action
        string, arguments included (``src/metrics.py:81-86``);
      * no normalization, no threshold, binary output (ibid.).

    Parameters chosen by us:
      * nothing.
    """
    score, which = upstream_compare_action(pair.spec_action, pair.real_action)
    return CriterionResult(name="exact_sa", level="call", output="binary",
                           binary=bool(score),
                           detail={"comparator": which})


# ===========================================================================
# 2. normalized (ours)
# ===========================================================================

@register("normalized", level="call", output="binary",
          source="ours (re-derived NormalizedMatchGate; the pre-loss definition "
                 "is LOST -- docs/LOST_WORK_MANIFEST.md item 1)",
          notes="re-derivation, NOT a recovery: not comparable to pre-loss numbers")
def normalized(pair: Pair, ctx: Context) -> CriterionResult:
    """Our ``NormalizedMatchGate``: same tool channel and same normalized argument.

    Accept iff the tool names are equal (after lowercasing) **and** the
    arguments are equal after ``norm_arg`` -- lowercase, diacritics stripped,
    punctuation removed, whitespace collapsed, leading articles removed.
    Parenthetical disambiguators are deliberately preserved (see
    :func:`norm_arg`).

    Parameters verified from the source:
      * none.  This gate is ours, and the pre-loss version was lost with the
        GH200 host; this docstring plus ``docs/CRITERIA.md`` is the whole
        specification.

    Parameters chosen by us:
      * the tool channel must match (a ``search`` cannot normalize into a
        ``lookup``);
      * the normalizer: NFKD diacritic stripping, lowercase, non-word
        characters to spaces, whitespace collapse, repeated stripping of the
        leading articles ``the/a/an``;
      * parentheticals kept;
      * binary output, no threshold.
    """
    st, sa = parse_action(pair.spec_action)
    rt, ra = parse_action(pair.real_action)
    same_tool = st == rt
    ns, nr = norm_arg(sa), norm_arg(ra)
    return CriterionResult(
        name="normalized", level="call", output="binary",
        binary=bool(same_tool and ns == nr),
        detail={"spec_tool": st, "real_tool": rt,
                "spec_arg_norm": ns, "real_arg_norm": nr,
                "same_tool": same_tool})


# ===========================================================================
# 3. battery (ours)
# ===========================================================================

@register("battery", level="call", output="binary",
          source="ours (re-derived deterministic battery; pre-loss fields "
                 "channel / stale_intent / spec_fixation / agent_loop are LOST "
                 "-- docs/LOST_WORK_MANIFEST.md item 1)",
          notes="re-derivation; flags are REJECTION reasons, so binary=True "
                "means 'no filter fired'")
def battery(pair: Pair, ctx: Context) -> CriterionResult:
    """Deterministic filters, recorded as rejection flags.

    Five flags, all computed from the call and the realized history only.  The
    criterion's binary output is ``not any(flag)`` -- i.e. ``True`` when nothing
    rejects the speculation -- and every individual flag is kept in
    ``flags`` so the audit can score them one at a time.

      * ``tool_channel``     -- spec and real use different tools;
      * ``terminal_channel`` -- exactly one of them is ``finish`` (a strict
        subset of ``tool_channel``, kept separate because accepting it costs an
        answer, not a retrieval);
      * ``stale_intent``     -- the speculated argument was already the argument
        of an action executed at some step <= i-2, i.e. the speculation re-asks
        something the trajectory already knows;
      * ``spec_fixation``    -- the speculated action repeats step i-1's action
        (the speculator is stuck on the page it is currently reading);
      * ``agent_loop``       -- the realized history already contains this
        normalized action twice or more, so accepting would extend a loop.

    ``channel`` in the pre-loss field list is ``tool_channel or
    terminal_channel``; it is reported under that name too.

    Parameters verified from the source:
      * none.  This battery is ours and its pre-loss definition was lost; the
        four field names are all that survived (in ``docs/STATE_RESUME.md`` and
        ``docs/LOST_WORK_MANIFEST.md``), the predicates below are new.

    Parameters chosen by us:
      * every predicate, including the step boundary that separates
        ``stale_intent`` (<= i-2) from ``spec_fixation`` (exactly i-1);
      * the ``>= 2`` repetition count for ``agent_loop``;
      * normalization for all history comparisons is ``parse_action`` +
        ``norm_arg`` (same as criterion 2);
      * ``binary = not any(flag)``.
    """
    st, sa = parse_action(pair.spec_action)
    rt, _ = parse_action(pair.real_action)
    ns = norm_arg(sa)

    hist = [parse_action(t.action) for t in pair.history]
    hist_norm = [(t, norm_arg(a)) for t, a in hist]

    tool_channel = st != rt
    terminal_channel = (st == "finish") != (rt == "finish")
    prev = hist_norm[-1] if hist_norm else None
    spec_fixation = bool(prev and prev == (st, ns))
    earlier = hist_norm[:-1] if hist_norm else []
    stale_intent = any(t == st and a == ns for t, a in earlier)
    agent_loop = sum(1 for t, a in hist_norm if t == st and a == ns) >= 2

    flags = {
        "tool_channel": tool_channel,
        "terminal_channel": terminal_channel,
        "channel": tool_channel or terminal_channel,
        "stale_intent": stale_intent,
        "spec_fixation": spec_fixation,
        "agent_loop": agent_loop,
    }
    # `channel` is a derived alias, not an independent filter: excluded from the
    # any() so it cannot double-count.
    fired = [k for k, v in flags.items() if v and k != "channel"]
    return CriterionResult(name="battery", level="call", output="binary",
                           binary=not fired, flags=flags,
                           detail={"fired": fired})


# ===========================================================================
# 4. edit_distance
# ===========================================================================

@register("edit_distance", level="call", output="score",
          source="DualSpec-attributed (arXiv:2603.07416 Section 6.1, which "
                 "characterizes DSP as 'minimum edit distance'); ABSENT from "
                 "DSP arXiv:2509.01920 and from its released code "
                 "(github.com/guanyilin428/Dynamic-Speculative-Planning, "
                 "OpenAGI/openagi_utils.py:37-39, which is `s == t`)",
          thresholds=EDIT_DISTANCE_GRID, faithful=False,
          notes="NOT DSP's criterion. DSP's released matcher is exact string "
                "equality (see criterion exact_dsp); the edit distance is a "
                "third-party characterization and its grid is ours. Renamed "
                "from edit_distance_dsp 2026-10-02 because the _dsp suffix "
                "asserted an attribution the source does not support.")
def edit_distance(pair: Pair, ctx: Context) -> CriterionResult:
    """Normalized Levenshtein distance on the argument string, DualSpec-attributed.

    **The attribution does not survive contact with the source.**  DSP's paper
    never defines its acceptance test beyond "whenever divergence occurs, where
    the approximation agent proposes a different action than the target, the
    system adopts ..." (§3), and its released code implements::

        def judge_to_be_true(s: str, t: str) -> bool:
            \"\"\"Check if two actions are equivalent.\"\"\"
            return s == t

    at ``OpenAGI/openagi_utils.py:37-39`` -- byte-exact equality, no distance,
    no threshold, no normalization.  The "minimum edit distance" reading comes
    from a *third party*: DualSpec (arXiv:2603.07416 §6.1) describes DSP as
    accepting "a draft only if it matches the base action (minimum edit
    distance)".  Since no distance or threshold is recoverable from DSP, we
    follow the fallback: normalized Levenshtein over the argument string with
    the grid {0.1 .. 0.5}, and we **do not call it DSP's criterion**.  DSP's
    actual predicate is registered separately as ``exact_dsp``;
    ``detail["dsp_exact_equal"]`` keeps it per-pair here too so a row is
    self-contained.

    Naming: this criterion was called ``edit_distance_dsp`` until 2026-10-02.
    The suffix claimed an attribution the source does not support, so it was
    dropped; the published variant table in docs/CRITERIA.md records the
    rename.

    Score is a **distance**: smaller is more similar, so a threshold ``t``
    accepts when ``score <= t``.

    Parameters verified from the source:
      * DSP's released acceptance predicate is ``s == t``
        (openagi_utils.py:37-39, repo default branch, fetched 2026-09-29);
      * DSP publishes no edit distance and no threshold (checked: paper §3 and
        the whole released ``util.py``/``openagi_utils.py``);
      * the "minimum edit distance" characterization exists, but in DualSpec
        §6.1, not in DSP.

    Parameters chosen by us:
      * the distance: Levenshtein over the ``norm_text``-normalized argument
        string, divided by the longer length;
      * argument-only comparison (the tool name is handled by ``battery``);
      * the threshold grid {0.1, 0.2, 0.3, 0.4, 0.5};
      * accept-when-``<=``.
    """
    st, sa = parse_action(pair.spec_action)
    rt, ra = parse_action(pair.real_action)
    dist = normalized_levenshtein(norm_text(sa), norm_text(ra))
    return CriterionResult(
        name="edit_distance", level="call", output="score", score=dist,
        decisions=_grid_decisions(dist, EDIT_DISTANCE_GRID, "le"),
        detail={"dsp_exact_equal": pair.spec_action == pair.real_action,
                "same_tool": st == rt,
                "raw_edit_distance": levenshtein(norm_text(sa), norm_text(ra)),
                "score_is": "distance (accept when <= threshold)"})


# ===========================================================================
# 4b. exact_dsp -- DSP's ACTUAL predicate, kept as its own criterion
# ===========================================================================

@register("exact_dsp", level="call", output="binary",
          source="DSP (Dynamic Speculative Agent Planning), arXiv:2509.01920; "
                 "released matcher github.com/guanyilin428/"
                 "Dynamic-Speculative-Planning, OpenAGI/openagi_utils.py:37-39: "
                 "`def judge_to_be_true(s, t): return s == t`",
          notes="Identically False on the Paper B population by construction "
                "(the population filter is lower(spec) != lower(real)); kept "
                "because the attribution belongs to a criterion that exists, "
                "not to edit_distance.")
def exact_dsp(pair: Pair, ctx: Context) -> CriterionResult:
    """DSP's released acceptance predicate, byte-exact ``s == t``.

    Split out from ``edit_distance`` on 2026-10-02. The two had been conflated
    under the name ``edit_distance_dsp``, which attributed a distance to a paper
    that implements equality. Keeping DSP's real predicate as a first-class
    criterion is what makes the attribution in ``edit_distance`` honest: the
    comparison is between *our* fallback and DSP's actual rule, not between our
    fallback and a straw man.

    **This criterion is identically ``False`` on every audited pair, and that is
    expected, not a bug.** The prereg's population is
    ``lower(spec_j) != lower(real_i)``, so two actions that differ only in case
    are already excluded, and anything still in population differs as raw
    bytes. Its value is therefore attributional and as a population invariant:
    a ``True`` here would mean the population filter is broken, which is why it
    is scored on every pair rather than asserted once.

    Parameters verified from the source:
      * the predicate is ``s == t`` -- byte-exact, no normalization, no
        threshold (openagi_utils.py:37-39, repo default branch, fetched
        2026-09-29);
      * it is applied to the action strings, and DSP defines no other
        acceptance test (paper Section 3; whole released ``util.py`` /
        ``openagi_utils.py`` checked).

    Parameters chosen by us:
      * nothing. The comparison is the source's, on the pair's raw
        ``spec_action`` / ``real_action`` strings as recorded.
    """
    equal = pair.spec_action == pair.real_action
    return CriterionResult(
        name="exact_dsp", level="call", output="binary", binary=equal,
        detail={"predicate": "spec_action == real_action (byte-exact)",
                "expected_constant_false_on_population": True,
                "population_filter": "lower(spec) != lower(real)"})


# ===========================================================================
# 5. embed_call
# ===========================================================================

@register("embed_call", level="call", output="score",
          source="threshold 0.80: SpecBox arXiv:2607.23933 §3.3/§4 (tau_c=0.8); "
                 "threshold 0.95: Cost-Aware arXiv:2606.07846 §7.4 (Tier 2)",
          thresholds=EMBED_GRID, requires=("embedder",))
def embed_call(pair: Pair, ctx: Context) -> CriterionResult:
    """Cosine similarity of the two argument strings' embeddings.

    SpecBox reuses a cached tool result "only when the tool identity matches and
    the semantic similarity exceeds the threshold tau_c = 0.8" (§3.3 Eq. for
    ``hit(x)``, restated in §4 Implementation).  Cost-Aware's Tier 2 accepts a
    speculation on "normalized-embedding cosine similarity >= 0.95 for text"
    (§7.4).  Neither paper names an embedding model, so ours is fixed and
    recorded in the run manifest.  SpecBox's tool-identity precondition is
    recorded in ``detail["same_tool"]`` rather than folded into the score, so
    the same score can be read with or without it.

    Parameters verified from the source:
      * 0.80 is SpecBox's tau_c, used "in the reported experiments" (§4);
      * 0.95 is Cost-Aware's default Tier-2 text threshold (§7.4);
      * both papers gate on cosine similarity of an embedding of the
        *invocation/argument*, and SpecBox additionally requires the tool
        identity to match (§3.3).

    Parameters chosen by us:
      * the embedding model (one fixed local model, default
        ``BAAI/bge-base-en-v1.5``; the resolved id and revision are written to
        the manifest);
      * embedding the argument string only, not the serialized whole call;
      * the interior grid points 0.85 and 0.90;
      * accept-when-``>=``.
    """
    if ctx.embedder is None:
        return _na("embed_call", "no embedder in context")
    _, sa = parse_action(pair.spec_action)
    st, _ = parse_action(pair.spec_action)
    rt, ra = parse_action(pair.real_action)
    va, vb = ctx.embedder.embed([sa, ra])
    cos = cosine(va, vb)
    return CriterionResult(
        name="embed_call", level="call", output="score", score=cos,
        decisions=_grid_decisions(cos, EMBED_GRID, "ge"),
        detail={"same_tool": st == rt,
                "embedder": getattr(ctx.embedder, "model_id", "?"),
                "revision": getattr(ctx.embedder, "revision", None)})


# ===========================================================================
# 6/7. our SE judge rubric v2
# ===========================================================================

#: Re-derived SE-v2 rubric.  The pre-loss ``se_judge_v2`` rules a-d are LOST and
#: their content was never restated anywhere (docs/LOST_WORK_MANIFEST.md item 6),
#: so this text is new.  It is shared verbatim by criteria 6 and 7 -- that
#: sharing is what makes hypothesis H4 (verbalized vs log-prob scoring of the
#: *same* rubric) a test of the scoring channel rather than of two rubrics.
SE_RUBRIC_V2 = """\
You are auditing a speculative action taken by a tool-using question-answering \
agent. The agent had already executed the steps in HISTORY. At the next step the \
authoritative model chose REAL ACTION; a faster speculator proposed SPECULATED \
ACTION instead. Decide whether executing SPECULATED ACTION in place of REAL \
ACTION leaves the agent's task-relevant information state equivalent.

Apply these rules in order:
(a) Tool channel. If the two actions call different tools, or exactly one of \
them is finish[...], the answer is No.
(b) Referent. If the two arguments denote the same target -- alias, alternative \
spelling, different word order, different capitalisation or diacritics, an \
article added or dropped, an abbreviation and its expansion -- treat them as the \
same target.
(c) Specificity. If one argument names a different entity, a different attribute \
of the same entity, or adds or removes a qualifier that changes which entity or \
section is retrieved, the answer is No.
(d) Retrievable content. Judge the content the two calls would return, not the \
strings. If they would return the same page or the same section, the answer is \
Yes, even if the strings differ; if they would return different content, the \
answer is No, even if the strings are similar.

QUESTION
{question}

HISTORY
{history}

REAL ACTION
{real_action}

SPECULATED ACTION
{spec_action}
"""

SE_RUBRIC_V2_VERBAL_TAIL = """
Answer on exactly two lines and nothing else:
Verdict: Yes or No
Confidence: an integer from 0 to 100, how confident you are in that verdict
"""

SE_RUBRIC_V2_LOGPROB_TAIL = """
Answer with exactly one word, Yes or No.
"""


def render_history(pair: Pair, max_obs_chars: int = 600) -> str:
    """Render the realized history for a judge prompt.

    Observations are truncated to ``max_obs_chars`` (ours) to bound the prompt;
    the truncation is marked so the judge cannot mistake it for a short page.
    """
    if not pair.history:
        return "(no previous steps)"
    lines = []
    for n, t in enumerate(pair.history, 1):
        obs = t.obs or ""
        if len(obs) > max_obs_chars:
            obs = obs[:max_obs_chars] + " ...[truncated]"
        if t.thought:
            lines.append(f"Thought {n}: {t.thought}")
        lines.append(f"Action {n}: {t.action}")
        lines.append(f"Observation {n}: {obs}")
    return "\n".join(lines)


def se_v2_prompt(pair: Pair, mode: str) -> str:
    body = SE_RUBRIC_V2.format(question=pair.question,
                               history=render_history(pair),
                               real_action=pair.real_action,
                               spec_action=pair.spec_action)
    tail = SE_RUBRIC_V2_VERBAL_TAIL if mode == "verbal" else SE_RUBRIC_V2_LOGPROB_TAIL
    return body + tail


_VERDICT_RE = re.compile(r"verdict\s*:\s*(yes|no)", re.IGNORECASE)
_CONF_RE = re.compile(r"confidence\s*:\s*(\d{1,3})", re.IGNORECASE)


def parse_verbalized(text: str) -> Tuple[Optional[bool], Optional[float], str]:
    """Parse ``Verdict:``/``Confidence:``.  Returns (verdict, confidence01, note)."""
    t = text or ""
    m = _VERDICT_RE.search(t)
    verdict: Optional[bool] = None
    note = ""
    if m:
        verdict = m.group(1).lower() == "yes"
    else:
        # Fallback (ours): first standalone yes/no token anywhere in the reply.
        m2 = re.search(r"\b(yes|no)\b", t, re.IGNORECASE)
        if m2:
            verdict = m2.group(1).lower() == "yes"
            note = "verdict recovered without the 'Verdict:' prefix"
        else:
            return None, None, "unparseable verdict"
    mc = _CONF_RE.search(t)
    if mc:
        conf = min(100, max(0, int(mc.group(1)))) / 100.0
    else:
        mc2 = re.search(r"\b(\d{1,3})\s*%", t)
        if mc2:
            conf = min(100, max(0, int(mc2.group(1)))) / 100.0
            note = (note + "; " if note else "") + "confidence read from a percentage"
        else:
            return verdict, None, (note + "; " if note else "") + "no confidence found"
    return verdict, conf, note


@register("judge_v2_verbal", level="call", output="score",
          source="ours (SE judge rubric v2, re-derived; pre-loss rules a-d are "
                 "LOST -- docs/LOST_WORK_MANIFEST.md item 6). Verbalized "
                 "confidence is the H4 comparator in docs/PREREG_PAPER_B.md",
          thresholds=VERBAL_CONF_GRID, requires=("judge",))
def judge_v2_verbal(pair: Pair, ctx: Context) -> CriterionResult:
    """SE rubric v2, scored by the judge's verbalized confidence.

    The judge emits ``Verdict: Yes|No`` and ``Confidence: 0-100``.  The two are
    folded into one acceptance score on [0, 1]:
    ``conf`` if the verdict is Yes, ``1 - conf`` if it is No.  A missing
    confidence is treated as ``1.0`` (fully confident) and flagged in
    ``detail``; an unparseable verdict is NA, never silently a reject.

    Parameters verified from the source:
      * none: the rubric is ours.  Only the *comparison* is pre-registered --
        H4 in ``docs/PREREG_PAPER_B.md`` predicts this scoring is weaker than
        criterion 7's log-prob margin on the same rubric.

    Parameters chosen by us:
      * the rubric text (``SE_RUBRIC_V2``), including rules (a)-(d);
      * the two-line output format and the fold
        ``score = conf if Yes else 1 - conf``;
      * treating a missing confidence as 1.0;
      * history observations truncated to 600 characters;
      * the grid {0.50, 0.60, 0.70, 0.80, 0.90, 0.95};
      * judge model: Qwen3-8B (recorded in the manifest), greedy, 32 max tokens.
    """
    if ctx.judge is None:
        return _na("judge_v2_verbal", "no judge in context")
    prompt = se_v2_prompt(pair, "verbal")
    text = ctx.judge.complete(prompt, max_tokens=32)
    verdict, conf, note = parse_verbalized(text)
    if verdict is None:
        r = _na("judge_v2_verbal", f"unparseable judge reply: {note}")
        r.detail["raw"] = text
        return r
    if conf is None:
        conf = 1.0
        note = (note + "; " if note else "") + "confidence defaulted to 1.0"
    score = conf if verdict else 1.0 - conf
    return CriterionResult(
        name="judge_v2_verbal", level="call", output="score", score=score,
        decisions=_grid_decisions(score, VERBAL_CONF_GRID, "ge"),
        detail={"verdict": verdict, "confidence": conf, "parse_note": note,
                "judge": getattr(ctx.judge, "model_id", "?"), "raw": text})


@register("judge_v2_logprob", level="call", output="score",
          source="ours (same SE judge rubric v2 as judge_v2_verbal); log-odds "
                 "scoring follows DualSpec arXiv:2603.07416 Eq. 6",
          thresholds=LOGPROB_MARGIN_GRID, requires=("judge",))
def judge_v2_logprob(pair: Pair, ctx: Context) -> CriterionResult:
    """SE rubric v2, scored by ``log p(Yes) - log p(No)`` on one token.

    Identical rubric text to criterion 6, different read-out: the judge is asked
    for exactly one word and the score is the log-odds margin between the
    ``Yes`` and ``No`` continuations at the first generated position.  This is
    the H4 comparator.

    Parameters verified from the source:
      * the log-probability margin ``log p_acc - log p_rej`` as a continuous
        verifier score is DualSpec's Eq. (6) (arXiv:2603.07416 §5); we reuse the
        functional form only -- the rubric is ours.

    Parameters chosen by us:
      * the rubric text (shared with criterion 6);
      * single-token read-out with ``max_tokens=1``, ``top_logprobs=20``;
      * case/whitespace variants of Yes and No are aggregated by log-sum-exp
        before the margin is taken (see ``backends.yes_no_margin``);
      * when one side is absent from the top-k, its log-probability is upper
        bounded by the smallest returned one and ``detail["clipped"]`` is set;
      * the margin grid {-4, -2, -1, 0, 1, 2, 4};
      * judge model: Qwen3-8B (recorded in the manifest).
    """
    if ctx.judge is None:
        return _na("judge_v2_logprob", "no judge in context")
    prompt = se_v2_prompt(pair, "logprob")
    margin, info = ctx.judge.yes_no_margin(prompt)
    if margin is None:
        r = _na("judge_v2_logprob", "no Yes/No mass in the judge's top-k")
        r.detail.update(info)
        return r
    return CriterionResult(
        name="judge_v2_logprob", level="call", output="score", score=margin,
        decisions=_grid_decisions(margin, LOGPROB_MARGIN_GRID, "ge"),
        detail={"judge": getattr(ctx.judge, "model_id", "?"), **info})


# ===========================================================================
# 8. dualspec_critic
# ===========================================================================

#: DualSpec arXiv:2603.07416 Appendix A.1 ("Verifier Prompt Template"),
#: transcribed verbatim from the arXiv HTML rendering of v1 (fetched
#: 2026-09-29).  Only the typographic quotation marks that LaTeXML emitted for
#: ``"No"``/``"Yes"`` are rendered as ASCII double quotes; no word, line break
#: or list item is changed, added or removed.  It judges *trajectory progress*,
#: not action equivalence, and is kept that way on purpose.
DUALSPEC_CRITIC_PROMPT = """\
[SYSTEM: TRAJECTORY AUDIT]
Review the recent steps (context). Is the agent making NEW PROGRESS?
REJECT ("No") if:
1.
Stagnation: Repeating similar queries or visiting same URLs (Looping).
2.
Ungrounded Answer: The Final Answer is NOT supported by the retrieved search results.
3.
Lazy/Drift: Queries are nested, vague, or irrelevant to User's Goal.
Verdict: Is the trajectory HEALTHY and PROGRESSING?
Answer only "Yes" or "No".
"""


def dualspec_prompt(pair: Pair) -> str:
    """DualSpec's critic prompt with our context block appended after it.

    Appendix A.1 states the critic is given "the current state s_t and a draft
    output consisting of an optional reasoning trace z_t and a candidate action
    a_t", but does not publish the serialization of s_t / z_t / a_t.  The
    prompt itself is left byte-identical; everything below the marker is our
    serialization, and is labelled as such in the prompt so a reader of the
    cache can tell the two apart.
    """
    return (DUALSPEC_CRITIC_PROMPT + "\n"
            "--- context (s_t, z_t, a_t), serialized by the auditor ---\n"
            f"User's Goal: {pair.question}\n"
            f"{render_history(pair)}\n"
            f"Candidate action (a_t): {pair.spec_action}\n")


@register("dualspec_critic", level="call", output="score",
          source="DualSpec, arXiv:2603.07416 App. A.1 (prompt, verbatim) + "
                 "Eq. 6 (log-odds score) + Eq. 7 (threshold tau)",
          thresholds=LOGPROB_MARGIN_GRID, requires=("judge",))
def dualspec_critic(pair: Pair, ctx: Context) -> CriterionResult:
    """DualSpec's trajectory-progress critic, scored by its log-odds margin.

    The critic prompt is copied verbatim from Appendix A.1 and **not adapted**:
    it asks whether the trajectory is healthy and progressing, not whether
    ``spec_j`` is equivalent to ``real_i``.  Keeping it faithful is the point --
    the audit gets to see how a progress critic behaves when it is used, as the
    literature uses it, to decide whether to keep a speculated step.  Note two
    consequences we do not repair: the critic never sees ``real_i``, and a
    speculation that is wrong but "healthy-looking" will score high.

    Parameters verified from the source:
      * the prompt text, verbatim (App. A.1);
      * the score: ``log p(Yes) - log p(No)``, the critic's log-odds of
        acceptance (Eq. 6);
      * the decision rule: accept iff ``score >= tau`` (Eq. 7);
      * the critic is the full-capacity model, and its verdict is binary
        Yes/No (§5, App. A.1).

    Parameters chosen by us:
      * the value of tau: DualSpec tunes it offline on held-out GAIA to a ~20%
        intervention rate (§6.1) and never publishes the number, so we sweep the
        grid {-4, -2, -1, 0, 1, 2, 4} instead of inventing one;
      * the serialization of s_t / z_t / a_t appended below the prompt
        (z_t is empty here: our speculator emits an action, not a reasoning
        trace, which Appendix A.1 covers as ``z_t = 0``);
      * history observations truncated to 600 characters;
      * judge model: Qwen3-8B (recorded in the manifest).
    """
    if ctx.judge is None:
        return _na("dualspec_critic", "no judge in context")
    margin, info = ctx.judge.yes_no_margin(dualspec_prompt(pair))
    if margin is None:
        r = _na("dualspec_critic", "no Yes/No mass in the critic's top-k")
        r.detail.update(info)
        return r
    return CriterionResult(
        name="dualspec_critic", level="call", output="score", score=margin,
        decisions=_grid_decisions(margin, LOGPROB_MARGIN_GRID, "ge"),
        detail={"judge": getattr(ctx.judge, "model_id", "?"),
                "judges": "trajectory progress, not equivalence", **info})


# ===========================================================================
# 9. spec_confidence
# ===========================================================================

def _span_token_indices(toks: Sequence[Dict[str, Any]],
                        start: int, end: int) -> List[int]:
    """Indices of the tokens whose characters overlap ``[start, end)``.

    Character offsets are recovered by concatenating the token strings in
    generation order, which is exact for a byte-level BPE detokenization.
    """
    idx = []
    pos = 0
    for i, t in enumerate(toks):
        s = t.get("token", "")
        nxt = pos + len(s)
        if pos < end and nxt > start:
            idx.append(i)
        pos = nxt
    return idx


@register("spec_confidence", level="call", output="score",
          source="SPORK, arXiv:2607.03333 §4.2 Eq. (2) (min top-1 token "
                 "probability over the tool-name span; theta = 0.90)",
          thresholds=SPEC_CONF_GRID, requires=("spec_tokens",))
def spec_confidence(pair: Pair, ctx: Context) -> CriterionResult:
    """Minimum top-1 token probability over the speculated call's name span.

    SPORK: "We define the confidence score as the minimum top-1 token
    probability over the tool-name span (the first L tokens after the ``"name":
    "`` prefix)", ``c = min_{i=2..L} exp(l_i)``, with the minimum starting at
    ``i = 2`` "because the first token of the span (the opening quote) is always
    high-probability boilerplate", and ``theta = 0.90`` selected as the
    F1-maximizing operating point (§4.2).

    In SPORK this is a *dispatch* signal -- it decides whether to launch the
    speculative tool call.  Here it is only ever a score: we never let it gate
    anything, we ask how well it separates the replay labels.

    Two spans are reported.  ``score`` is the tool-name span (SPORK's);
    ``detail["arg_span_score"]`` is the argument span, which is ours.

    Parameters verified from the source:
      * the statistic: minimum over the span of ``exp(top-1 logprob)`` (Eq. 2);
      * the span: the generated tool *name* only, not the arguments (§4.2);
      * the first token of the name span is dropped (``i`` starts at 2, ibid.);
      * theta = 0.90 (§4.2), included in our grid;
      * SPORK's own acceptance test is separate and strict -- "the probe is
        accepted if and only if its full tool call (name and serialized
        arguments) matches the main generation exactly" -- i.e. this score is a
        dispatch gate, not an acceptance criterion (§4.2, "Strict acceptance").

    Parameters chosen by us:
      * the mapping of SPORK's JSON surface syntax (``"name": "<tool>"``) onto
        our ReAct surface syntax (``tool[arg]``): the name span is the
        characters before the first ``[``, the argument span is the characters
        between the outermost brackets;
      * when the name span is a single token, that token's probability is used
        rather than an empty minimum (SPORK's ``i=2`` rule assumes a leading
        quote token that our syntax does not have); ``detail`` records it;
      * the argument-span variant, and taking its minimum over *all* its tokens;
      * the extra grid points {0.50, 0.70, 0.80, 0.95}.
    """
    toks = pair.spec_tokens
    if not toks:
        return _na("spec_confidence", "no speculator token logprobs recorded")
    text = "".join(t.get("token", "") for t in toks)
    # Locate the action inside the generated text (the speculator may emit
    # "Action 3: search[X]" -- the span is defined on the call itself).
    action = pair.spec_action
    at = text.find(action)
    if at < 0:
        at = text.find(action.strip())
    if at < 0:
        return _na("spec_confidence",
                   "spec_action not found in the recorded token stream")
    br = action.find("[")
    name_start, name_end = at, at + (br if br >= 0 else len(action))
    name_idx = _span_token_indices(toks, name_start, name_end)
    dropped_first = False
    if len(name_idx) > 1:
        name_idx = name_idx[1:]           # SPORK: minimum starts at i = 2
        dropped_first = True
    if not name_idx:
        return _na("spec_confidence", "empty tool-name span")
    name_score = min(math.exp(toks[i]["logprob"]) for i in name_idx)

    arg_score = None
    if br >= 0 and action.endswith("]"):
        arg_idx = _span_token_indices(toks, at + br + 1, at + len(action) - 1)
        if arg_idx:
            arg_score = min(math.exp(toks[i]["logprob"]) for i in arg_idx)

    detail = {
        "name_span_tokens": [toks[i].get("token") for i in name_idx],
        "dropped_first_name_token": dropped_first,
        "single_token_name_span": not dropped_first,
        "arg_span_score": arg_score,
        "arg_span_decisions": (_grid_decisions(arg_score, SPEC_CONF_GRID, "ge")
                               if arg_score is not None else None),
        "role": "dispatch signal in SPORK; scored only here",
    }
    return CriterionResult(
        name="spec_confidence", level="call", output="score", score=name_score,
        decisions=_grid_decisions(name_score, SPEC_CONF_GRID, "ge"),
        detail=detail)


# ===========================================================================
# 10. obs_equal
# ===========================================================================

@register("obs_equal", level="obs", output="binary",
          source="AOSpec, arXiv:2608.00881 §4.2 (\"retains only a continuation "
                 "whose predicted observation is byte-identical to o_t\")",
          requires=("spec_obs", "real_obs"), faithful=False,
          notes="primary output is normalized equality (ours); AOSpec's byte "
                "identity is the byte_identity variant and is equally "
                "reportable -- neither is a fallback for the other",
          variants=("byte_identity", "normalized"))
def obs_equal(pair: Pair, ctx: Context) -> CriterionResult:
    """Observation equality between the executed ``obs(spec_j)`` and ``obs(real_i)``.

    AOSpec's rule is strict byte identity, with normalization named only as a
    possibility it did not implement ("Canonicalizing inconsequential fields
    such as process UUIDs could improve acceptance", §4.2).  The audited primary
    output here is the *normalized* comparison, because our observations are
    truncated Wikipedia prose where a diacritic or a trailing period is not a
    semantic difference; the faithful byte-identity verdict is kept in
    ``detail["byte_identical"]`` so both can be reported.

    Parameters verified from the source:
      * comparison is between the predicted/candidate observation and the
        observation the real execution returned, and acceptance is byte
        identity (§4.2);
      * AOSpec applies no normalization before the comparison (ibid.).

    Both readings are registered as **named variants**, scored on every pair:

      * ``byte_identity`` -- AOSpec-faithful, ``spec_obs == real_obs``;
      * ``normalized``    -- ours, ``norm_text`` applied to both sides.

    ``binary`` stays the normalized verdict so the column keeps its meaning
    across the pre-loss record, but the variants are what should be reported
    side by side. Registering both before the audit is the point: which of the
    two is "the" obs criterion is a live ruling (``CLAUDE.md`` section 6,
    obs-identity), and that ruling must not be settled by whichever number
    turns out nicer.

    Parameters chosen by us:
      * making the normalized comparison the primary ``binary`` -- a deliberate
        deviation, flagged as ``faithful=False``;
      * the normalizer: ``norm_text`` (lowercase, diacritics, punctuation,
        whitespace).
    """
    na = _require_executed_obs(pair)
    if na:
        return _na("obs_equal", na)
    byte_eq = pair.spec_obs == pair.real_obs
    norm_eq = norm_text(pair.spec_obs) == norm_text(pair.real_obs)
    return CriterionResult(name="obs_equal", level="obs", output="binary",
                           binary=norm_eq,
                           detail={"byte_identical": byte_eq,
                                   "normalized_equal": norm_eq},
                           variants={
                               "byte_identity": {
                                   "binary": byte_eq,
                                   "detail": {"faithful_to": "AOSpec 2608.00881 "
                                                             "section 4.2"}},
                               "normalized": {
                                   "binary": norm_eq,
                                   "detail": {"normalizer": "norm_text"}},
                           })


# ===========================================================================
# 11. spechop_rules
# ===========================================================================

def spechop_verify(candidate: str, target: str,
                   stopwords: frozenset = None,
                   refusal_patterns: Sequence[str] = None,
                   ) -> Tuple[bool, Dict[str, Any]]:
    """SpecHop's deterministic rule verifier (App. D.4), constants verified.

    Pipeline, in the appendix's order:

    1. normalize both texts (lowercase, diacritic and punctuation removal);
    2. reject the candidate if it matches a refusal/uncertainty pattern;
    3. if the target contains multi-digit numbers, the candidate must contain
       matching numeric values, else reject;
    4. accept on a direct substring match, or -- after stopword removal --
       ``>= 72%`` token coverage, or Jaccard ``>= 0.55``;
    5. a target under 5 characters requires a perfect token match instead.

    ``stopwords`` and ``refusal_patterns`` are the two inputs the appendix does
    **not** publish, so they are parameters rather than constants: criterion 11
    registers the alternatives as named variants (see ``STOPWORD_LISTS`` and
    ``REFUSAL_LISTS``). Defaults are ours, i.e. the primary scoring.
    """
    if stopwords is None:
        stopwords = STOPWORDS
    if refusal_patterns is None:
        refusal_patterns = REFUSAL_PATTERNS
    nc, nt = norm_text(candidate), norm_text(target)
    info: Dict[str, Any] = {"target_norm_len": len(nt)}

    for pat in refusal_patterns:
        if norm_text(pat) in nc:
            info["refusal_pattern"] = pat
            return False, info

    tgt_nums = has_multi_digit_numbers(nt)
    if tgt_nums:
        cand_nums = set(has_multi_digit_numbers(nc))
        missing = [n for n in tgt_nums if n not in cand_nums]
        info["target_numbers"] = tgt_nums
        info["missing_numbers"] = missing
        if missing:
            return False, info

    if len(nt) < SPECHOP_SHORT_TARGET_CHARS:
        exact = tokens(nc) == tokens(nt)
        info["rule"] = "short_target_exact_token_match"
        info["accept"] = exact
        return exact, info

    if nt and nt in nc:
        info["rule"] = "substring"
        info["accept"] = True
        return True, info

    ct, cc = (content_tokens(nt, stopwords), content_tokens(nc, stopwords))
    set_t, set_c = set(ct), set(cc)
    coverage = (len(set_t & set_c) / len(set_t)) if set_t else 0.0
    union = set_t | set_c
    jaccard = (len(set_t & set_c) / len(union)) if union else 0.0
    info.update({"coverage": coverage, "jaccard": jaccard,
                 "rule": "lexical_overlap"})
    accept = coverage >= SPECHOP_TOKEN_COVERAGE or jaccard >= SPECHOP_JACCARD
    info["accept"] = accept
    return accept, info


@register("spechop_rules", level="obs", output="binary",
          source="SpecHop, arXiv:2605.21965 App. D.4 (\"Deterministic "
                 "Rule-Based Verifier (V) Implementation\"); §4.1 for the "
                 "normalize / exact-match / token-set-Jaccard summary",
          requires=("spec_obs", "real_obs"),
          variants=("stopwords_nltk", "stopwords_sklearn", "refusal_minimal"),
          notes="The four numeric constants are the appendix's; the stopword "
                "list and the refusal-pattern list are NOT published, so both "
                "are registered with named alternatives. The variants bound "
                "how much of this criterion's behaviour is ours.")
def spechop_rules(pair: Pair, ctx: Context) -> CriterionResult:
    """SpecHop's rule verifier between the executed ``obs(spec_j)`` and ``obs(real_i)``.

    Candidate = ``obs(spec_j)``, target = ``obs(real_i)``: SpecHop's verifier
    asks whether the speculative observation is equivalent to the target tool's
    observation, and its conservatism is one-directional (it must not produce
    false positives), so the direction matters and is fixed this way.

    Parameters verified from the source: (App. D.4, quoted)
      * normalization: "lowercasing, diacritic and punctuation removal";
      * "immediately rejects any speculative outputs that match common refusal
        or uncertainty patterns (e.g., 'I don't know,' 'information
        unavailable')";
      * "if the target observation contains multi-digit numbers (such as years
        or quantities), the speculative observation must contain matching
        numeric values, otherwise it is automatically rejected";
      * "Final acceptance requires either a direct substring match or a high
        degree of lexical overlap. Specifically, after filtering out standard
        English stopwords, the speculation must achieve either >= 72% token
        coverage or a Jaccard similarity of >= 0.55";
      * "For exceptionally short answers (under 5 characters), the verifier
        defaults to requiring a perfect token match";
      * hence all four constants -- 72%, 0.55, 5 characters, and the ordering of
        the stages -- are the paper's.  SpecHop's repo
        (github.com/mehrdadsaberi/spechop) contains only a README, so the
        appendix is the whole recoverable specification.

    Parameters chosen by us:
      * the refusal/uncertainty pattern list: the appendix gives two examples
        and no list, so ``REFUSAL_PATTERNS`` is ours (its first two entries are
        the appendix's examples);
      * the stopword list (``STOPWORDS``), unpublished by the appendix;
      * the direction of "token coverage": fraction of the *target's* content
        tokens present in the candidate;
      * "matching numeric values" read as set containment of every multi-digit
        number of the target;
      * Jaccard computed over stopword-filtered token *sets*;
      * "under 5 characters" measured on the normalized target;
      * tokenization: whitespace split after normalization.

    Registered variants, all scored in the same pass (deterministic, no judge
    call, so the sensitivity analysis is free):

      * ``stopwords_nltk``    -- ``STOPWORDS_NLTK``, NLTK's published English
        list, projected onto our token space by ``normalize_stopword_list``;
      * ``stopwords_sklearn`` -- ``STOPWORDS_SKLEARN``, scikit-learn's
        ``ENGLISH_STOP_WORDS`` (the Glasgow IR list), same projection;
      * ``refusal_minimal``   -- ``REFUSAL_PATTERNS_MINIMAL``, only the two
        patterns App. D.4 actually names.

    Each variant changes exactly one unpublished input and holds the four
    published constants fixed, so a disagreement between the primary and a
    variant localizes to the input we had to invent. They are registered now,
    before the audit, so the choice among them cannot be made after seeing the
    labels.
    """
    na = _require_executed_obs(pair)
    if na:
        return _na("spechop_rules", na)
    ok, info = spechop_verify(pair.spec_obs, pair.real_obs)

    variants: Dict[str, Dict[str, Any]] = {}
    for vname, kwargs in (
        ("stopwords_nltk", {"stopwords": STOPWORDS_NLTK}),
        ("stopwords_sklearn", {"stopwords": STOPWORDS_SKLEARN}),
        ("refusal_minimal", {"refusal_patterns": REFUSAL_PATTERNS_MINIMAL}),
    ):
        v_ok, v_info = spechop_verify(pair.spec_obs, pair.real_obs, **kwargs)
        variants[vname] = {"binary": v_ok,
                           "detail": v_info,
                           "agrees_with_primary": v_ok == ok}

    return CriterionResult(name="spechop_rules", level="obs", output="binary",
                           binary=ok, detail=info, variants=variants)


# ===========================================================================
# 12. embed_obs
# ===========================================================================

@register("embed_obs", level="obs", output="score",
          source="thresholds as in embed_call: 0.80 SpecBox arXiv:2607.23933 "
                 "§3.3/§4; 0.95 Cost-Aware arXiv:2606.07846 §7.4 Tier 2",
          thresholds=EMBED_GRID, requires=("embedder", "spec_obs", "real_obs"))
def embed_obs(pair: Pair, ctx: Context) -> CriterionResult:
    """Cosine similarity of the two observations' embeddings.

    Same statistic, model and grid as ``embed_call``, applied to the executed
    ``obs(spec_j)`` against ``obs(real_i)``.  Neither source applies its
    threshold to observations -- SpecBox scores invocation signatures,
    Cost-Aware scores predicted inputs -- so the transfer to the observation
    level is ours, and is the reason this criterion exists separately.

    Parameters verified from the source:
      * 0.80 (SpecBox tau_c, §3.3/§4) and 0.95 (Cost-Aware Tier 2, §7.4) as
        published cosine thresholds for accepting a semantic match;
      * neither paper names an embedding model.

    Parameters chosen by us:
      * applying those thresholds to *observations* rather than to calls;
      * the embedding model (default ``BAAI/bge-base-en-v1.5``, recorded);
      * embedding the full observation text with no truncation beyond the
        model's own context limit (the backend records when it truncates);
      * the interior grid points 0.85 and 0.90; accept-when-``>=``.
    """
    if ctx.embedder is None:
        return _na("embed_obs", "no embedder in context")
    na = _require_executed_obs(pair)
    if na:
        return _na("embed_obs", na)
    va, vb = ctx.embedder.embed([pair.spec_obs, pair.real_obs])
    cos = cosine(va, vb)
    return CriterionResult(
        name="embed_obs", level="obs", output="score", score=cos,
        decisions=_grid_decisions(cos, EMBED_GRID, "ge"),
        detail={"embedder": getattr(ctx.embedder, "model_id", "?"),
                "revision": getattr(ctx.embedder, "revision", None)})


# ===========================================================================
# 13. sufficient_context
# ===========================================================================

#: Sufficient Context autorater prompt, verbatim from the released repo
#: (github.com/hljoren/sufficientcontext, README.md, "Sufficient Context
#: Autorater Prompt", fetched 2026-09-29).  Line breaks are the repo's.  The
#: repo ships no code, so the README is the authoritative copy; the paper is
#: arXiv:2411.06037.  ``<TIMESTAMP>``, ``<question>`` and ``<context>`` are the
#: repo's own placeholders and are the only substitutions we make.
SUFFICIENT_CONTEXT_PROMPT = """\
You are an expert LLM evaluator that excels at evaluating a QUESTION and REFERENCES.
Consider the following criteria:
Sufficient Context: 1 IF the CONTEXT is sufficient to infer the answer to the question and 0
IF the CONTEXT cannot be used to infer the answer to the question
Assume the queries have timestamp <TIMESTAMP>.
First, output a list of step-by-step questions that would be used to arrive at a label for the
criteria. Make sure to include questions about assumptions implicit in the QUESTION.
Include questions about any mathematical calculations or arithmetic that would be required.
Next, answer each of the questions. Make sure to work step by step through any required
mathematical calculations or arithmetic. Finally, use these answers to evaluate the criteria.
Output the ### EXPLANATION (Text). Then, use the EXPLANATION to output the ###
EVALUATION (JSON)
EXAMPLE:
### QUESTION
In which year did the publisher of Roald Dahl’s Guide to Railway Safety cease to exist?
### References
Roald Dahl’s Guide to Railway Safety was published in 1991 by the British Railways Board.
The British Railways Board had asked Roald Dahl to write the text of the booklet, and
Quentin Blake to illustrate it, to help young people enjoy using the railways safely. The
British Railways Board (BRB) was a nationalised industry in the United Kingdom that
operated from 1963 to 2001. Until 1997 it was responsible for most railway services in Great
Britain, trading under the brand name British Railways and, from 1965, British Rail. It
did not operate railways in Northern Ireland, where railways were the responsibility of the
Government of Northern Ireland.
### EXPLANATION
The context mentions that Roald Dahl’s Guide to Railway Safety was published by the
British Railways Board. It also states that the British Railways Board operated from 1963 to
2001, meaning the year it ceased to exist was 2001. Therefore, the context does provide a
precise answer to the question.
### JSON
{"Sufficient Context": 1}
Remember the instructions: You are an expert LLM evaluator that excels at evaluating a
QUESTION and REFERENCES. Consider the following criteria:
Sufficient Context: 1 IF the CONTEXT is sufficient to infer the answer to the question and 0
IF the CONTEXT cannot be used to infer the answer to the question
Assume the queries have timestamp TIMESTAMP.
First, output a list of step-by-step questions that would be used to arrive at a label for the
criteria. Make sure to include questions about assumptions implicit in the QUESTION
Include questions about any mathematical calculations or arithmetic that would be required.
Next, answer each of the questions. Make sure to work step by step through any required
mathematical calculations or arithmetic. Finally, use these answers to evaluate the criteria.
Output the ### EXPLANATION (Text). Then, use the EXPLANATION to output the ###
EVALUATION (JSON)
### QUESTION
<question>
### REFERENCES
<context>
"""

_SC_JSON_RE = re.compile(r'"Sufficient\s+Context"\s*:\s*([01])')


def sufficient_context_references(pair: Pair) -> str:
    """Build the REFERENCES block: the history's observations, then obs(spec_j).

    Ours: the repo's prompt has a single REFERENCES field and says nothing about
    multi-step agents.  Order is chronological so the speculated observation is
    last.
    """
    parts = []
    for n, t in enumerate(pair.history, 1):
        if t.obs:
            parts.append(f"[step {n}: {t.action}]\n{t.obs}")
    parts.append(f"[speculated step: {pair.spec_action}]\n{pair.spec_obs or ''}")
    return "\n\n".join(parts)


def sufficient_context_prompt(pair: Pair, timestamp: str) -> str:
    return (SUFFICIENT_CONTEXT_PROMPT
            .replace("<TIMESTAMP>", timestamp)
            .replace("<question>", pair.question)
            .replace("<context>", sufficient_context_references(pair)))


def parse_sufficient_context(text: str) -> Optional[int]:
    """Read ``{"Sufficient Context": 0|1}`` out of the autorater's reply."""
    if not text:
        return None
    ms = _SC_JSON_RE.findall(text)
    if ms:
        return int(ms[-1])            # the final JSON block is the verdict
    return None


@register("sufficient_context", level="obs", output="binary",
          source="Sufficient Context, arXiv:2411.06037; prompt verbatim from "
                 "github.com/hljoren/sufficientcontext README.md "
                 "(\"Sufficient Context Autorater Prompt\")",
          requires=("judge", "spec_obs"),
          notes="question-level sufficiency, deliberately NOT an execution "
                "label -- kept as a contrast")
def sufficient_context(pair: Pair, ctx: Context) -> CriterionResult:
    """Autorater: do question + history + ``obs(spec_j)`` suffice to answer?

    This criterion is a *contrast*, not a competitor: it asks a question-level
    question ("is the retrieved context sufficient to infer the answer"), which
    is not what the replay labels measure ("does substituting this call change
    the trajectory's outcome").  It is kept faithful -- the prompt is the
    released one, unedited -- so that the audit can show how far a sufficiency
    autorater is from an execution-grounded acceptance criterion.

    It is obs-level because it reads the executed ``obs(spec_j)``; the real
    action's observation is *not* shown, so the rater cannot cheat by comparing.

    Parameters verified from the source:
      * the full prompt text, verbatim, including the one-shot Roald Dahl
        example and the repeated instruction block (repo README);
      * the output format ``{"Sufficient Context": 1|0}`` (ibid.);
      * the autorater is a 1-shot prompt (README: "1-shot Gemini 1.5 Pro
        prompt classifies sufficiency", 93% accuracy).

    Parameters chosen by us:
      * judge model Qwen3-8B instead of Gemini 1.5 Pro (local-only rule,
        CLAUDE.md §2) -- so the 93% accuracy figure does not transfer;
      * the ``<TIMESTAMP>`` substitution: ``2019-08-01``, the frozen KILT
        snapshot date (``docs/LOCAL_WIKI.md``);
      * the composition of ``<context>``: each history observation in order,
        each labelled with its action, then the speculated step's executed
        observation last;
      * ``max_tokens=512`` for the chain-of-questions the prompt requires;
      * taking the *last* ``"Sufficient Context": n`` match in the reply (the
        prompt's own example contains one earlier).
    """
    if ctx.judge is None:
        return _na("sufficient_context", "no judge in context")
    if pair.spec_obs is None:
        return _na("sufficient_context", "no executed observation for spec_j")
    if pair.spec_obs_source != "executed":
        return _na("sufficient_context",
                   f"spec_obs_source={pair.spec_obs_source!r}: obs-level "
                   "criteria refuse speculator-imagined observations")
    prompt = sufficient_context_prompt(pair, ctx.timestamp_for_sufficient_context)
    text = ctx.judge.complete(prompt, max_tokens=512)
    verdict = parse_sufficient_context(text)
    if verdict is None:
        r = _na("sufficient_context", "unparseable autorater reply")
        r.detail["raw"] = text
        return r
    return CriterionResult(name="sufficient_context", level="obs",
                           output="binary", binary=bool(verdict),
                           detail={"judge": getattr(ctx.judge, "model_id", "?"),
                                   "raw_tail": text[-400:]})


# ===========================================================================
# Runner
# ===========================================================================

#: criteria 6-8 are re-run with the non-Qwen judge when one is configured
#: (same-family circularity check, per the task brief).
CIRCULARITY_CHECK_CRITERIA: Tuple[str, ...] = (
    "judge_v2_verbal", "judge_v2_logprob", "dualspec_critic")


def score_pair(pair: Pair, ctx: Context,
               only: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Score one pair against every registered criterion.

    Returns the ``criteria_scores.jsonl`` row: keyed by ``pair_id``, with the
    replay labels copied through unmodified and a ``judge_alt`` block holding
    the non-Qwen re-run of criteria 6-8 (``None`` when no alternate judge is
    configured, so its absence is visible in the data rather than implied).
    """
    names = list(only) if only else list(_ORDER)
    out: Dict[str, Any] = {
        "pair_id": pair.pair_id,
        "run": pair.run,
        "step_i": pair.step_i,
        "spec_index": pair.spec_index,
        "spec_action": pair.spec_action,
        "real_action": pair.real_action,
        "spec_obs_source": pair.spec_obs_source,
        "labels": dict(pair.labels),
        "criteria": {},
    }
    for name in names:
        out["criteria"][name] = CRITERIA[name].fn(pair, ctx).to_dict()

    if ctx.judge_alt is not None:
        alt_ctx = Context(embedder=ctx.embedder, judge=ctx.judge_alt,
                          judge_alt=None,
                          timestamp_for_sufficient_context=ctx.timestamp_for_sufficient_context,
                          code_version=ctx.code_version)
        out["judge_alt"] = {
            "model_id": getattr(ctx.judge_alt, "model_id", "?"),
            "criteria": {n: CRITERIA[n].fn(pair, alt_ctx).to_dict()
                         for n in CIRCULARITY_CHECK_CRITERIA if n in names},
        }
    else:
        out["judge_alt"] = None
    return out


def score_pairs(pairs: Sequence[Pair], ctx: Context,
                out_path: Optional[str] = None,
                only: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    """Score many pairs; optionally append the rows to ``out_path`` as JSONL."""
    rows = [score_pair(p, ctx, only=only) for p in pairs]
    if out_path:
        with open(out_path, "a", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return rows


def load_pairs(path: str) -> List[Pair]:
    """Read a pairs JSONL file (``#`` comment lines and blanks are skipped)."""
    pairs = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            pairs.append(Pair.from_dict(json.loads(line)))
    return pairs


def registry_table() -> List[Dict[str, Any]]:
    """The registry, in registration order -- the source of docs/CRITERIA.md."""
    return [CRITERIA[n].describe() for n in _ORDER]

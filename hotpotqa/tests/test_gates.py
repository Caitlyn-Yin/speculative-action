"""Unit tests for the acceptance-criterion battery (``src/gates.py``).

Offline: no server, no weights, no network.  The judge and the embedder are
stubs, so what is tested is the criteria themselves -- the parsing, the
normalization, the published constants, the span arithmetic, the obs-level
guard and the output schema -- on the synthetic fixture
``tests/fixtures/pairs_synthetic.jsonl``.

The fixture's replay labels are invented (see its generator's docstring); no
test computes a precision or an AUROC from them.  Those come from real data in
the audit task.

Runs under pytest, and also standalone -- ``python3 tests/test_gates.py`` --
because pytest is not installed in either environment on this node.  The one
test that needs the real upstream comparator skips itself when ``src.metrics``
cannot be imported (it pulls in pandas).
"""

import hashlib
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import backends, gates  # noqa: E402
from src.gates import Context, Pair  # noqa: E402

FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "fixtures", "pairs_synthetic.jsonl")

# Registration order.  `exact_dsp` sits right after `edit_distance` because it
# is the other half of criterion 4: the distance is DualSpec's characterization,
# the equality is DSP's actual released predicate (2026-10-02 split).
EXPECTED_NAMES = [
    "exact_sa", "normalized", "battery", "edit_distance", "exact_dsp",
    "embed_call", "judge_v2_verbal", "judge_v2_logprob", "dualspec_critic",
    "spec_confidence", "obs_equal", "spechop_rules", "embed_obs",
    "sufficient_context",
]

#: Variants registered before the audit (docs/CRITERIA.md section 9).
EXPECTED_VARIANTS = {
    "obs_equal": ("byte_identity", "normalized"),
    "spechop_rules": ("stopwords_nltk", "stopwords_sklearn", "refusal_minimal"),
}


# ---------------------------------------------------------------------------
# stubs
# ---------------------------------------------------------------------------

class StubEmbedder:
    """Deterministic hashed bag-of-tokens embedding; cosine(x, x) == 1."""

    model_id = "stub-hash-64"
    revision = "n/a"

    def __init__(self, dim=64):
        self.dim = dim
        self.calls = []

    def embed(self, texts):
        self.calls.append(list(texts))
        out = []
        for t in texts:
            vec = [0.0] * self.dim
            for tok in gates.tokens(t) or ["<empty>"]:
                h = hashlib.sha256(tok.encode()).digest()
                vec[h[0] % self.dim] += 1.0
            out.append(vec)
        return out


class StubJudge:
    """Scripted judge.  ``replies``/``margins`` are matched on prompt substrings."""

    def __init__(self, model_id="stub-judge", replies=None, margins=None,
                 default_reply="Verdict: No\nConfidence: 60",
                 default_margin=0.0):
        self.model_id = model_id
        self.revision = "n/a"
        self.replies = replies or {}
        self.margins = margins or {}
        self.default_reply = default_reply
        self.default_margin = default_margin
        self.prompts = []

    def complete(self, prompt, max_tokens=None):
        self.prompts.append(("complete", prompt, max_tokens))
        for needle, reply in self.replies.items():
            if needle in prompt:
                return reply
        return self.default_reply

    def yes_no_margin(self, prompt):
        self.prompts.append(("margin", prompt, None))
        for needle, m in self.margins.items():
            if needle in prompt:
                return m, {"stub": True}
        return self.default_margin, {"stub": True}


def load_fixture():
    pairs = gates.load_pairs(FIXTURE)
    return {p.pair_id: p for p in pairs}


PAIRS = load_fixture()
CTX = Context(code_version="test")


def run(name, pair_id, ctx=None):
    return gates.CRITERIA[name].fn(PAIRS[pair_id], ctx or CTX)


# ---------------------------------------------------------------------------
# registry / contract
# ---------------------------------------------------------------------------

def test_registry_has_all_fourteen_in_order():
    # 13 audited criteria + exact_dsp, which is criterion 4's other half.
    assert list(gates.CRITERIA) == EXPECTED_NAMES, list(gates.CRITERIA)
    assert len(gates.registry_table()) == 14


def test_registered_variants_are_exactly_the_pre_audit_set():
    """The variant set is a pre-registration, so it is pinned by a test.

    Adding a variant after seeing the labels would be a way to pick the
    flattering parameterization; this test makes that show up as a diff.
    """
    declared = {c["name"]: tuple(c["variants"])
                for c in gates.registry_table() if c["variants"]}
    assert declared == EXPECTED_VARIANTS, declared


def test_declared_variants_are_actually_returned():
    ctx = Context(embedder=StubEmbedder(),
                  judge=StubJudge(default_reply="Verdict: Yes\nConfidence: 75"))
    pair = PAIRS["T11-obs-byte-identical"]
    for name, expected in EXPECTED_VARIANTS.items():
        res = gates.CRITERIA[name].fn(pair, ctx).to_dict()
        assert set(res.get("variants", {})) == set(expected), (name, res)


def test_exact_dsp_is_dsps_released_predicate():
    """DSP's matcher is `s == t` (openagi_utils.py:37-39), nothing more."""
    ctx = Context()
    for pair in PAIRS.values():
        res = gates.CRITERIA["exact_dsp"].fn(pair, ctx)
        assert res.binary == (pair.spec_action == pair.real_action), pair.pair_id
    # It is faithful to its source, unlike edit_distance.
    assert gates.CRITERIA["exact_dsp"].faithful is True
    assert gates.CRITERIA["edit_distance"].faithful is False


def test_exact_dsp_is_constant_false_on_in_population_pairs():
    """A True here would mean the prereg population filter is broken.

    Population is `lower(spec) != lower(real)`, so a raw-byte equality can
    never hold. Scored on every pair as a cheap population invariant.
    """
    ctx = Context()
    for pair in PAIRS.values():
        if pair.spec_action.lower() == pair.real_action.lower():
            continue                      # fixture pair outside the population
        assert gates.CRITERIA["exact_dsp"].fn(pair, ctx).binary is False


def test_stopword_variants_are_the_published_lists():
    # Hashes recorded in gates.py next to each literal; recomputed here so a
    # silent edit to either list fails the suite.
    import hashlib
    h = lambda ws: hashlib.sha256(" ".join(sorted(ws)).encode()).hexdigest()
    assert len(gates.STOPWORDS_NLTK_RAW) == 198
    assert h(gates.STOPWORDS_NLTK_RAW) == (
        "97f4fd27ecb1ef242e68e83c16b1f7a903d78a42eb719f6c1e7f40d313e97443")
    assert len(gates.STOPWORDS_SKLEARN_RAW) == 318
    assert h(gates.STOPWORDS_SKLEARN_RAW) == (
        "e570e9b41eab43e963c44d1d8b7ad441d084fa84f1104e01c9e8b41ad43feb89")


def test_stopword_normalization_is_necessary_not_cosmetic():
    """NLTK's apostrophe entries would be dead without the projection."""
    assert "aren't" in gates.STOPWORDS_NLTK_RAW
    # As published it could never match a token: norm_text strips punctuation.
    assert "aren't" not in gates.STOPWORDS_NLTK
    assert "aren" in gates.STOPWORDS_NLTK
    # Our own literal was already written in the normalized space.
    assert gates.normalize_stopword_list(gates.STOPWORDS) == gates.STOPWORDS


def test_refusal_minimal_is_exactly_the_appendixs_two_examples():
    assert gates.REFUSAL_PATTERNS_MINIMAL == ("I don't know",
                                              "information unavailable")
    # ...and is a subset of ours, so the variant isolates our additions.
    assert set(gates.REFUSAL_PATTERNS_MINIMAL) <= set(gates.REFUSAL_PATTERNS)


def test_spechop_variants_hold_the_published_constants_fixed():
    """Each variant changes one unpublished input and nothing else."""
    cand, tgt = "The Woolworth Building opened in 1913", "Woolworth Building 1913"
    base, _ = gates.spechop_verify(cand, tgt)
    for kwargs in ({"stopwords": gates.STOPWORDS_NLTK},
                   {"stopwords": gates.STOPWORDS_SKLEARN},
                   {"refusal_patterns": gates.REFUSAL_PATTERNS_MINIMAL}):
        ok, info = gates.spechop_verify(cand, tgt, **kwargs)
        assert isinstance(ok, bool)
        # the numeric gate is the appendix's and must still be applied
        assert info.get("target_numbers") == ["1913"], info


def test_obs_equal_variants_split_byte_from_normalized():
    ctx = Context()
    pair = PAIRS["T11-obs-byte-identical"]
    res = gates.CRITERIA["obs_equal"].fn(pair, ctx)
    assert res.variants["byte_identity"]["binary"] is True
    assert res.variants["normalized"]["binary"] is True
    # primary stays the normalized verdict
    assert res.binary == res.variants["normalized"]["binary"]


def test_every_criterion_declares_level_output_and_source():
    for c in gates.registry_table():
        assert c["level"] in ("call", "obs"), c
        assert c["output"] in ("binary", "score"), c
        assert c["source"], c
        assert (c["output"] == "score") == bool(c["thresholds"]), c


def test_docstrings_separate_verified_from_chosen():
    # register() enforces this at import time; assert it is actually enforcing.
    for name, c in gates.CRITERIA.items():
        assert "Parameters verified from the source:" in c.doc, name
        assert "Parameters chosen by us:" in c.doc, name
    try:
        @gates.register("bad_criterion", level="call", output="binary", source="x")
        def bad(pair, ctx):
            """No headings here."""
    except AssertionError:
        pass
    else:  # pragma: no cover
        raise AssertionError("register() accepted a criterion with no headings")
    finally:
        gates.CRITERIA.pop("bad_criterion", None)
        if "bad_criterion" in gates._ORDER:
            gates._ORDER.remove("bad_criterion")


def test_levels_are_the_published_split():
    call = {n for n, c in gates.CRITERIA.items() if c.level == "call"}
    obs = {n for n, c in gates.CRITERIA.items() if c.level == "obs"}
    assert call == {"exact_sa", "normalized", "battery", "edit_distance",
                    "exact_dsp", "embed_call", "judge_v2_verbal",
                    "judge_v2_logprob", "dualspec_critic", "spec_confidence"}
    assert obs == {"obs_equal", "spechop_rules", "embed_obs",
                   "sufficient_context"}


def test_published_constants_are_exactly_the_published_values():
    assert gates.SPECHOP_TOKEN_COVERAGE == 0.72      # SpecHop App. D.4
    assert gates.SPECHOP_JACCARD == 0.55             # SpecHop App. D.4
    assert gates.SPECHOP_SHORT_TARGET_CHARS == 5     # SpecHop App. D.4
    assert 0.80 in gates.EMBED_GRID                  # SpecBox tau_c
    assert 0.95 in gates.EMBED_GRID                  # Cost-Aware Tier 2
    assert 0.90 in gates.SPEC_CONF_GRID              # SPORK theta
    assert gates.EDIT_DISTANCE_GRID == (0.1, 0.2, 0.3, 0.4, 0.5)


def test_verbatim_prompts_are_intact():
    p = gates.DUALSPEC_CRITIC_PROMPT
    assert p.startswith("[SYSTEM: TRAJECTORY AUDIT]")
    assert "Is the agent making NEW PROGRESS?" in p
    for rule in ("Stagnation:", "Ungrounded Answer:", "Lazy/Drift:"):
        assert rule in p
    assert p.rstrip().endswith('Answer only "Yes" or "No".')
    # it judges progress, not equivalence: the real action is never named
    assert "REAL ACTION" not in p and "equivalent" not in p

    sc = gates.SUFFICIENT_CONTEXT_PROMPT
    assert sc.startswith("You are an expert LLM evaluator")
    assert '{"Sufficient Context": 1}' in sc
    assert "Roald Dahl" in sc
    for ph in ("<TIMESTAMP>", "<question>", "<context>"):
        assert ph in sc


def test_the_two_judges_share_one_rubric():
    pair = PAIRS["T02-alias"]
    a = gates.se_v2_prompt(pair, "verbal")
    b = gates.se_v2_prompt(pair, "logprob")
    body = gates.SE_RUBRIC_V2.format(question=pair.question,
                                     history=gates.render_history(pair),
                                     real_action=pair.real_action,
                                     spec_action=pair.spec_action)
    assert a == body + gates.SE_RUBRIC_V2_VERBAL_TAIL
    assert b == body + gates.SE_RUBRIC_V2_LOGPROB_TAIL
    for rule in ("(a) Tool channel", "(b) Referent", "(c) Specificity",
                 "(d) Retrievable content"):
        assert rule in body


# ---------------------------------------------------------------------------
# 1. exact_sa
# ---------------------------------------------------------------------------

def test_exact_sa_is_case_insensitive_equality():
    assert run("exact_sa", "T01-exact").binary is True
    for pid in ("T02-alias", "T03-tool-channel", "T08-near-miss"):
        assert run("exact_sa", pid).binary is False, pid


def test_exact_sa_agrees_with_upstream_metrics_when_importable():
    try:
        from src.metrics import Metrics
    except Exception as exc:                      # pandas missing
        print(f"  [skip] src.metrics not importable: {exc}")
        return
    for pair in PAIRS.values():
        mine = gates._upstream_compare_action(pair.spec_action, pair.real_action)
        theirs = Metrics.compare_action(pair.spec_action, pair.real_action, False)
        assert mine == theirs, pair.pair_id
        assert run("exact_sa", pair.pair_id).detail["comparator"] == "upstream"


# ---------------------------------------------------------------------------
# 2. normalized
# ---------------------------------------------------------------------------

def test_normalized_accepts_alias_and_diacritics_and_articles():
    r = run("normalized", "T02-alias")
    assert r.binary is True
    assert r.detail["spec_arg_norm"] == "beyonce" == r.detail["real_arg_norm"]


def test_normalized_requires_the_same_tool_channel():
    assert run("normalized", "T03-tool-channel").binary is False
    assert run("normalized", "T04-terminal-channel").binary is False


def test_normalized_rejects_a_different_entity():
    assert run("normalized", "T08-near-miss").binary is False


def test_normalizer_keeps_parenthetical_disambiguators():
    assert gates.norm_arg("Mercury (planet)") == "mercury planet"
    assert gates.norm_arg("Mercury") == "mercury"
    assert gates.norm_arg("Mercury (planet)") != gates.norm_arg("Mercury")


# ---------------------------------------------------------------------------
# 3. battery
# ---------------------------------------------------------------------------

def test_battery_flags_are_rejection_reasons():
    r = run("battery", "T02-alias")
    assert r.binary is True and not r.detail["fired"]

    r = run("battery", "T03-tool-channel")
    assert r.flags["tool_channel"] is True
    assert r.flags["terminal_channel"] is False
    assert r.flags["channel"] is True
    assert r.binary is False

    r = run("battery", "T04-terminal-channel")
    assert r.flags["terminal_channel"] is True and r.flags["tool_channel"] is True

    r = run("battery", "T05-stale-intent")
    assert r.flags["stale_intent"] is True
    assert r.flags["spec_fixation"] is False
    assert r.flags["agent_loop"] is False

    r = run("battery", "T06-spec-fixation")
    assert r.flags["spec_fixation"] is True
    assert r.flags["stale_intent"] is False

    r = run("battery", "T07-agent-loop")
    assert r.flags["agent_loop"] is True
    assert r.flags["stale_intent"] is True and r.flags["spec_fixation"] is True


def test_battery_channel_alias_does_not_double_count():
    r = run("battery", "T03-tool-channel")
    assert "channel" not in r.detail["fired"]
    assert "tool_channel" in r.detail["fired"]


# ---------------------------------------------------------------------------
# 4. edit_distance
# ---------------------------------------------------------------------------

def test_edit_distance_is_a_distance_and_sweeps_the_grid():
    r = run("edit_distance", "T08-near-miss")
    # "british railways board" -> "british railway board": one deletion of 22
    assert r.score == 1 / 22
    assert all(r.decisions.values())                 # accepted at 0.1 .. 0.5
    assert r.detail["raw_edit_distance"] == 1
    assert r.detail["score_is"].startswith("distance")


def test_edit_distance_records_the_faithful_dsp_predicate():
    assert run("edit_distance", "T01-exact").detail["dsp_exact_equal"] is False
    same = Pair(pair_id="x", question="q", real_action="search[A]",
                spec_action="search[A]")
    r = gates.CRITERIA["edit_distance"].fn(same, CTX)
    assert r.detail["dsp_exact_equal"] is True and r.score == 0.0


def test_edit_distance_grid_rejects_a_far_pair():
    r = run("edit_distance", "T04-terminal-channel")
    assert r.score > 0.5
    assert not any(r.decisions.values())


def test_criterion_four_is_flagged_as_not_faithful():
    c = gates.CRITERIA["edit_distance"]
    assert c.faithful is False
    assert "openagi_utils.py:37-39" in c.source


# ---------------------------------------------------------------------------
# 5/12. embeddings
# ---------------------------------------------------------------------------

def test_embed_criteria_are_na_without_an_embedder():
    assert run("embed_call", "T02-alias").na_reason == "no embedder in context"
    assert run("embed_obs", "T11-obs-byte-identical").na_reason \
        == "no embedder in context"


def test_embed_call_scores_cosine_and_sweeps_the_grid():
    ctx = Context(embedder=StubEmbedder())
    r = run("embed_call", "T03-tool-channel", ctx)      # identical arguments
    assert abs(r.score - 1.0) < 1e-9
    assert set(r.decisions) == {"0.8", "0.85", "0.9", "0.95"}
    assert all(r.decisions.values())
    assert r.detail["same_tool"] is False               # SpecBox's precondition
    assert r.detail["embedder"] == "stub-hash-64"

    r2 = run("embed_call", "T04-terminal-channel", ctx)
    assert r2.score < 0.8 and not any(r2.decisions.values())


def test_embed_obs_is_obs_level_and_uses_the_observations():
    emb = StubEmbedder()
    ctx = Context(embedder=emb)
    r = run("embed_obs", "T11-obs-byte-identical", ctx)
    assert abs(r.score - 1.0) < 1e-9
    assert emb.calls[-1][0].startswith("The British Railways Board")


# ---------------------------------------------------------------------------
# 6/7. our judge
# ---------------------------------------------------------------------------

def test_verbal_confidence_folds_verdict_and_confidence():
    ctx = Context(judge=StubJudge(replies={"SPECULATED ACTION":
                                           "Verdict: Yes\nConfidence: 80"}))
    r = run("judge_v2_verbal", "T02-alias", ctx)
    assert r.detail["verdict"] is True and abs(r.score - 0.80) < 1e-9
    assert r.decisions["0.8"] is True and r.decisions["0.9"] is False

    ctx = Context(judge=StubJudge(default_reply="Verdict: No\nConfidence: 90"))
    r = run("judge_v2_verbal", "T02-alias", ctx)
    assert r.detail["verdict"] is False and abs(r.score - 0.10) < 1e-9


def test_verbal_parser_handles_slop_and_refuses_garbage():
    assert gates.parse_verbalized("yes, clearly")[0] is True
    assert gates.parse_verbalized("Verdict: No\nConfidence: 120")[1] == 1.0
    assert gates.parse_verbalized("Verdict: Yes\nI am 70% sure")[1] == 0.70
    ctx = Context(judge=StubJudge(default_reply="I cannot comply."))
    r = run("judge_v2_verbal", "T02-alias", ctx)
    assert r.score is None and "unparseable" in r.na_reason


def test_missing_confidence_defaults_to_one_and_is_recorded():
    ctx = Context(judge=StubJudge(default_reply="Verdict: Yes"))
    r = run("judge_v2_verbal", "T02-alias", ctx)
    assert r.score == 1.0
    assert "confidence defaulted to 1.0" in r.detail["parse_note"]


def test_logprob_judge_scores_the_margin():
    ctx = Context(judge=StubJudge(margins={"SPECULATED ACTION": 1.5}))
    r = run("judge_v2_logprob", "T02-alias", ctx)
    assert r.score == 1.5
    assert r.decisions["1"] is True and r.decisions["2"] is False


def test_margin_arithmetic_and_clipping():
    top = {"Yes": math.log(0.6), " yes": math.log(0.2), "No": math.log(0.1)}
    m, info = backends.yes_no_margin_from_top(top)
    assert abs(m - (math.log(0.8) - math.log(0.1))) < 1e-9
    assert info["clipped"] is None

    top = {"Yes": math.log(0.9), "Maybe": math.log(0.05)}
    m, info = backends.yes_no_margin_from_top(top)
    assert info["clipped"] == ["no"]
    assert abs(m - (math.log(0.9) - math.log(0.05))) < 1e-9

    m, info = backends.yes_no_margin_from_top({"Maybe": -1.0, "Perhaps": -2.0})
    assert m is None and info["clipped"] == ["yes", "no"]

    ctx = Context(judge=StubJudge())
    ctx.judge.yes_no_margin = lambda p: (None, {"reason": "no mass"})
    r = run("judge_v2_logprob", "T02-alias", ctx)
    assert r.score is None and "top-k" in r.na_reason


# ---------------------------------------------------------------------------
# 8. dualspec_critic
# ---------------------------------------------------------------------------

def test_dualspec_prompt_is_verbatim_then_our_context():
    p = gates.dualspec_prompt(PAIRS["T05-stale-intent"])
    assert p.startswith(gates.DUALSPEC_CRITIC_PROMPT)
    marker = "--- context (s_t, z_t, a_t), serialized by the auditor ---"
    assert marker in p
    head, tail = p.split(marker)
    assert head.strip() == gates.DUALSPEC_CRITIC_PROMPT.strip()
    # the critic sees the candidate action but never the real one
    assert "search[British Railways Board]" in tail
    assert "Strategic Rail Authority" not in p


def test_dualspec_critic_scores_the_log_odds_margin():
    ctx = Context(judge=StubJudge(margins={"TRAJECTORY AUDIT": -3.0}))
    r = run("dualspec_critic", "T05-stale-intent", ctx)
    assert r.score == -3.0
    assert r.decisions["-4"] is True and r.decisions["-2"] is False
    assert r.detail["judges"].startswith("trajectory progress")


# ---------------------------------------------------------------------------
# 9. spec_confidence
# ---------------------------------------------------------------------------

def test_spec_confidence_drops_the_first_name_token_like_spork():
    r = run("spec_confidence", "T09-spec-tokens")
    assert abs(r.score - 0.97) < 1e-9                 # not 0.60
    assert r.detail["dropped_first_name_token"] is True
    assert r.detail["name_span_tokens"] == ["rch"]
    assert abs(r.detail["arg_span_score"] - 0.40) < 1e-9
    assert r.decisions["0.9"] is True                 # SPORK's theta
    assert r.detail["arg_span_decisions"]["0.9"] is False


def test_spec_confidence_single_token_name_span_is_flagged():
    r = run("spec_confidence", "T10-spec-tokens-1tok")
    assert abs(r.score - 0.55) < 1e-9
    assert r.detail["single_token_name_span"] is True
    assert r.detail["dropped_first_name_token"] is False
    assert r.decisions["0.9"] is False


def test_spec_confidence_is_na_without_token_logprobs():
    r = run("spec_confidence", "T02-alias")
    assert r.score is None and "logprobs" in r.na_reason


# ---------------------------------------------------------------------------
# 10. obs_equal
# ---------------------------------------------------------------------------

def test_obs_equal_reports_both_byte_and_normalized():
    r = run("obs_equal", "T11-obs-byte-identical")
    assert r.binary is True
    assert r.detail["byte_identical"] is True

    r = run("obs_equal", "T12-obs-normalized-equal")
    assert r.binary is True                     # ours
    assert r.detail["byte_identical"] is False  # AOSpec's faithful verdict

    assert run("obs_equal", "T13-obs-paraphrase").binary is False


# ---------------------------------------------------------------------------
# 11. spechop_rules
# ---------------------------------------------------------------------------

def test_spechop_accepts_a_substring_match():
    r = run("spechop_rules", "T11-obs-byte-identical")
    assert r.binary is True and r.detail["rule"] == "substring"


def test_spechop_rejects_refusal_patterns_first():
    r = run("spechop_rules", "T14-obs-refusal")
    assert r.binary is False
    assert r.detail["refusal_pattern"] in gates.REFUSAL_PATTERNS
    assert "coverage" not in r.detail          # rejected before overlap is run


def test_spechop_requires_the_targets_multi_digit_numbers():
    r = run("spechop_rules", "T15-obs-missing-number")
    assert r.binary is False
    assert r.detail["target_numbers"] == ["1963", "2001"]
    assert r.detail["missing_numbers"] == ["1963", "2001"]


def test_spechop_short_target_needs_an_exact_token_match():
    r = run("spechop_rules", "T16-obs-short-target")
    assert r.binary is True
    assert r.detail["rule"] == "short_target_exact_token_match"
    ok, info = gates.spechop_verify("no", "Yes")
    assert ok is False and info["rule"] == "short_target_exact_token_match"


def test_spechop_lexical_overlap_uses_the_published_thresholds():
    r = run("spechop_rules", "T13-obs-paraphrase")
    assert r.detail["rule"] == "lexical_overlap"
    assert r.binary is (r.detail["coverage"] >= 0.72
                        or r.detail["jaccard"] >= 0.55)
    # a paraphrase that reorders every content word still clears coverage
    assert r.detail["coverage"] >= 0.72 and r.binary is True

    # just under 72% coverage and under 0.55 Jaccard -> reject
    ok, info = gates.spechop_verify(
        "alpha bravo charlie delta echo",
        "alpha bravo charlie delta echo foxtrot golf hotel india juliett")
    assert info["coverage"] < 0.72 and info["jaccard"] < 0.55 and ok is False


def test_spechop_stopwords_do_not_carry_the_overlap():
    # content tokens are disjoint; only stopwords are shared
    ok, info = gates.spechop_verify("the of and a in", "the of and a zurich")
    assert ok is False


# ---------------------------------------------------------------------------
# 13. sufficient_context
# ---------------------------------------------------------------------------

def test_sufficient_context_prompt_substitutes_only_placeholders():
    pair = PAIRS["T13-obs-paraphrase"]
    p = gates.sufficient_context_prompt(pair, "2019-08-01")
    assert "<TIMESTAMP>" not in p and "<question>" not in p and "<context>" not in p
    assert "Assume the queries have timestamp 2019-08-01." in p
    assert pair.question in p
    assert "[speculated step: search[BRB]]" in p
    assert "Roald Dahl" in p                      # the one-shot example survives
    assert "British Railways Board, a nationalised" in p   # obs(spec_j)
    # the real action's observation is never shown to the rater
    assert p.count("nationalised industry in the\nUnited Kingdom") <= 1


def test_sufficient_context_reads_the_last_json_verdict():
    reply = ('### EXPLANATION\nThe context names the years.\n### JSON\n'
             '{"Sufficient Context": 1}')
    assert gates.parse_sufficient_context(reply) == 1
    assert gates.parse_sufficient_context('{"Sufficient Context": 0}') == 0
    assert gates.parse_sufficient_context("no json here") is None

    ctx = Context(judge=StubJudge(default_reply=reply))
    r = run("sufficient_context", "T13-obs-paraphrase", ctx)
    assert r.binary is True

    ctx = Context(judge=StubJudge(default_reply="I refuse."))
    r = run("sufficient_context", "T13-obs-paraphrase", ctx)
    assert r.binary is None and "unparseable" in r.na_reason


# ---------------------------------------------------------------------------
# the obs-level guard
# ---------------------------------------------------------------------------

def test_no_obs_criterion_scores_an_imagined_observation():
    ctx = Context(embedder=StubEmbedder(),
                  judge=StubJudge(default_reply='{"Sufficient Context": 1}'))
    for name, c in gates.CRITERIA.items():
        if c.level != "obs":
            continue
        r = gates.CRITERIA[name].fn(PAIRS["T17-obs-imagined"], ctx)
        assert r.binary is None and r.score is None, name
        assert "imagined" in (r.na_reason or ""), (name, r.na_reason)


def test_obs_criteria_are_na_when_an_observation_is_missing():
    ctx = Context(embedder=StubEmbedder(), judge=StubJudge())
    for name in ("obs_equal", "spechop_rules", "embed_obs"):
        r = gates.CRITERIA[name].fn(PAIRS["T02-alias"], ctx)
        assert r.na_reason and (r.binary is None and r.score is None), name


def test_call_criteria_never_read_an_observation():
    """A call-level criterion must score identically with the obs stripped."""
    ctx = Context(embedder=StubEmbedder(),
                  judge=StubJudge(default_reply="Verdict: Yes\nConfidence: 70"))
    pair = PAIRS["T11-obs-byte-identical"]
    stripped = Pair.from_dict(dict(pair.to_dict(), real_obs=None, spec_obs=None,
                                   spec_obs_source="unavailable"))
    for name, c in gates.CRITERIA.items():
        if c.level != "call":
            continue
        a = gates.CRITERIA[name].fn(pair, ctx).to_dict()
        b = gates.CRITERIA[name].fn(stripped, ctx).to_dict()
        assert a == b, name


# ---------------------------------------------------------------------------
# output schema
# ---------------------------------------------------------------------------

def test_score_pair_row_schema():
    ctx = Context(embedder=StubEmbedder(),
                  judge=StubJudge(default_reply="Verdict: Yes\nConfidence: 75"),
                  code_version="deadbeef")
    row = gates.score_pair(PAIRS["T11-obs-byte-identical"], ctx)
    assert row["pair_id"] == "T11-obs-byte-identical"
    assert list(row["criteria"]) == EXPECTED_NAMES
    assert row["labels"] == PAIRS["T11-obs-byte-identical"].labels
    assert row["judge_alt"] is None          # absence is recorded, not implied
    assert row["spec_obs_source"] == "executed"
    for name, res in row["criteria"].items():
        assert res["level"] == gates.CRITERIA[name].level
        assert res["output"] == gates.CRITERIA[name].output
    json.dumps(row)                          # must be serializable


def test_judge_alt_reruns_only_the_circularity_check_criteria():
    ctx = Context(judge=StubJudge("qwen-stub", default_margin=1.0,
                                  default_reply="Verdict: Yes\nConfidence: 60"),
                  judge_alt=StubJudge("llama-stub", default_margin=-1.0,
                                      default_reply="Verdict: No\nConfidence: 60"))
    row = gates.score_pair(PAIRS["T02-alias"], ctx)
    assert row["judge_alt"]["model_id"] == "llama-stub"
    assert list(row["judge_alt"]["criteria"]) == list(gates.CIRCULARITY_CHECK_CRITERIA)
    assert row["criteria"]["judge_v2_logprob"]["score"] == 1.0
    assert row["judge_alt"]["criteria"]["judge_v2_logprob"]["score"] == -1.0


def test_score_pairs_writes_one_jsonl_row_per_pair(tmp_path=None):
    import tempfile
    ctx = Context()
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "criteria_scores.jsonl")
        rows = gates.score_pairs(list(PAIRS.values()), ctx, out_path=out)
        with open(out, encoding="utf-8") as fh:
            lines = [json.loads(x) for x in fh if x.strip()]
        assert len(lines) == len(rows) == len(PAIRS)
        assert [r["pair_id"] for r in lines] == [p.pair_id for p in PAIRS.values()]
        # appending twice appends, never rewrites
        gates.score_pairs(list(PAIRS.values())[:2], ctx, out_path=out)
        with open(out, encoding="utf-8") as fh:
            assert sum(1 for x in fh if x.strip()) == len(PAIRS) + 2


def test_manifest_records_models_and_grids():
    ctx = Context(embedder=StubEmbedder(), judge=StubJudge("qwen-stub"),
                  code_version="abc123")
    m = ctx.manifest()
    assert m["embedder"]["model_id"] == "stub-hash-64"
    assert m["judge"]["model_id"] == "qwen-stub"
    assert m["judge_alt"] is None
    assert m["grids"]["embed"] == [0.80, 0.85, 0.90, 0.95]
    assert m["code_version"] == "abc123"


# ---------------------------------------------------------------------------
# the cache contract
# ---------------------------------------------------------------------------

def test_cache_is_append_only_and_first_record_wins():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "judge_cache.jsonl")
        c = backends.AppendOnlyCache(path)
        k = backends.cache_key({"model_id": "m", "prompt": "p"})
        assert c.get(k) is None and c.misses == 1
        c.put(k, {"model_id": "m", "prompt": "p", "response": {"text": "A"}})
        assert c.get(k)["response"]["text"] == "A"
        # a second, differing record for the same key is appended, not merged
        c.put(k, {"model_id": "m", "prompt": "p", "response": {"text": "B"}})
        c.close()
        with open(path, encoding="utf-8") as fh:
            recs = [json.loads(x) for x in fh if x.strip()]
        assert len(recs) == 2
        reloaded = backends.AppendOnlyCache(path)
        assert reloaded.get(k)["response"]["text"] == "A"
        assert reloaded.collisions == [k]
        assert reloaded.stats()["records"] == 1


def test_cache_key_covers_model_prompt_and_params():
    base = {"model_id": "m", "kind": "complete", "prompt": "p",
            "params": {"temperature": 0.0}}
    k = backends.cache_key(base)
    assert k != backends.cache_key(dict(base, model_id="n"))
    assert k != backends.cache_key(dict(base, prompt="q"))
    assert k != backends.cache_key(dict(base, params={"temperature": 0.7}))
    assert k == backends.cache_key(dict(base))


def test_cache_survives_a_truncated_tail():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "c.jsonl")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"key": "k1", "response": {"text": "A"}}) + "\n")
            fh.write('{"key": "k2", "resp')          # killed mid-write
        c = backends.AppendOnlyCache(path)
        assert c.get("k1")["response"]["text"] == "A"
        assert c.get("k2") is None


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def test_text_helpers():
    assert gates.norm_text("Beyoncé's — Album!") == "beyonce s album"
    assert gates.parse_action("Search[Foo Bar]") == ("search", "Foo Bar")
    assert gates.parse_action("finish[1991]") == ("finish", "1991")
    assert gates.parse_action("garbage") == ("", "garbage")
    assert gates.levenshtein("kitten", "sitting") == 3
    assert gates.normalized_levenshtein("", "") == 0.0
    assert gates.cosine([1, 0], [0, 1]) == 0.0
    assert gates.has_multi_digit_numbers("in 1963 and 5") == ["1963"]
    assert gates.content_tokens("the board of the year") == ["board", "year"]


def test_fixture_covers_every_criterion_branch():
    assert len(PAIRS) == 17
    assert sum(1 for p in PAIRS.values() if p.spec_obs_source == "executed") == 6
    assert any(p.spec_obs_source == "speculator_imagined" for p in PAIRS.values())
    assert sum(1 for p in PAIRS.values() if p.spec_tokens) == 2
    for p in PAIRS.values():
        assert p.labels, p.pair_id


# ---------------------------------------------------------------------------
# standalone runner (pytest is not installed on this node)
# ---------------------------------------------------------------------------

def _main():
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = []
    for name, fn in tests:
        try:
            fn()
            print(f"  ok   {name}")
        except Exception as exc:                        # noqa: BLE001
            import traceback
            failed.append(name)
            print(f"  FAIL {name}: {exc.__class__.__name__}: {exc}")
            traceback.print_exc()
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_main())

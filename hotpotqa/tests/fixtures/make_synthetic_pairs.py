#!/usr/bin/env python3
"""Generate the synthetic criterion fixture (``pairs_synthetic.jsonl``).

The fixture is hand-designed, not sampled: every pair exists to pin one branch
of one criterion, and the expectations live next to the pair in
``tests/test_gates.py``.  It is committed so the tests are offline and stable;
this generator is committed so the fixture is reproducible and reviewable.

Replay labels (``S1``/``S2``/``S3``/``harmful``/``delayed``, see
``docs/PREREG_PAPER_B.md``) are *invented* here.  They exist so the output
schema and the audit join are exercised end to end; no accuracy, precision or
AUROC number may be computed from them.  Real labels come from the re-execution
audit.

Run:  python3 tests/fixtures/make_synthetic_pairs.py
"""

import json
import math
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "pairs_synthetic.jsonl")


def toks(*pairs):
    """[(token, probability), ...] -> the LLMClient.call_with_logprobs shape."""
    return [{"token": t, "logprob": math.log(p), "top": {t: math.log(p)}}
            for t, p in pairs]


BRB = ("The British Railways Board (BRB) was a nationalised industry in the "
       "United Kingdom that operated from 1963 to 2001.")
BRB_PUNCT = ("the british railways board brb was a nationalised industry in "
             "the united kingdom that operated from 1963 to 2001")
BRB_PARAPHRASE = ("British Railways Board, a nationalised United Kingdom "
                  "industry, operated 1963 to 2001 as BRB.")

PAIRS = [
    # -- call level ------------------------------------------------------
    dict(pair_id="T01-exact", question="Who designed the Eiffel Tower?",
         step_i=1, spec_index=0,
         real_action="search[Gustave Eiffel]",
         spec_action="Search[gustave eiffel]",
         history=[],
         labels={"S1": True, "S2": True, "S3": True,
                 "note": "out of population: exact match modulo case"}),

    dict(pair_id="T02-alias", question="Which label released Lemonade?",
         step_i=2, spec_index=0,
         real_action="search[Beyoncé]",
         spec_action="search[the Beyonce]",
         history=[dict(thought="I should find the artist.",
                       action="search[Lemonade (album)]",
                       obs="Lemonade is the sixth studio album by Beyoncé.")],
         labels={"S1": True, "S2": True, "S3": True, "harmful": False,
                 "delayed": False}),

    dict(pair_id="T03-tool-channel", question="When was the BRB dissolved?",
         step_i=2, spec_index=1,
         real_action="search[British Railways Board]",
         spec_action="lookup[British Railways Board]",
         history=[dict(action="search[Roald Dahl's Guide to Railway Safety]",
                       obs="Published in 1991 by the British Railways Board.")],
         labels={"S1": False, "S2": False, "S3": False, "harmful": True,
                 "delayed": True}),

    dict(pair_id="T04-terminal-channel", question="When was the BRB dissolved?",
         step_i=3, spec_index=2,
         real_action="search[British Railways Board]",
         spec_action="finish[1991]",
         history=[dict(action="search[Roald Dahl's Guide to Railway Safety]",
                       obs="Published in 1991 by the British Railways Board.")],
         labels={"S1": False, "S2": False, "S3": False, "harmful": True,
                 "delayed": False}),

    dict(pair_id="T05-stale-intent", question="Who succeeded the BRB?",
         step_i=3, spec_index=0,
         real_action="search[Strategic Rail Authority]",
         spec_action="search[British Railways Board]",
         history=[dict(action="search[British Railways Board]", obs=BRB),
                  dict(action="lookup[dissolved]", obs="(Result 1 / 1) In 2001.")],
         labels={"S1": False, "S2": True, "S3": True, "harmful": False,
                 "delayed": True}),

    dict(pair_id="T06-spec-fixation", question="Who succeeded the BRB?",
         step_i=3, spec_index=0,
         real_action="search[Strategic Rail Authority]",
         spec_action="lookup[dissolved]",
         history=[dict(action="search[British Railways Board]", obs=BRB),
                  dict(action="lookup[dissolved]", obs="(Result 1 / 1) In 2001.")],
         labels={"S1": False, "S2": False, "S3": False, "harmful": False,
                 "delayed": True}),

    dict(pair_id="T07-agent-loop", question="Who succeeded the BRB?",
         step_i=4, spec_index=0,
         real_action="search[Strategic Rail Authority]",
         spec_action="search[British Railways Board]",
         history=[dict(action="search[British Railways Board]", obs=BRB),
                  dict(action="search[Railtrack]", obs="Railtrack was a group."),
                  dict(action="search[British Railways Board]", obs=BRB)],
         labels={"S1": False, "S2": False, "S3": False, "harmful": False,
                 "delayed": True}),

    dict(pair_id="T08-near-miss", question="Which year did the BRB cease?",
         step_i=2, spec_index=0,
         real_action="search[British Railways Board]",
         spec_action="search[British Railway Board]",
         history=[dict(action="search[Roald Dahl's Guide to Railway Safety]",
                       obs="Published in 1991 by the British Railways Board.")],
         labels={"S1": True, "S2": True, "S3": True, "harmful": False,
                 "delayed": False}),

    # tool name split across two tokens: the SPORK span drops the first one, so
    # the name-span minimum is 0.97, not 0.60; the argument span minimum is 0.40.
    dict(pair_id="T09-spec-tokens", question="Which year did the BRB cease?",
         step_i=2, spec_index=0,
         real_action="search[British Railways Board]",
         spec_action="search[British Railway Board]",
         history=[],
         spec_tokens=toks(("sea", 0.60), ("rch", 0.97), ("[", 0.99),
                          ("British", 0.95), (" Railway", 0.40),
                          (" Board", 0.88), ("]", 0.99)),
         labels={"S1": True, "S2": True, "S3": True}),

    # single-token tool name: SPORK's i=2 rule would leave an empty span, so we
    # fall back to that one token (flagged in detail).
    dict(pair_id="T10-spec-tokens-1tok", question="Who is the CEO?",
         step_i=1, spec_index=0,
         real_action="search[Apple Inc.]",
         spec_action="search[Apple]",
         history=[],
         spec_tokens=toks(("search", 0.55), ("[", 0.99), ("Apple", 0.81),
                          ("]", 0.99)),
         labels={"S1": False, "S2": True, "S3": False}),

    # -- obs level -------------------------------------------------------
    dict(pair_id="T11-obs-byte-identical", question="When did the BRB cease?",
         step_i=2, spec_index=0,
         real_action="search[British Railways Board]",
         spec_action="search[British Railways Board (BRB)]",
         history=[],
         real_obs=BRB, spec_obs=BRB, spec_obs_source="executed",
         labels={"S1": True, "S2": True, "S3": True}),

    dict(pair_id="T12-obs-normalized-equal", question="When did the BRB cease?",
         step_i=2, spec_index=0,
         real_action="search[British Railways Board]",
         spec_action="search[british railways board]",
         history=[],
         real_obs=BRB, spec_obs=BRB_PUNCT, spec_obs_source="executed",
         labels={"S1": True, "S2": True, "S3": True}),

    dict(pair_id="T13-obs-paraphrase", question="When did the BRB cease?",
         step_i=2, spec_index=0,
         real_action="search[British Railways Board]",
         spec_action="search[BRB]",
         history=[],
         real_obs=BRB, spec_obs=BRB_PARAPHRASE, spec_obs_source="executed",
         labels={"S1": False, "S2": True, "S3": True}),

    dict(pair_id="T14-obs-refusal", question="When did the BRB cease?",
         step_i=2, spec_index=0,
         real_action="search[British Railways Board]",
         spec_action="search[BRB history]",
         history=[],
         real_obs=BRB,
         spec_obs="I don't know. The requested information is unavailable.",
         spec_obs_source="executed",
         labels={"S1": False, "S2": False, "S3": False, "harmful": True}),

    dict(pair_id="T15-obs-missing-number", question="When did the BRB cease?",
         step_i=2, spec_index=0,
         real_action="search[British Railways Board]",
         spec_action="search[British Railways]",
         history=[],
         real_obs=BRB,
         spec_obs=("The British Railways Board (BRB) was a nationalised "
                   "industry in the United Kingdom."),
         spec_obs_source="executed",
         labels={"S1": False, "S2": False, "S3": False}),

    dict(pair_id="T16-obs-short-target", question="Is the BRB still operating?",
         step_i=3, spec_index=0,
         real_action="lookup[status]", spec_action="lookup[Status]",
         history=[dict(action="search[British Railways Board]", obs=BRB)],
         real_obs="Yes", spec_obs="yes.", spec_obs_source="executed",
         labels={"S1": True, "S2": True, "S3": True}),

    # the guard: an imagined observation must not be scored at the obs level.
    dict(pair_id="T17-obs-imagined", question="When did the BRB cease?",
         step_i=2, spec_index=0,
         real_action="search[British Railways Board]",
         spec_action="search[BRB]",
         history=[],
         real_obs=BRB,
         spec_obs="BRB was a UK rail body that closed in 2001.",
         spec_obs_source="speculator_imagined",
         labels={"S1": False, "S2": False, "S3": False}),
]


def main():
    with open(OUT, "w", encoding="utf-8") as fh:
        for p in PAIRS:
            p.setdefault("run", "synthetic")
            fh.write(json.dumps(p, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"wrote {len(PAIRS)} pairs to {OUT}")


if __name__ == "__main__":
    main()

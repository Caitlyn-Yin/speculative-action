"""Paper B: collection, re-execution audit and replay labels.

Implements the unit, arms and labels of ``docs/PREREG_PAPER_B.md`` on top of the
existing harness, with no second copy of the ReAct loop for the *collection*
side: collection is ``runner.webthink(simulate=True)`` exactly as the isolation
gate runs it, and this module only reads what that loop recorded.

The *replay* side needs something the upstream loop cannot do -- run a forced
action prefix and then hand control back to the actor -- so :func:`replay` is a
second loop.  It is written to mirror ``HotPotQARun.webthink``'s realized branch
statement for statement (same prompt templates, same ``action_lowercase`` before
``env.step``, same ``obs.replace('\\\\n', '')``, same ``continue`` on an
unparseable actor reply, same trailing ``finish[]`` when the episode did not
terminate), because the control arm's job is to reproduce the logged trajectory
byte-for-byte.  A drift between the two loops shows up as a nondeterministic
pair, which is why the nondeterminism rate is a pre-registered go/no-go number
and not a diagnostic.

Nothing here reads a criterion, and no criterion reads a label
(``docs/CRITERIA.md`` §6).
"""

import os
from typing import Any, Dict, List, Optional

from . import constants
from . import gates
from .prompts import PromptTemplates
from .utils import Utils


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

def build_react_prompt() -> str:
    """The ReAct prompt ``runner.run()`` builds, without running anything else."""
    examples = Utils.read_json(
        os.path.join(constants.prompts_folder, constants.prompt_file)
    ).get("webthink_simple6")
    return Utils.join_prompt(
        PromptTemplates.REACT_INSTRUCTION, examples,
        PromptTemplates.PROMPT_INSTRUCTION,
    )


# ---------------------------------------------------------------------------
# Replay: forced prefix, then the actor continues
# ---------------------------------------------------------------------------

def replay(runner, idx: int, prompt: str, forced: List[Dict[str, str]],
           n: Optional[int] = None) -> Dict[str, Any]:
    """Re-execute ``forced`` then let the actor finish the episode.

    Parameters
    ----------
    forced
        ``[{"thought": str, "action": str}, ...]`` for steps ``1..len(forced)``.
        The thoughts matter: they go into the running prompt, so the actor's
        continuation sees the same history the collection run produced.  The
        treatment arm passes the logged thought of step ``i`` together with
        ``spec_j`` as its action, per the prereg ("record ``(actor's thought_i,
        spec_j, ACTUAL observation of spec_j)``").
    n
        Step cap, ``constants.n_steps_to_run`` by default; the loop runs
        ``1..n-1`` like ``webthink``.

    Returns a dict with the realized steps, ``n_steps`` (recorded steps, the
    same convention as ``len(normal_trajectory_dict["actions"])``), ``em``,
    ``f1`` and the final answer.  Observations are the environment's real ones:
    the speculator's imagined page is never used here.
    """
    n = constants.n_steps_to_run if n is None else n
    done = False
    info: Dict[str, Any] = {}
    running_prompt = prompt
    question = runner.env.reset(idx=idx)
    running_prompt += question + "\n"

    n_calls_badcalls = [0, 0]
    steps: List[Dict[str, Any]] = []
    bad_actor_replies = 0

    for i in range(1, n):
        if i <= len(forced):
            thought = forced[i - 1]["thought"]
            action = forced[i - 1]["action"]
            source = "forced"
        else:
            try:
                thought, actions = runner.generate_thought_actions(
                    i, running_prompt, n_calls_badcalls,
                    max_retries=constants.max_agent_retries,
                )
                action = actions[0]
            except ValueError:
                # webthink does exactly this: the step index is consumed and no
                # step is recorded.
                bad_actor_replies += 1
                continue
            source = "actor"

        obs, _r, done, info, dt = runner.step(
            runner.env, runner.action_lowercase(action))
        obs = obs.replace('\\n', '')
        steps.append({"i": i, "thought": thought, "action": action,
                      "obs": obs, "source": source,
                      "wall_s": round(dt, 4)})
        running_prompt += PromptTemplates.NEXT_STEP_PROMPT.format(
            i=i, thought=thought, action=action, obs=obs)
        if done:
            break

    if not done:
        _obs, _r, done, info, _dt = runner.step(runner.env, "finish[]")

    return {
        "idx": idx,
        "steps": steps,
        "n_steps": len(steps),
        "em": int(info.get("em", 0)),
        "f1": float(info.get("f1", 0.0)),
        "answer": info.get("answer"),
        "terminated": bool(done),
        "n_forced": len(forced),
        "bad_actor_replies": bad_actor_replies,
        "n_actor_calls": n_calls_badcalls[0],
    }


# ---------------------------------------------------------------------------
# Collection -> records
# ---------------------------------------------------------------------------

def episode_record(runner, idx: int, info: Dict[str, Any]) -> Dict[str, Any]:
    """Structure what one ``webthink(simulate=True)`` call left on the runner.

    The realized trajectory comes from ``normal_trajectory_dict``, the k
    speculated actions per step from ``sim_trajectory_dict`` (whose ``actions``
    entries are *lists*), and the speculator's token stream from
    ``runner.spec_token_records`` when ``constants.capture_spec_logprobs`` is
    on.  ``sim_obs`` is recorded for provenance only -- it is the speculator's
    *imagined* page and no Paper B arm or obs-level criterion may consume it.
    """
    normal = runner.env.normal_trajectory_dict
    sim = runner.env.sim_trajectory_dict
    tokens_by_step = {r["i"]: r.get("tokens") for r in runner.spec_token_records}

    steps = []
    for i, (thought, action, obs) in enumerate(
            zip(normal["thoughts"], normal["actions"], normal["observations"]), 1):
        spec_actions = sim["actions"][i - 1] if i - 1 < len(sim["actions"]) else []
        steps.append({
            "i": i,
            "thought": thought,
            "action": action,
            "obs": obs,
            "spec_thought": (sim["thoughts"][i - 1]
                             if i - 1 < len(sim["thoughts"]) else None),
            "spec_actions": list(spec_actions),
            "sim_obs": (sim["observations"][i - 1]
                        if i - 1 < len(sim["observations"]) else None),
            "spec_tokens": tokens_by_step.get(i),
        })

    # webthink's `info` carries the question only on reset, and gt_answer only
    # when the episode terminated with an answer; read both from the dataset so
    # a truncated episode still gets a labelled record.
    question = info.get("question")
    gt_answer = info.get("gt_answer")
    data = getattr(runner.env, "data", None)
    if data is not None and 0 <= idx < len(data):
        question = question or data[idx][0]
        gt_answer = gt_answer if gt_answer is not None else data[idx][1]

    return {
        "idx": idx,
        "question": question,
        "gt_answer": gt_answer,
        "steps": steps,
        "n_steps": len(steps),
        "em": int(info.get("em", 0)),
        "f1": float(info.get("f1", 0.0)),
        "answer": info.get("answer"),
        "n_calls": info.get("n_calls"),
        "n_badcalls": info.get("n_badcalls"),
        "spec_model": constants.spec_model_name,
        "actor_model": constants.actor_model_name,
        "spec_actions_from": constants.spec_actions_from,
        "retrieval_backend": constants.retrieval_backend,
    }


# ---------------------------------------------------------------------------
# Population filter and pair construction
# ---------------------------------------------------------------------------

def in_population(spec_action: str, real_action: str) -> bool:
    """The prereg's unit: ``lower(spec_j) != lower(real_i)``.

    Exactly the complement of criterion 1 (`exact_sa`), so every pair we keep is
    one Speculative Actions would have rejected.
    """
    return (spec_action or "").strip().lower() != (real_action or "").strip().lower()


def same_action(a: str, b: str) -> bool:
    """Normalized action equality: same tool AND same normalized argument.

    This is criterion 2's comparison (``gates.parse_action`` + ``gates.norm_arg``),
    reused for the S1/S2 labels so that "normalized next actions equal" in the
    prereg and "normalized" in the battery cannot drift apart.
    """
    tool_a, arg_a = gates.parse_action(a)
    tool_b, arg_b = gates.parse_action(b)
    return tool_a == tool_b and gates.norm_arg(arg_a) == gates.norm_arg(arg_b)


def pairs_from_episode(record: Dict[str, Any], run: str) -> List[Dict[str, Any]]:
    """Every in-population (step i, spec j) of one episode, unlabelled.

    The rows are ``gates.Pair``-shaped except for ``spec_obs`` / ``labels``,
    which only the re-execution audit can fill.
    """
    out = []
    history: List[Dict[str, str]] = []
    for step in record["steps"]:
        for j, spec in enumerate(step["spec_actions"]):
            if not in_population(spec, step["action"]):
                continue
            out.append({
                "pair_id": f"{run}-q{record['idx']}-s{step['i']}-j{j}",
                "run": run,
                "question": record["question"],
                "step_i": step["i"],
                "spec_index": j,
                "real_action": step["action"],
                "real_obs": step["obs"],
                "spec_action": spec,
                "spec_obs": None,
                "spec_obs_source": "unavailable",
                "spec_tokens": step.get("spec_tokens"),
                "history": [dict(t) for t in history],
                "labels": {},
                "meta": {
                    "idx": record["idx"],
                    "gt_answer": record["gt_answer"],
                    "retrieval_backend": record["retrieval_backend"],
                    "spec_actions_from": record["spec_actions_from"],
                    "spec_model": record["spec_model"],
                    "actor_model": record["actor_model"],
                    "logged_em": record["em"],
                    "logged_n_steps": record["n_steps"],
                    "sim_obs_imagined": step.get("sim_obs"),
                },
            })
        history.append({"thought": step["thought"], "action": step["action"],
                        "obs": step["obs"]})
    return out


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

def action_at(record: Dict[str, Any], i: int) -> Optional[str]:
    for step in record["steps"]:
        if step["i"] == i:
            return step["action"]
    return None


def actions_between(record: Dict[str, Any], lo: int, hi: int) -> List[str]:
    return [s["action"] for s in record["steps"] if lo <= s["i"] <= hi]


def label_pair(step_i: int, control: Dict[str, Any],
               treatment: Dict[str, Any]) -> Dict[str, Any]:
    """The prereg's five labels plus the deltas, for one pair.

    * ``S1``  normalized next actions equal -- NA if either arm ends at step i.
    * ``S2``  the control's step-(i+1) action appears among treatment steps
      i+1..i+3 -- NA on the same condition as S1.
    * ``S3``  (primary) ``em_T >= em_C and n_steps_T <= n_steps_C``.
    * ``harmful`` ``em_T < em_C``;  ``delayed`` ``n_steps_T > n_steps_C``.

    "Ends at step i" is read as: that arm recorded no step after ``i``.  S1/S2
    are then undefined rather than false -- a terminated episode has no next
    action to agree about, and scoring that as disagreement would inflate the
    rejection rate exactly where the speculation ended the episode.
    """
    c_next = action_at(control, step_i + 1)
    t_next = action_at(treatment, step_i + 1)

    if c_next is None or t_next is None:
        s1: Optional[bool] = None
        s2: Optional[bool] = None
        na = ("control ends at step i" if c_next is None
              else "treatment ends at step i")
    else:
        s1 = same_action(c_next, t_next)
        s2 = any(same_action(c_next, a)
                 for a in actions_between(treatment, step_i + 1, step_i + 3))
        na = None

    em_c, em_t = control["em"], treatment["em"]
    ns_c, ns_t = control["n_steps"], treatment["n_steps"]

    labels = {
        "S1": s1,
        "S2": s2,
        "S3": bool(em_t >= em_c and ns_t <= ns_c),
        "harmful": bool(em_t < em_c),
        "delayed": bool(ns_t > ns_c),
        "d_em": em_t - em_c,
        "d_f1": round(treatment["f1"] - control["f1"], 6),
        "d_steps": ns_t - ns_c,
        "em_control": em_c, "em_treatment": em_t,
        "f1_control": round(control["f1"], 6),
        "f1_treatment": round(treatment["f1"], 6),
        "n_steps_control": ns_c, "n_steps_treatment": ns_t,
        "answer_control": control["answer"],
        "answer_treatment": treatment["answer"],
        "next_action_control": c_next,
        "next_action_treatment": t_next,
    }
    if na:
        labels["S1_na_reason"] = labels["S2_na_reason"] = na
    return labels


# ---------------------------------------------------------------------------
# Nondeterminism control
# ---------------------------------------------------------------------------

def control_reproduces_log(record: Dict[str, Any],
                           control: Dict[str, Any]) -> Dict[str, Any]:
    """Did the control re-run reproduce the logged trajectory?

    Compared object: the ordered ``(action, obs)`` pairs, plus ``n_steps`` and
    ``em`` -- the same object the isolation gate compares
    (``scripts/run_invariant.py``).  A pair whose control fails this is
    **nondeterministic** and is excluded from primary analysis, with the rate
    reported (prereg, Arms).
    """
    logged = [(s["action"], s["obs"]) for s in record["steps"]]
    got = [(s["action"], s["obs"]) for s in control["steps"]]
    first_div = None
    for k in range(max(len(logged), len(got))):
        a = logged[k] if k < len(logged) else None
        b = got[k] if k < len(got) else None
        if a != b:
            first_div = k + 1
            break
    same = (first_div is None
            and record["n_steps"] == control["n_steps"]
            and record["em"] == control["em"])
    return {
        "deterministic": bool(same),
        "first_divergent_step": first_div,
        "n_steps_logged": record["n_steps"],
        "n_steps_control": control["n_steps"],
        "em_logged": record["em"],
        "em_control": control["em"],
    }

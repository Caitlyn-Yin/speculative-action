#!/usr/bin/env python
"""Health-check the local vLLM servers and the llm_client path into them.

  python scripts/check_server.py --all          # actor :8000 + speculator :8001
  python scripts/check_server.py --role spec    # one role
  python scripts/check_server.py --port 8001 --model Qwen/Qwen3-0.6B

Checks, per server:
  1. /v1/models answers and advertises the expected model id;
  2. a greedy completion round-trips *through* hotpotqa/src/llm_client.LLMClient
     (not raw HTTP), i.e. the code path the pipeline actually uses;
  3. no <think> leakage in the reply;
  4. determinism: the same prompt twice returns byte-identical text.

Exit code 0 = all green, 1 = something failed.
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
HOTPOTQA_DIR = os.path.join(os.path.dirname(HERE), "hotpotqa")
sys.path.insert(0, HOTPOTQA_DIR)

from src import constants                      # noqa: E402
from src import durability                     # noqa: E402
from src.llm_client import LLMClient           # noqa: E402

PROMPT = "Answer with exactly one word: what is the capital of France?"


def check(role, model, base_url):
    print(f"\n=== {role}: {model} @ {base_url} ===")
    ok = True

    import openai
    client = openai.OpenAI(base_url=base_url, api_key=constants.local_api_key)

    # 1. /v1/models
    try:
        served = [m.id for m in client.models.list().data]
        print(f"  [ok] /v1/models -> {served}")
        if model not in served:
            print(f"  [FAIL] expected {model!r} to be served")
            ok = False
    except Exception as exc:
        print(f"  [FAIL] /v1/models unreachable: {type(exc).__name__}: {exc}")
        return False

    # 2/3. greedy completion through LLMClient
    llm = LLMClient(
        model_name=model,
        temperature=0,
        max_tokens=32,
        top_p=1,
        backend="local",
        base_url=base_url,
    )
    try:
        first = llm.call(PROMPT)
    except Exception as exc:
        print(f"  [FAIL] LLMClient.call raised {type(exc).__name__}: {exc}")
        return False
    print(f"  [ok] LLMClient greedy reply: {first!r}")

    if "<think>" in first or "</think>" in first:
        print("  [FAIL] <think> leakage in reply")
        ok = False
    else:
        print("  [ok] no <think> leakage")

    # 4. determinism
    second = llm.call(PROMPT)
    if first == second:
        print("  [ok] deterministic across two identical calls")
    else:
        print(f"  [FAIL] non-deterministic:\n    1: {first!r}\n    2: {second!r}")
        ok = False

    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--role", choices=["actor", "spec"])
    ap.add_argument("--port", type=int)
    ap.add_argument("--model")
    args = ap.parse_args()

    # This script writes nothing itself, but it is the preflight that gates a
    # run, so it fails here rather than letting the run discover it later:
    # $LOG_DIR is where serve_local.sh puts the server logs.
    if os.environ.get("LOG_DIR"):
        durability.require_durable_outputs(log_dir=os.environ["LOG_DIR"])

    roles = []
    if args.all:
        roles = [
            ("actor", os.environ.get("ACTOR_MODEL", constants.actor_model_name),
             constants.actor_base_url),
            ("spec", os.environ.get("SPEC_MODEL", constants.spec_model_name),
             constants.spec_base_url),
        ]
    elif args.role == "actor":
        roles = [("actor", args.model or constants.actor_model_name,
                  constants.actor_base_url)]
    elif args.role == "spec":
        roles = [("spec", args.model or constants.spec_model_name,
                  constants.spec_base_url)]
    elif args.port:
        roles = [("custom", args.model or constants.spec_model_name,
                  f"http://127.0.0.1:{args.port}/v1")]
    else:
        ap.error("pass --all, --role, or --port")

    results = [check(r, m, u) for r, m, u in roles]
    print("\n" + ("ALL GREEN" if all(results) else "FAILURES PRESENT"))
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())

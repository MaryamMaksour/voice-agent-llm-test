"""Step 1: is Ollama reachable, are the models pulled, and can each model call a tool?

    python check_server.py                    # all models in config.py
    python check_server.py --model qwen3:8b   # only one model
"""

from __future__ import annotations

import argparse
import sys

import config
from llmtest.ollama_client import OllamaClient, OllamaError
from llmtest.tools import TOOL_SCHEMAS


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", "--models", dest="models", default=",".join(config.MODELS),
                   help="one model or several, comma-separated")
    models = [m.strip() for m in p.parse_args().models.split(",") if m.strip()]
    client = OllamaClient(config.OLLAMA_URL, timeout=config.REQUEST_TIMEOUT)
    print(f"Ollama URL : {config.OLLAMA_URL}")
    try:
        print(f"Ollama     : version {client.version()}")
        available = client.list_models()
    except OllamaError as e:
        print(f"[FAIL] {e}\n       Is Ollama running? From another machine, start it with OLLAMA_HOST=0.0.0.0")
        return 1

    ok = True
    for model in models:
        present = model in available or f"{model}:latest" in available
        if not present:
            print(f"[MISSING] {model:<22} -> run: ollama pull {model}")
            ok = False
            continue
        # tool-calling smoke test: the core skill the agent needs
        try:
            res = client.chat(model, [
                {"role": "system", "content": "You save facts with the update_lead tool. Use it now."},
                {"role": "user", "content": "My budget is two million dirhams."},
            ], tools=TOOL_SCHEMAS[:1], temperature=0, num_ctx=config.NUM_CTX)
        except OllamaError as e:
            print(f"[FAIL]    {model:<22} {e}")
            ok = False
            continue
        call = res.tool_calls[0] if res.tool_calls else None
        good = bool(call and call["name"] == "update_lead"
                    and str(call["arguments"].get("budget_aed", "")).replace(",", "").startswith("2000000"))
        status = "[ OK ]" if good else "[WARN]"
        print(f"{status}    {model:<22} tool call: {call or 'none'}  ({res.seconds:.1f}s, first call includes loading)")
    caller = config.CALLER_MODEL
    if caller not in models and caller not in available and f"{caller}:latest" not in available:
        print(f"[INFO]    caller model {caller} is not pulled: simulated mode will use the tested model "
              f"as the customer instead")
    print("\nReady." if ok else "\nFix the items above, then run this again.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

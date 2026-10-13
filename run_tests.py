"""Step 2: run every scenario against every model and write a comparison report.

    python run_tests.py                                  # all models, scripted + simulated
    python run_tests.py --models qwen3:8b --mode scripted
    python run_tests.py --only 03,08 --repeat 3          # some scenarios, simulated 3 times

Output: runs/<timestamp>/report.md (+ summary.csv, results.csv, leads CSVs and conversation logs)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime

import config
from llmtest.agent import Agent
from llmtest.callers import ScriptedCaller, SimulatedCaller
from llmtest.conversation import run_call
from llmtest.crm import CsvCRM
from llmtest.evaluate import evaluate
from llmtest.ollama_client import OllamaClient, OllamaError
from llmtest.report import summarize, write_reports
from llmtest.transcript import Transcript


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", default=",".join(config.MODELS), help="comma-separated Ollama model tags")
    p.add_argument("--mode", choices=["scripted", "simulated", "both"], default="both")
    p.add_argument("--only", default="", help="run only scenarios whose id starts with these, e.g. 01,03")
    p.add_argument("--repeat", type=int, default=1, help="repeat simulated calls (the caller varies each time)")
    p.add_argument("--caller-model", default=config.CALLER_MODEL)
    args = p.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    modes = ["scripted", "simulated"] if args.mode == "both" else [args.mode]
    scenarios = json.load(open(config.SCENARIOS_FILE, encoding="utf-8"))
    if args.only:
        prefixes = tuple(x.strip() for x in args.only.split(","))
        scenarios = [s for s in scenarios if s["id"].startswith(prefixes)]
    now = datetime.fromisoformat(config.SIMULATED_NOW)

    client = OllamaClient(config.OLLAMA_URL, timeout=config.REQUEST_TIMEOUT)
    try:
        available = client.list_models()
    except OllamaError as e:
        print(f"ERROR: {e}")
        return 1
    have = lambda m: m in available or f"{m}:latest" in available  # noqa: E731
    missing = [m for m in models if not have(m)]
    if missing:
        print("Missing models, pull them first:\n" + "\n".join(f"  ollama pull {m}" for m in missing))
        return 1
    if "simulated" in modes and not have(args.caller_model):
        print(f"Caller model {args.caller_model} is not pulled; running scripted mode only "
              f"(or: ollama pull {args.caller_model})")
        modes = [m for m in modes if m != "simulated"]
        if not modes:
            return 1

    run_dir = os.path.join(config.RUNS_DIR, datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    print(f"Run folder: {run_dir}\nModels: {models}  Modes: {modes}  Scenarios: {len(scenarios)}\n")

    results = []
    for model in models:
        for mode in modes:
            reps = args.repeat if mode == "simulated" else 1
            for rep in range(1, reps + 1):
                tag = f"{mode}" + (f"_r{rep}" if reps > 1 else "")
                folder = os.path.join(run_dir, slug(model), tag)
                crm = CsvCRM.fresh_copy(config.LEADS_SEED, os.path.join(folder, "leads.csv"))
                for sc in scenarios:
                    agent = Agent(client, model, crm, sc["lead_id"], now,
                                  company=config.COMPANY_NAME, agent_name=config.AGENT_NAME,
                                  currency=config.CURRENCY, temperature=config.AGENT_TEMPERATURE,
                                  num_ctx=config.NUM_CTX, max_tool_rounds=config.MAX_TOOL_ROUNDS)
                    caller = (ScriptedCaller(sc["script"]) if mode == "scripted" else
                              SimulatedCaller(client, args.caller_model, sc["persona"],
                                              config.CALLER_TEMPERATURE, config.NUM_CTX))
                    log = Transcript(os.path.join(folder, "logs"), sc["id"], {
                        "model": model, "mode": tag, "scenario": sc["id"], "tests": sc["what_it_tests"],
                        "lead": f"{sc['lead_id']} {crm.get(sc['lead_id'])['name']}",
                        "simulated_now": config.SIMULATED_NOW})
                    call = run_call(agent, caller, log, config.MAX_TURNS)
                    final_row = crm.get(sc["lead_id"])
                    ev = evaluate(sc, final_row, log.events, call.ended_by)
                    log.close({"passed": ev["passed"], "ended_by": call.ended_by,
                               "final_crm_row": json.dumps(final_row, ensure_ascii=False),
                               "wrong_fields": [f["field"] for f in ev["fields"] if not f["ok"]]})
                    results.append({"model": model, "mode": tag, "scenario": sc["id"], "evaluation": ev,
                                    "ended_by": call.ended_by, "agent_turns": call.agent_turns,
                                    "model_seconds": call.model_seconds, "log": log.txt_path})
                    wrong = [f["field"] for f in ev["fields"] if not f["ok"]]
                    print(f"{'PASS' if ev['passed'] else 'FAIL'}  {model:<18} {tag:<12} {sc['id']:<26} "
                          f"{('wrong: ' + ', '.join(wrong)) if wrong else ''}"
                          f"{'' if call.ended_by == 'end_call' else '  ended_by=' + call.ended_by}")

    summary = summarize(results, config.TARGET_FIELD_ACCURACY, config.TARGET_SCENARIO_PASS_RATE)
    report = write_reports(run_dir, results, summary, {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"), "ollama": config.OLLAMA_URL,
        "modes": ", ".join(modes), "caller model": args.caller_model, "scenarios": len(scenarios)})
    print("\n" + "=" * 90)
    for s in summary:
        print(f"{s['model']:<20} accuracy {s['field_accuracy']:.1%}  passed {s['scenarios_passed']:<7} "
              f"ended properly {s['ended_properly']}/{s['calls']}  -> {s['verdict']}")
    print(f"\nFull report: {report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

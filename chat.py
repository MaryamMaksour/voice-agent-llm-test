"""Talk to the agent yourself: you play the lead, the agent updates a real CSV.

    python chat.py --model qwen3:8b --lead L013
Type /end to hang up. The CSV is data/leads_live.csv; the log is saved under runs/chat/.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime

import config
from llmtest.agent import Agent
from llmtest.callers import HumanCaller
from llmtest.conversation import run_call
from llmtest.crm import CsvCRM
from llmtest.ollama_client import OllamaClient
from llmtest.postcall import post_call_check
from llmtest.transcript import Transcript

LIVE_CSV = "data/leads_live.csv"


class PrintingTranscript(Transcript):
    """Also shows the agent's replies and tool calls in the terminal."""

    def agent(self, text: str, seconds: float) -> None:
        super().agent(text, seconds)
        print(f"agent > {text}   ({seconds:.1f}s)")

    def tool(self, name: str, args: dict, result: str) -> None:
        super().tool(name, args, result)
        print(f"        [{name} {json.dumps(args, ensure_ascii=False)} -> {result}]")

    def note(self, text: str) -> None:
        super().note(text)
        print(f"        [{text}]")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=config.MODELS[0])
    p.add_argument("--lead", default="L013")
    p.add_argument("--reset", action="store_true", help="start again from the original leads.csv")
    args = p.parse_args()

    if args.reset or not os.path.exists(LIVE_CSV):
        shutil.copyfile(config.LEADS_SEED, LIVE_CSV)
    crm = CsvCRM(LIVE_CSV)
    client = OllamaClient(config.OLLAMA_URL, timeout=config.REQUEST_TIMEOUT)
    now = datetime.now().replace(second=0, microsecond=0)
    agent = Agent(client, args.model, crm, args.lead, now, company=config.COMPANY_NAME,
                  agent_name=config.AGENT_NAME, currency=config.CURRENCY,
                  temperature=config.AGENT_TEMPERATURE, num_ctx=config.NUM_CTX,
                  max_tool_rounds=config.MAX_TOOL_ROUNDS)
    lead = crm.get(args.lead)
    print(f"Calling {lead['name']} ({args.lead}) about {lead['project_interest']} with {args.model}.")
    print("You are the lead. Start with 'Hello?'. Type /end to hang up.\n")
    name = f"{datetime.now():%Y%m%d-%H%M%S}_{args.model.replace(':', '_')}_{args.lead}"
    log = PrintingTranscript(os.path.join(config.RUNS_DIR, "chat"), name,
                             {"model": args.model, "lead": f"{args.lead} {lead['name']}", "now": now.isoformat()})
    call = run_call(agent, HumanCaller(), log, max_turns=50)
    if config.POST_CALL_CHECK and call.ended_by != "error":
        _, line = post_call_check(client, args.model, crm, args.lead, log.events, num_ctx=config.NUM_CTX)
        log.note(line)
    row = crm.get(args.lead)
    log.close({"ended_by": call.ended_by, "final_crm_row": json.dumps(row, ensure_ascii=False)})
    print("\nCRM row now:")
    for k in ("status", "budget_aed", "preferred_area", "property_type", "timeline", "interest_level",
              "callback_at", "do_not_call", "notes"):
        print(f"  {k:<15} {row.get(k)}")
    print(f"\nLog: {log.txt_path}")


if __name__ == "__main__":
    main()

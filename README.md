# Voice Agent LLM Test

Answers one question: **which Qwen3 model (8B, 14B or 30B-A3B) is good enough to run our real-estate phone agent?**

Each model has text conversations with leads, the same as phone calls but without the audio. It updates a **CSV of leads** through tools, and every conversation is **logged**. Each model gets a score and a verdict: **GOOD ENOUGH / NOT ENOUGH**.

- Runs on the company server with **Ollama only**: no Docker, and **no Python packages to install** (standard library only).
- Tests **quality**, not speed or many calls at once.

---

## 1. System design

```
                         ┌──────────────────────── run_tests.py (orchestrator) ───────────────────────┐
                         │  for each MODEL × MODE × SCENARIO:                                         │
 scenarios/              │                                                                            │
 scenarios.json ───────► │   Caller ───── lead's words ─────►  Agent  ────── tool calls ────►  Tools  │
 (12 test calls,         │  (scripted lines,                 (model under test,             (validate,│
  expected results)      │   or LLM role-play,               system prompt,                  bound to │
                         │   or you in chat.py)              tool loop)                      one lead)│
 data/leads.csv ───────► │      ▲                                │                               │    │
 (seed CRM)              │      └────── agent's reply ───────────┘                               ▼    │
                         │                                                             leads.csv (copy│
                         │   Transcript ◄── every line + tool call + result           per model/mode) │
                         │   (logs/*.txt + *.jsonl)                                                   │
                         │                                                                            │
                         │   Evaluator: final CSV row vs expected row  +  behaviour checks            │
                         └──────────────────────────────┬─────────────────────────────────────────────┘
                                                        ▼
                                  report.md · summary.csv · results.csv   (runs/<timestamp>/)

        All model calls go to Ollama:  POST /api/chat  (tools, think=false)   ← on the server
```

| Component | File | Responsibility |
|---|---|---|
| Config | `config.py` | Ollama URL, the 3 models, caller model, fixed "now", targets |
| Ollama client | `llmtest/ollama_client.py` | `/api/chat` with tools and thinking off; removes `<think>` text; also reads tool calls the model writes as text; retries without the think flag on old Ollama versions |
| CRM | `llmtest/crm.py` | Leads CSV: validated updates, atomic saves (never a half-written file), one row per lead |
| Tools | `llmtest/tools.py` | `update_lead`, `schedule_callback`, `mark_do_not_call`, `end_call`. Bound to **one lead**, so the model can't touch other rows. Invalid values return a message the model can fix |
| Business rules | `llmtest/rules.py` | Decisions taken in code, not by the model: lead status, interest level, callback date, vague areas, price guard (see below) |
| Post-call check | `llmtest/postcall.py` | After each call, reads the conversation once more and fills in details that were said but not saved |
| Prompts | `llmtest/prompts.py` | Agent system prompt (the product) and caller persona prompt (the test driver) |
| Agent | `llmtest/agent.py` | One model + one lead: message history, tool-calling loop |
| Callers | `llmtest/callers.py` | `ScriptedCaller` (fixed lines), `SimulatedCaller` (an LLM plays the lead), `HumanCaller` (you) |
| Call runner | `llmtest/conversation.py` | Alternates lead and agent until `end_call`, hang-up, or max turns; never crashes the run on a model error |
| Logs | `llmtest/transcript.py` | Per call: readable `.txt` and machine-readable `.jsonl` |
| Evaluator | `llmtest/evaluate.py` | Compares the final CSV row with the expected row, field by field, plus behaviour checks |
| Report | `llmtest/report.py` | Per-model summary, verdict, list of failures with links to the logs |

### What happens in one test call
1. A **fresh copy** of `data/leads.csv` is made for the model and mode, so every model starts from the same data.
2. The agent gets a system prompt containing the lead's CRM row and today's date.
3. The **caller** says "Hello?". The agent replies, and calls tools when it learns facts.
4. Every tool call is **validated** and written to the CSV immediately. An invalid value (for example `property_type="castle"`) is not saved; the model gets an error message and can try again.
5. The call ends when the agent calls `end_call` (which saves a summary into `notes`), when the lead hangs up (the agent gets one last turn), or after `MAX_TURNS` replies.
6. The **post-call check** fills any empty detail the lead clearly gave (it never overwrites what the agent saved).
7. The **final CSV row** is compared with the scenario's expected result, and everything is logged.

### Business rules in code
The first `qwen3:8b` run showed the model understands the lead well but is unreliable at some decisions, so the code takes them over. The production agent works the same way.

| Rule | What the code does | Fixes (first 8B run) |
|---|---|---|
| Status | `qualified` when budget, area, type and timeline are all saved; `interested` when some are. Final outcomes (not interested, callback, do-not-call, wrong number) are never changed | 01, 06, 07, 09, 11, 12 |
| Interest level | Worked out from the timeline: within 3 months = hot, 3 to 12 months = warm, more than 12 months or unsure = cold | 09, 12 |
| Callback date | The model passes the day as said ("next monday", "tomorrow") and the time; the code works out the exact date | 08 |
| Area | "Anywhere", "I don't mind" and similar are never saved as an area | 11 |
| Price guard | A spoken sentence with an amount of money the lead never said is replaced by "A property consultant will share the exact prices and details with you." | 11 |
| Post-call check | One more model pass over the conversation fills details that were said but not saved | 12 |

The report shows how often the code had to step in (**Price guard** and **Post-call fills** columns), and every log shows it as `[note]` lines. To measure what the model saves on its own, run with `--no-post-call`.

### Two test modes
| Mode | The lead is played by | Why |
|---|---|---|
| `scripted` | Fixed lines from `scenarios.json` | Identical input for every model, so the comparison is fair and repeatable |
| `simulated` | An LLM (`CALLER_MODEL`, fixed) playing a persona | Realistic: answers the questions the agent actually asks. Use `--repeat 3` to see consistency |

### Prompt design (why it should be reliable)
- **Explicit call flow** with a branch for every situation: wrong number, someone else answers, busy, not interested, do-not-call, qualified, partial info.
- **Exact data rules:**
  - Budget in whole dirhams; for a range, save the upper limit.
  - The allowed values for property type and interest level are listed.
  - "Anywhere" means no area is saved.
  - Callback date format, with a worked example.
  - Default times: evening = 18:00, morning = 10:00, afternoon = 15:00.
- **"Save only what the lead actually said"**, to stop invented data.
- **Phone style:** one or two sentences, at most one question, no prices invented, never mention tools.
- **Thinking off:** `think=false` in the request, plus `/no_think` for Qwen3.
- **The same rules are enforced in code:** the tool schemas list the allowed values, and the CRM validates every change.

---

## 2. The 12 test scenarios (`scenarios/scenarios.json`)
| # | Scenario | What it checks |
|---|---|---|
| 01 | Qualified, full info | Collects budget, area, type, timeline → `qualified`, `hot` |
| 02 | Not interested | `not_interested`, nothing else saved |
| 03 | "Call me tomorrow at four" | Callback date and time worked out correctly (2026-10-15T16:00) |
| 04 | "Stop calling me" | `do_not_call` |
| 05 | Wrong number | `wrong_number`, no data invented |
| 06 | Won't give a budget | `interested` (not qualified), no invented budget |
| 07 | Budget range 5–6 million | Saves the upper limit, 6,000,000 |
| 08 | "Next Monday at ten" | Callback 2026-10-19T10:00 |
| 09 | All details but buying in 2 years | `qualified` + `cold` |
| 10 | His wife answers | Callback only, no property talk |
| 11 | Asks for prices; "anywhere" for area | Must not invent a price; must not save an area |
| 12 | Everything in one sentence | Saves all of it from a single answer |

"Now" is fixed at **Wednesday 2026-10-14 10:00**, so "tomorrow at four" always has one correct answer.

**Add your own cases:** copy a block in `scenarios.json`, point it at a lead ID from `data/leads.csv`, and write the lines and the expected row. Real-life phrasing from your own calls makes the test much more trustworthy.

## 3. How a model is scored
**Field accuracy:** 8 fields per call are compared with the expected result. A value the lead never gave counts as **wrong**.
- `status`, `budget_aed`, `property_type`, `interest_level`, `do_not_call`: must match exactly.
- `preferred_area`: must contain an accepted spelling (for example "Marina", or "JVC" / "Jumeirah Village Circle").
- `timeline`: must be saved when the lead gave one.
- `callback_at`: must be the exact date and hour.

**Behaviour checks:**
- The call ended with `end_call`.
- Replies are short (≤ 40 words).
- At most one question per reply.
- No price invented after a price question.
- The number of rejected tool calls is counted.

**A call passes** when all 8 fields are right, it ended with `end_call`, and no price was invented.

**Verdict: GOOD ENOUGH** = field accuracy **≥ 95%** and **≥ 90%** of calls passed.

---

## 4. Minimal setup on the server
Only Ollama is needed on the server. **Python 3.9+** is needed wherever you run the tests: on the server, or on your laptop.

**Linux**
```bash
curl -fsSL https://ollama.com/install.sh | sh
sudo systemctl edit ollama        # add the two lines below, save, then:
#   [Service]
#   Environment="OLLAMA_HOST=0.0.0.0:11434"
sudo systemctl restart ollama
ollama pull qwen3:8b && ollama pull qwen3:14b && ollama pull qwen3:30b-a3b
```
**Windows server:** install Ollama from ollama.com. In *System → Environment Variables* add `OLLAMA_HOST = 0.0.0.0:11434`, restart Ollama, then run the same `ollama pull` commands.

`OLLAMA_HOST` is only needed if you run the tests from your laptop. If you run them on the server, skip it.

## 5. Run the tests
Get the code onto the server, or keep it on your laptop:
- `git clone` the repo, or
- copy the folder with ManageEngine File Transfer.

Nothing to install: there is no `pip install`.

```bash
# on the server:
python check_server.py
# from your laptop instead (Windows PowerShell: $env:OLLAMA_URL="http://<server-ip>:11434"):
OLLAMA_URL=http://<server-ip>:11434 python check_server.py
```
`check_server.py` confirms that Ollama is reachable, that every model is pulled, and that **each model can make a tool call**. To check one model only: `python check_server.py --model qwen3:8b`.

**Testing just one model** (for example, only `qwen3:8b` is pulled): `python run_tests.py --model qwen3:8b`. In simulated mode the customer is normally played by `qwen3:14b`. If that model isn't pulled, the tested model plays the customer too, so nothing is skipped. For a fixed customer across models, pull `qwen3:14b` as well.

```bash
python run_tests.py                          # 3 models × (scripted + simulated) × 12 scenarios
python run_tests.py --model qwen3:8b         # only one model
python run_tests.py --models qwen3:8b,qwen3:14b   # a chosen set of models
python run_tests.py --mode scripted          # fast, fully repeatable
python run_tests.py --only 03,08,10          # only the callback scenarios
python run_tests.py --mode simulated --repeat 3   # consistency check
python run_tests.py --model qwen3:8b --no-post-call   # the model alone, without the post-call check
```
On slow hardware a full run takes a while. That's fine, because this test is about quality, not speed.

**Talk to it yourself** (you play the lead; it updates `data/leads_live.csv`):
```bash
python chat.py --model qwen3:8b --lead L013
python chat.py --model qwen3:14b --lead L014 --reset   # --reset = start from the original CSV
```

## 6. Output
```
runs/20261014-101500/
├── report.md            ← read this first: summary table, verdict, every failure with a link to its log
├── summary.csv          ← one row per model
├── results.csv          ← one row per call: passed?, wrong fields, how it ended, log path
└── qwen3_8b/
    ├── scripted/
    │   ├── leads.csv    ← the CRM after this model's calls (compare with data/leads.csv)
    │   └── logs/
    │       ├── 01_qualified_full.txt     ← readable conversation + tool calls + final row
    │       └── 01_qualified_full.jsonl   ← same, machine-readable
    └── simulated/ ...
```

## 7. Reading the result
| Result | Decision |
|---|---|
| `qwen3:8b` = GOOD ENOUGH | Use 8B (fastest, cheapest) |
| 8B not enough, 14B GOOD ENOUGH | Use 14B |
| Both not enough | Use 30B-A3B, or improve the prompt using the failure logs |
| 8B just below target | Also try `qwen3:8b-q8_0`: Ollama's default `qwen3:8b` is 4-bit; 8-bit is a bit more accurate |

Always open a few logs, not just the numbers. They show *why* a model failed: a wrong date, an invented value, forgetting `end_call`.

## 8. Settings (`config.py`, or environment variables)
| Variable | Default | Meaning |
|---|---|---|
| `OLLAMA_URL` | `http://localhost:11434` | Where Ollama runs |
| `MODELS` | `qwen3:8b,qwen3:14b,qwen3:30b-a3b` | Models under test |
| `CALLER_MODEL` | `qwen3:14b` | Plays the lead in simulated mode (keep it fixed) |
| `SIMULATED_NOW` | `2026-10-14T10:00` | The test's "now" |
| `COMPANY_NAME` / `AGENT_NAME` | Bayari Investments / Sara | Used in the prompt |
| `AGENT_TEMPERATURE` | `0.2` | Low = consistent agent |
| `MAX_TURNS` | `16` | Agent replies before a call is stopped |
| `REQUEST_TIMEOUT` | `600` s | Generous for slow servers |
| `POST_CALL_CHECK` | `1` | `0` turns the post-call check off |

## 9. Self-tests (no server needed)
```bash
python -m unittest discover -s tests -t .
```
These run the whole pipeline against a **fake Ollama** server: tool calls, CSV updates, logs and the report. They check the harness itself, not model quality.

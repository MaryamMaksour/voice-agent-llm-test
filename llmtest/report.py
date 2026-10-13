"""Aggregates all calls into a per-model comparison: report.md, summary.csv, results.csv."""

from __future__ import annotations

import csv
import os
from collections import defaultdict


def summarize(results: list[dict], target_acc: float, target_pass: float) -> list[dict]:
    by_model: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_model[r["model"]].append(r)
    summary = []
    for model, rs in by_model.items():
        fields = [f for r in rs for f in r["evaluation"]["fields"]]
        acc = sum(f["ok"] for f in fields) / max(len(fields), 1)
        passed = sum(r["evaluation"]["passed"] for r in rs)
        pass_rate = passed / max(len(rs), 1)
        conduct = [r["evaluation"]["conduct"] for r in rs]
        replies = sum(r["agent_turns"] for r in rs)
        summary.append({
            "model": model,
            "calls": len(rs),
            "field_accuracy": round(acc, 3),
            "scenarios_passed": f"{passed}/{len(rs)}",
            "pass_rate": round(pass_rate, 3),
            "ended_properly": sum(c["ended_with_end_call"] for c in conduct),
            "short_replies": round(sum(c["short_replies_share"] for c in conduct) / len(conduct), 2),
            "one_question": round(sum(c["one_question_share"] for c in conduct) / len(conduct), 2),
            "invented_money": sum(len(c["invented_money_lines"]) for c in conduct),
            "tool_errors": sum(c["tool_errors"] for c in conduct),
            "errors": sum(1 for r in rs if r["ended_by"] == "error"),
            "avg_seconds_per_reply": round(sum(r["model_seconds"] for r in rs) / max(replies, 1), 2),
            "verdict": "GOOD ENOUGH" if acc >= target_acc and pass_rate >= target_pass else "NOT ENOUGH",
        })
    return summary


def write_reports(run_dir: str, results: list[dict], summary: list[dict], meta: dict) -> str:
    with open(os.path.join(run_dir, "summary.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    with open(os.path.join(run_dir, "results.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "mode", "scenario", "passed", "wrong_fields", "ended_by", "agent_turns",
                    "avg_seconds_per_reply", "log"])
        for r in results:
            wrong = "; ".join(f"{x['field']}: expected {x['expected']!r} got {x['got']!r}"
                              for x in r["evaluation"]["fields"] if not x["ok"])
            w.writerow([r["model"], r["mode"], r["scenario"], r["evaluation"]["passed"], wrong, r["ended_by"],
                        r["agent_turns"], round(r["model_seconds"] / max(r["agent_turns"], 1), 2), r["log"]])

    lines = [f"# LLM test report", "", *(f"- **{k}:** {v}" for k, v in meta.items()), "",
             "## Summary", "",
             "| Model | Calls | Field accuracy | Scenarios passed | Ended properly | Short replies | One question | "
             "Invented money | Tool errors | Sec/reply | Verdict |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in summary:
        lines.append(f"| {s['model']} | {s['calls']} | {s['field_accuracy']:.1%} | {s['scenarios_passed']} | "
                     f"{s['ended_properly']}/{s['calls']} | {s['short_replies']:.0%} | {s['one_question']:.0%} | "
                     f"{s['invented_money']} | {s['tool_errors']} | {s['avg_seconds_per_reply']} | **{s['verdict']}** |")
    lines += ["", "Targets: field accuracy ≥ 95% and ≥ 90% of scenarios passed. "
              "Sec/reply depends on the hardware and is shown for information only.", "",
              "## Failed scenarios", ""]
    for r in results:
        if not r["evaluation"]["passed"]:
            wrong = [f"{x['field']} (expected {x['expected']!r}, got {x['got']!r})"
                     for x in r["evaluation"]["fields"] if not x["ok"]]
            c = r["evaluation"]["conduct"]
            extra = []
            if not c["ended_with_end_call"]:
                extra.append(f"call ended by: {r['ended_by']}")
            if c["invented_money_lines"]:
                extra.append(f"invented money: {c['invented_money_lines'][0]!r}")
            lines.append(f"- **{r['model']} / {r['mode']} / {r['scenario']}**: {', '.join(wrong + extra)} "
                         f"— log: `{r['log']}`")
    path = os.path.join(run_dir, "report.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path

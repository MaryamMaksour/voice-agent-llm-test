"""The extra data each call log keeps for the dashboard: timings, tool results, thinking, classification."""

import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

from llmtest.ollama_client import OllamaClient, thinking_text
from llmtest.outcome import classify_call
from tests.fake_ollama import FakeOllama, good_agent_policy, tool_call

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class ClientTimings(unittest.TestCase):
    def test_timings_tokens_and_thinking(self):
        policy = lambda body: {"content": "<think>budget is missing</think>What is your budget?"}  # noqa: E731
        with FakeOllama(policy) as fake:
            res = OllamaClient(fake.url).chat("qwen3:8b", [{"role": "user", "content": "hi"}])
        self.assertEqual(res.content, "What is your budget?")
        self.assertEqual(res.thinking, "budget is missing")
        self.assertEqual((res.prompt_tokens, res.output_tokens), (10, 5))
        self.assertEqual((res.load_seconds, res.prompt_seconds, res.generate_seconds), (0.002, 0.03, 0.05))
        self.assertFalse(res.tool_calls_as_text)

    def test_tool_call_written_as_text_is_flagged(self):
        policy = lambda body: {"content": '<tool_call>{"name": "end_call", "arguments": {}}</tool_call>'}  # noqa: E731
        with FakeOllama(policy) as fake:
            res = OllamaClient(fake.url).chat("qwen3:8b", [{"role": "user", "content": "bye"}])
        self.assertEqual(res.tool_calls[0]["name"], "end_call")
        self.assertTrue(res.tool_calls_as_text)

    def test_real_tool_call_is_not_flagged(self):
        policy = lambda body: {"tool_calls": [tool_call("end_call")]}  # noqa: E731
        with FakeOllama(policy) as fake:
            res = OllamaClient(fake.url).chat("qwen3:8b", [{"role": "user", "content": "bye"}])
        self.assertFalse(res.tool_calls_as_text)

    def test_thinking_text(self):
        self.assertEqual(thinking_text("a<think> x </think>b<think>y</think>"), "x\ny")
        self.assertEqual(thinking_text("no thinking"), "")


class Classification(unittest.TestCase):
    def test_classify(self):
        c = classify_call({"status": "qualified"}, "end_call")
        self.assertEqual(c, {"result": "qualified", "classification": "Qualified", "agent_closed": True,
                             "ending": "Agent closed the call"})
        c = classify_call({"status": "new"}, "max_turns")
        self.assertEqual((c["classification"], c["agent_closed"]), ("No result saved", False))
        self.assertEqual(classify_call({}, "caller_hung_up")["ending"], "Lead hung up first")


class CallLogFields(unittest.TestCase):
    def test_log_has_tracking_fields(self):
        runs = tempfile.mkdtemp()
        try:
            with FakeOllama(good_agent_policy, models=("qwen3:8b",)) as fake:
                env = os.environ | {"OLLAMA_URL": fake.url, "RUNS_DIR": runs}
                out = subprocess.run([sys.executable, "run_tests.py", "--model", "qwen3:8b", "--mode", "scripted",
                                      "--only", "01"], cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
            self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
            path = glob.glob(os.path.join(runs, "*", "qwen3_8b", "scripted", "logs", "01_qualified_full.jsonl"))[0]
            with open(path, encoding="utf-8") as f:
                events = [json.loads(line) for line in f]
            header, footer = events[0], events[-1]
            self.assertEqual(len(header["call_id"]), 12)
            self.assertEqual(len(header["prompt_version"]), 10)
            self.assertIn("temperature", header["settings"])
            self.assertRegex(header["time"], r"[+-]\d\d:\d\d$")  # local time with its UTC offset

            agent = [e for e in events if e["type"] == "agent"]
            self.assertTrue(all("reply_seconds" in e and e["rounds"] for e in agent))
            first_round = agent[0]["rounds"][0]
            for key in ("seconds", "load_seconds", "prompt_seconds", "generate_seconds", "prompt_tokens",
                        "output_tokens", "tool_calls", "tool_calls_as_text", "thinking_chars"):
                self.assertIn(key, first_round)
            tools = [e for e in events if e["type"] == "tool"]
            self.assertTrue(tools and all("seconds" in t and "ok" in t for t in tools))

            self.assertEqual(footer["classification"], "Qualified")
            self.assertTrue(footer["agent_closed"])
            self.assertEqual(footer["result"], "qualified")
            self.assertGreaterEqual(footer["duration_seconds"], 0)
        finally:
            shutil.rmtree(runs)


if __name__ == "__main__":
    unittest.main()

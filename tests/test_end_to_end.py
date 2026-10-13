"""Runs the real run_tests.py pipeline against the fake Ollama: calls, CSV, logs, report."""

import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

from tests.fake_ollama import FakeOllama, good_agent_policy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


class EndToEnd(unittest.TestCase):
    def test_full_run_writes_csv_logs_and_report(self):
        runs = tempfile.mkdtemp()
        try:
            with FakeOllama(good_agent_policy, models=("qwen3:8b", "qwen3:14b")) as fake:
                env = os.environ | {"OLLAMA_URL": fake.url, "RUNS_DIR": runs}
                out = subprocess.run(
                    [sys.executable, "run_tests.py", "--models", "qwen3:8b", "--mode", "both", "--only", "01,04"],
                    cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
            self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
            run_dir = glob.glob(os.path.join(runs, "*"))[0]

            # scripted scenario 01: the fake agent saves all four details and qualifies the lead
            leads = _read(os.path.join(run_dir, "qwen3_8b", "scripted", "leads.csv"))
            self.assertIn("L001,Sarah Johnson,+971500000001,Marina Vista Residences,qualified,2000000,Dubai Marina,"
                          "apartment,within two months,hot", leads)
            self.assertIn("L004,James Wilson,+971500000004,Marina Vista Residences,do_not_call", leads)
            self.assertIn("PASS  qwen3:8b           scripted     01_qualified_full", out.stdout)
            self.assertIn("PASS  qwen3:8b           scripted     04_do_not_call", out.stdout)

            # logs: readable text and machine-readable JSONL for every call
            log_txt = _read(os.path.join(run_dir, "qwen3_8b", "scripted", "logs", "01_qualified_full.txt"))
            self.assertIn("LEAD : Around two million dirhams.", log_txt)
            self.assertIn("[tool] update_lead", log_txt)
            events = [json.loads(line) for line in _read(os.path.join(
                run_dir, "qwen3_8b", "scripted", "logs", "01_qualified_full.jsonl")).splitlines()]
            self.assertEqual(events[0]["type"], "header")
            self.assertEqual(events[-1]["type"], "footer")

            report = _read(os.path.join(run_dir, "report.md"))
            self.assertIn("| qwen3:8b |", report)
            self.assertTrue(os.path.exists(os.path.join(run_dir, "summary.csv")))
            self.assertTrue(os.path.exists(os.path.join(run_dir, "results.csv")))
        finally:
            shutil.rmtree(runs)

    def test_single_model_with_only_that_model_pulled(self):
        """python run_tests.py --model qwen3:8b, when the default caller model (qwen3:14b) is not pulled."""
        runs = tempfile.mkdtemp()
        try:
            with FakeOllama(good_agent_policy, models=("qwen3:8b",)) as fake:
                env = os.environ | {"OLLAMA_URL": fake.url, "RUNS_DIR": runs}
                out = subprocess.run([sys.executable, "run_tests.py", "--model", "qwen3:8b", "--only", "01"],
                                     cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
                check = subprocess.run([sys.executable, "check_server.py", "--model", "qwen3:8b"],
                                       cwd=ROOT, env=env, capture_output=True, text=True, timeout=60)
            self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
            self.assertIn("qwen3:8b will play the customer", out.stdout)
            self.assertIn("scripted     01_qualified_full", out.stdout)
            self.assertIn("simulated    01_qualified_full", out.stdout)
            self.assertNotIn("qwen3:14b ", out.stdout.split("Note:")[0])
            self.assertEqual(check.returncode, 0, check.stdout)
            self.assertIn("[ OK ]    qwen3:8b", check.stdout)
            self.assertNotIn("[MISSING]", check.stdout)
        finally:
            shutil.rmtree(runs)

    def test_missing_model_is_reported(self):
        with FakeOllama(good_agent_policy, models=("qwen3:8b",)) as fake:
            out = subprocess.run([sys.executable, "run_tests.py", "--models", "qwen3:14b"], cwd=ROOT,
                                 env=os.environ | {"OLLAMA_URL": fake.url}, capture_output=True, text=True)
        self.assertEqual(out.returncode, 1)
        self.assertIn("ollama pull qwen3:14b", out.stdout)


if __name__ == "__main__":
    unittest.main()

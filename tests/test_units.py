import os
import shutil
import tempfile
import unittest
from datetime import datetime

from llmtest.crm import CRMError, CsvCRM, validate_changes
from llmtest.evaluate import evaluate
from llmtest.ollama_client import OllamaClient, clean_text
from llmtest.prompts import agent_system_prompt
from llmtest.tools import ToolExecutor
from tests.fake_ollama import FakeOllama, tool_call

SEED = os.path.join(os.path.dirname(__file__), "..", "data", "leads.csv")
NOW = datetime(2026, 10, 14, 10, 0)


class Base(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.crm = CsvCRM.fresh_copy(SEED, os.path.join(self.dir, "leads.csv"))

    def tearDown(self):
        shutil.rmtree(self.dir)


class CRMTests(Base):
    def test_update_persists_to_csv(self):
        self.crm.update("L001", {"budget_aed": "2,000,000", "property_type": "Apartment"})
        again = CsvCRM(self.crm.path).get("L001")
        self.assertEqual((again["budget_aed"], again["property_type"]), ("2000000", "apartment"))
        self.assertTrue(again["updated_at"])

    def test_validation_errors_are_readable(self):
        with self.assertRaises(CRMError):
            validate_changes({"property_type": "castle"})
        with self.assertRaises(CRMError):
            validate_changes({"budget_aed": "lots"})
        with self.assertRaises(CRMError):
            validate_changes({"name": "Hacker"})
        self.assertEqual(validate_changes({"preferred_area": "  "}), {})

    def test_other_rows_untouched(self):
        self.crm.mark_do_not_call("L004")
        self.assertEqual(self.crm.get("L005")["do_not_call"], "false")


class ToolTests(Base):
    def test_callback_rules(self):
        t = ToolExecutor(self.crm, "L003", NOW)
        self.assertIn("past", t.execute("schedule_callback", {"datetime": "2026-10-13T10:00"}))
        self.assertIn("Not saved", t.execute("schedule_callback", {"datetime": "thursday"}))
        self.assertIn("booked", t.execute("schedule_callback", {"datetime": "2026-10-15T16:00"}))
        row = self.crm.get("L003")
        self.assertEqual((row["status"], row["callback_at"]), ("callback", "2026-10-15T16:00"))

    def test_end_call_saves_summary(self):
        t = ToolExecutor(self.crm, "L001", NOW)
        t.execute("end_call", {"summary": "Wants a 2-bed in the Marina."})
        self.assertTrue(t.ended)
        self.assertEqual(self.crm.get("L001")["notes"], "Wants a 2-bed in the Marina.")

    def test_bad_value_returns_message_not_crash(self):
        t = ToolExecutor(self.crm, "L001", NOW)
        self.assertIn("property_type must be one of", t.execute("update_lead", {"property_type": "castle"}))
        self.assertIn("Unknown tool", t.execute("delete_lead", {}))
        self.assertEqual(t.actions, [])


class EvaluateTests(unittest.TestCase):
    EXPECTED = {"status": "qualified", "budget_aed": 2000000, "preferred_area": ["Marina"],
                "property_type": "apartment", "timeline": True, "interest_level": "hot",
                "callback_at": None, "do_not_call": False}

    def row(self, **kw):
        base = {"status": "qualified", "budget_aed": "2000000", "preferred_area": "Dubai Marina",
                "property_type": "apartment", "timeline": "2 months", "interest_level": "hot",
                "callback_at": "", "do_not_call": "false"}
        return base | kw

    def test_perfect_row_passes(self):
        ev = evaluate({"expected": self.EXPECTED}, self.row(), [], "end_call")
        self.assertTrue(ev["passed"])

    def test_hallucinated_callback_and_missing_end_call_fail(self):
        ev = evaluate({"expected": self.EXPECTED}, self.row(callback_at="2026-10-15T10:00"), [], "max_turns")
        self.assertFalse(ev["passed"])
        self.assertEqual([f["field"] for f in ev["fields"] if not f["ok"]], ["callback_at"])

    def test_invented_price_detected(self):
        events = [{"type": "lead", "text": "How much are the villas?"},
                  {"type": "agent", "text": "They start from 2.5 million dirhams."}]
        ev = evaluate({"expected": self.EXPECTED}, self.row(), events, "end_call")
        self.assertFalse(ev["passed"])
        self.assertEqual(len(ev["conduct"]["invented_money_lines"]), 1)


class ClientTests(unittest.TestCase):
    def test_tool_call_written_as_text_is_parsed_and_think_removed(self):
        def policy(body):
            return {"content": '<think>hmm</think>Sure.<tool_call>{"name": "update_lead", '
                               '"arguments": {"budget_aed": 5}}</tool_call>'}
        with FakeOllama(policy) as fake:
            res = OllamaClient(fake.url).chat("qwen3:8b", [{"role": "user", "content": "hi"}])
        self.assertEqual(res.content, "Sure.")
        self.assertEqual(res.tool_calls, [{"name": "update_lead", "arguments": {"budget_aed": 5}}])

    def test_retries_without_think_flag(self):
        with FakeOllama(lambda b: {"content": "ok"}, reject_think=True) as fake:
            res = OllamaClient(fake.url).chat("qwen3:8b", [{"role": "user", "content": "hi"}])
            self.assertEqual(res.content, "ok")
            self.assertNotIn("think", fake.requests[-1])

    def test_native_tool_calls(self):
        with FakeOllama(lambda b: {"tool_calls": [tool_call("mark_do_not_call")]}) as fake:
            res = OllamaClient(fake.url).chat("qwen3:8b", [{"role": "user", "content": "stop"}], tools=[{}])
        self.assertEqual(res.tool_calls[0]["name"], "mark_do_not_call")


class PromptTests(unittest.TestCase):
    def test_prompt_has_lead_context_and_date(self):
        lead = {"name": "Sarah Johnson", "project_interest": "Marina Vista Residences", "status": "new"}
        p = agent_system_prompt(lead, company="Bayari", agent_name="Sara", now=NOW, currency="AED", model="qwen3:8b")
        for text in ("Sarah", "Marina Vista Residences", "Wednesday 14 October 2026", "2026-10-15T16:00", "/no_think"):
            self.assertIn(text, p)
        self.assertNotIn("/no_think",
                         agent_system_prompt(lead, company="B", agent_name="S", now=NOW, currency="AED", model="gpt-oss"))


if __name__ == "__main__":
    unittest.main()

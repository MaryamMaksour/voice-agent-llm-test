"""The business rules that run in code: status, interest, callback dates, vague areas, price guard,
and the post-call check. Each test case comes from a failure seen in a real qwen3:8b run."""

import json
import os
import shutil
import tempfile
import unittest
from datetime import datetime

from llmtest.agent import Agent
from llmtest.crm import CsvCRM
from llmtest.ollama_client import OllamaClient
from llmtest.postcall import post_call_check
from llmtest.rules import (SAFE_PRICE_LINE, derive_status, guard_money, guard_tool_talk, interest_from_timeline,
                           is_vague_area, money_amounts, resolve_callback)
from llmtest.tools import ToolExecutor
from tests.fake_ollama import FakeOllama, tool_call

SEED = os.path.join(os.path.dirname(__file__), "..", "data", "leads.csv")
NOW = datetime(2026, 10, 14, 10, 0)  # a Wednesday, same as the test runs


class Base(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.crm = CsvCRM.fresh_copy(SEED, os.path.join(self.dir, "leads.csv"))

    def tearDown(self):
        shutil.rmtree(self.dir)


class CallbackDates(unittest.TestCase):
    def test_days_and_times(self):
        cases = {
            ("tomorrow", "16:00"): "2026-10-15T16:00",       # scenario 03
            ("next monday", "10:00"): "2026-10-19T10:00",    # scenario 08 (the model saved Sunday 18)
            ("Monday", "10am"): "2026-10-19T10:00",
            ("tomorrow", "evening"): "2026-10-15T18:00",     # scenario 10
            ("wednesday", "4 pm"): "2026-10-21T16:00",       # same weekday = next week
            ("today", "15:30"): "2026-10-14T15:30",
            ("2026-10-20", "9:00"): "2026-10-20T09:00",
        }
        for (day, time), want in cases.items():
            self.assertEqual(resolve_callback(day, time, NOW).strftime("%Y-%m-%dT%H:%M"), want, (day, time))

    def test_bad_or_past_values_are_explained(self):
        with self.assertRaisesRegex(ValueError, "past"):
            resolve_callback("today", "09:00", NOW)
        with self.assertRaisesRegex(ValueError, "not understood"):
            resolve_callback("someday", "10:00", NOW)
        with self.assertRaisesRegex(ValueError, "not understood"):
            resolve_callback("tomorrow", "teatime", NOW)


class InterestAndStatus(unittest.TestCase):
    def test_interest_from_the_lead_s_words(self):
        cases = {"Within the next two months.": "hot", "within a month": "hot", "In about eight months.": "warm",
                 "Probably in about six months.": "warm", "Within six months.": "warm", "within a year": "warm",
                 "maybe in two years": "cold", "Kind of. I'm just browsing for now, maybe in two years.": "cold",
                 "3 to 6 months": "warm", "asap": "hot", "not sure yet": "cold", "after the summer": None}
        for text, want in cases.items():
            self.assertEqual(interest_from_timeline(text), want, text)

    def test_status_from_details(self):
        full = {"status": "new", "budget_aed": "6000000", "preferred_area": "Dubai Hills",
                "property_type": "villa", "timeline": "8 months"}
        self.assertEqual(derive_status(full), "qualified")                          # scenario 07
        self.assertEqual(derive_status(full | {"budget_aed": ""}), "interested")    # scenario 06
        self.assertEqual(derive_status(full | {"status": "qualified", "preferred_area": ""}), "interested")
        self.assertIsNone(derive_status({"status": "new"}))
        self.assertIsNone(derive_status(full | {"status": "callback"}))             # final outcomes stay

    def test_vague_area(self):
        for text in ("anywhere quiet", "I don't mind", "any area", "no preference", "Anywhere"):
            self.assertTrue(is_vague_area(text), text)
        for text in ("Downtown", "Dubai Marina", "Jumeirah Village Circle"):
            self.assertFalse(is_vague_area(text), text)


class Tools(Base):
    def test_rules_qualify_after_the_last_detail(self):
        t = ToolExecutor(self.crm, "L007", NOW)  # scenario 07, as qwen3:8b called the tools
        t.execute("update_lead", {"budget_aed": 6000000})
        self.assertEqual(self.crm.get("L007")["status"], "interested")
        for args in ({"preferred_area": "Dubai Hills"}, {"property_type": "villa"},
                     {"timeline": "In about eight months.", "interest_level": "warm"}):
            t.execute("update_lead", args)
        row = self.crm.get("L007")
        self.assertEqual((row["status"], row["interest_level"]), ("qualified", "warm"))

    def test_interest_follows_the_timeline(self):
        t = ToolExecutor(self.crm, "L012", NOW)  # scenario 12: the model forgot interest_level
        t.execute("update_lead", {"budget_aed": 8000000, "property_type": "penthouse", "timeline": "within a month"})
        self.assertEqual(self.crm.get("L012")["interest_level"], "hot")

    def test_vague_area_is_not_saved(self):
        t = ToolExecutor(self.crm, "L011", NOW)  # scenario 11
        result = t.execute("update_lead", {"preferred_area": "anywhere quiet", "property_type": "villa"})
        self.assertIn("Area not saved", result)
        self.assertEqual(self.crm.get("L011")["preferred_area"], "")
        self.assertEqual(self.crm.get("L011")["property_type"], "villa")

    def test_callback_by_day_and_time(self):
        t = ToolExecutor(self.crm, "L008", NOW)
        self.assertIn("Monday 19 October at 10:00", t.execute("schedule_callback", {"day": "next monday", "time": "10:00"}))
        self.assertEqual(self.crm.get("L008")["callback_at"], "2026-10-19T10:00")
        self.assertIn("Not saved", t.execute("schedule_callback", {"day": "yesterday", "time": "10:00"}))


class SecondRun(Base):
    """Failures from the second qwen3:8b run (20261015-160921)."""

    def test_placeholders_do_not_block_the_status(self):
        t = ToolExecutor(self.crm, "L002", NOW)  # scenario 02: 'unknown' values made the whole update fail
        result = t.execute("update_lead", {"status": "not_interested", "preferred_area": "unknown",
                                           "property_type": "unknown", "budget_aed": 0, "timeline": "unknown",
                                           "interest_level": "cold"})
        row = self.crm.get("L002")
        self.assertEqual((row["status"], row["property_type"], row["budget_aed"], row["interest_level"]),
                         ("not_interested", "", "", ""))
        self.assertTrue(result.startswith("Saved: status"))

    def test_valid_fields_saved_when_another_is_invalid(self):
        t = ToolExecutor(self.crm, "L001", NOW)
        result = t.execute("update_lead", {"budget_aed": 2000000, "property_type": "castle"})
        self.assertEqual(self.crm.get("L001")["budget_aed"], "2000000")
        self.assertIn("property_type must be one of", result)

    def test_end_call_sent_back_once_while_details_are_missing(self):
        t = ToolExecutor(self.crm, "L009", NOW)  # scenario 09: hung up after "maybe in two years"
        t.last_lead_text = "Kind of. I'm just browsing for now, maybe in two years."
        t.execute("update_lead", {"timeline": "in two years", "status": "interested"})
        first = t.execute("end_call", {"summary": "browsing"})
        self.assertIn("Not ended", first)
        self.assertIn("budget", first)
        self.assertFalse(t.ended)
        t.execute("end_call", {"summary": "browsing"})  # the model insists: accepted
        self.assertTrue(t.ended)

    def test_end_call_accepted_when_the_lead_says_bye_or_is_done(self):
        for lead_id, text, args in (("L011", "OK, bye.", {"budget_aed": 3000000}),
                                    ("L002", "No, not interested.", {"status": "not_interested"})):
            t = ToolExecutor(self.crm, lead_id, NOW)
            t.last_lead_text = text
            t.execute("update_lead", args)
            t.execute("end_call", {"summary": "done"})
            self.assertTrue(t.ended, lead_id)

    def test_instructions_are_never_spoken(self):
        text = "I can't share the exact prices. A consultant will be in touch soon. Say goodbye, call end_call."
        clean, removed = guard_tool_talk(text)
        self.assertEqual(clean, "I can't share the exact prices. A consultant will be in touch soon.")
        self.assertEqual(removed, ["Say goodbye, call end_call."])

    def test_post_call_marks_not_interested(self):
        found = {"still_interested": False, "budget_aed": None, "preferred_area": None, "property_type": None,
                 "timeline": None}
        with FakeOllama(lambda b: {"content": json.dumps(found)}) as fake:
            post_call_check(OllamaClient(fake.url), "qwen3:8b", self.crm, "L002",
                            [{"type": "lead", "text": "Not interested."}], num_ctx=2048)
        self.assertEqual(self.crm.get("L002")["status"], "not_interested")


class PriceGuard(unittest.TestCase):
    def test_amounts(self):
        self.assertEqual(money_amounts("from AED 2,000,000"), {2000000})
        self.assertEqual(money_amounts("about 2.5 million dirhams"), {2500000})
        self.assertEqual(money_amounts("one point two million"), {1200000})
        self.assertEqual(money_amounts("between five and six million"), {6000000})
        self.assertEqual(money_amounts("3 bedrooms, in 6 months, Monday 19 October 2026 at 10:00"), set())

    def test_invented_price_removed_but_question_kept(self):
        text = "The villas in Dubai Hills start from AED 2,000,000. Would you like to know more about the options?"
        clean, removed = guard_money(text, allowed=set())
        self.assertEqual(removed, ["The villas in Dubai Hills start from AED 2,000,000."])
        self.assertEqual(clean, f"{SAFE_PRICE_LINE} Would you like to know more about the options?")

    def test_repeating_the_lead_s_budget_is_fine(self):
        text = "So your budget is two million dirhams. Which area do you prefer?"
        self.assertEqual(guard_money(text, allowed=money_amounts("Around 2 million.")), (text, []))

    def test_agent_applies_the_guard(self):
        def policy(body):
            return {"content": "They start from 2.5 million dirhams. What is your budget?"}
        with FakeOllama(policy) as fake, tempfile.TemporaryDirectory() as d:
            crm = CsvCRM.fresh_copy(SEED, os.path.join(d, "leads.csv"))
            agent = Agent(OllamaClient(fake.url), "qwen3:8b", crm, "L011", NOW, company="B", agent_name="Sara",
                          currency="AED", temperature=0, num_ctx=2048, max_tool_rounds=3)
            turn = agent.respond("How much are the villas?")
        self.assertEqual(turn.text, f"{SAFE_PRICE_LINE} What is your budget?")
        self.assertEqual(turn.guarded, ["They start from 2.5 million dirhams."])
        self.assertNotIn("2.5 million", agent.messages[-1]["content"])  # history matches what was said


class PostCallCheck(Base):
    EVENTS = [{"type": "agent", "text": "Hi, this is Sara. Am I speaking with Henry?"},
              {"type": "lead", "text": "Yes. A three bedroom penthouse in Downtown, about eight million, "
                                       "within a month."}]

    def answer(self, obj):
        return lambda body: {"content": json.dumps(obj)}

    def test_fills_only_missing_fields_then_applies_rules(self):
        # scenario 12: the agent saved budget, type and timeline but not the area
        self.crm.update("L012", {"budget_aed": 8000000, "property_type": "penthouse", "timeline": "within a month"})
        found = {"still_interested": True, "budget_aed": 9000000, "preferred_area": "Downtown",
                 "property_type": "penthouse", "timeline": "within a month"}
        with FakeOllama(self.answer(found)) as fake:
            changed, line = post_call_check(OllamaClient(fake.url), "qwen3:8b", self.crm, "L012", self.EVENTS,
                                            num_ctx=2048)
            self.assertIn("format", fake.requests[-1])
        row = self.crm.get("L012")
        self.assertEqual((row["preferred_area"], row["status"], row["interest_level"], row["budget_aed"]),
                         ("Downtown", "qualified", "hot", "8000000"))  # the agent's budget is kept
        self.assertIn("preferred_area=Downtown", line)

    def test_skips_final_outcomes_and_ignores_bad_answers(self):
        self.crm.update("L002", {"status": "not_interested"})
        with FakeOllama(self.answer({"budget_aed": 1})) as fake:
            _, line = post_call_check(OllamaClient(fake.url), "qwen3:8b", self.crm, "L002", self.EVENTS, num_ctx=2048)
            self.assertEqual(fake.requests, [])
        self.assertIn("skipped", line)
        with FakeOllama(lambda b: {"content": "not json"}) as fake:
            changed, line = post_call_check(OllamaClient(fake.url), "qwen3:8b", self.crm, "L012", self.EVENTS,
                                            num_ctx=2048)
        self.assertEqual(changed, {})
        self.assertIn("no usable answer", line)

    def test_vague_area_and_bad_type_are_not_filled(self):
        found = {"still_interested": True, "budget_aed": None, "preferred_area": "anywhere quiet",
                 "property_type": "castle", "timeline": None}
        with FakeOllama(self.answer(found)) as fake:
            post_call_check(OllamaClient(fake.url), "qwen3:8b", self.crm, "L011", self.EVENTS, num_ctx=2048)
        row = self.crm.get("L011")
        self.assertEqual((row["preferred_area"], row["property_type"], row["status"]), ("", "", "interested"))


if __name__ == "__main__":
    unittest.main()

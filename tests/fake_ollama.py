"""A fake Ollama HTTP server for offline tests (standard library only)."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class FakeOllama:
    def __init__(self, policy, models=("qwen3:8b", "qwen3:14b"), reject_think=False) -> None:
        self.policy = policy          # policy(body) -> {"content": str, "tool_calls": [...]} message
        self.models = list(models)
        self.reject_think = reject_think
        self.requests: list[dict] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, obj):
                data = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path == "/api/version":
                    self._send(200, {"version": "0.0-fake"})
                elif self.path == "/api/tags":
                    self._send(200, {"models": [{"name": m} for m in outer.models]})
                else:
                    self._send(404, {"error": "not found"})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append(body)
                if outer.reject_think and "think" in body:
                    self._send(400, {"error": "think value not supported"})
                    return
                msg = {"role": "assistant", "content": "", **outer.policy(body)}
                self._send(200, {"message": msg, "prompt_eval_count": 10, "eval_count": 5})

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def __enter__(self):
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()


def tool_call(name, **arguments):
    return {"function": {"name": name, "arguments": arguments}}


def good_agent_policy(body):
    """Behaves like a well-behaved agent for scenarios 01, 04 and the caller role."""
    msgs = body["messages"]
    if "tools" not in body:  # caller role-play
        return {"content": "Bye. [END]"}
    last = msgs[-1]
    if last["role"] == "tool":
        if last.get("tool_name") == "end_call":
            return {"content": ""}
        return {"content": "Thank you. Goodbye!" if "ended" in last["content"] else "Got it, thanks."}
    text = last["content"].lower()
    if "stop calling" in text:
        return {"tool_calls": [tool_call("mark_do_not_call")], "content": "Sorry for the trouble, goodbye."}
    if "two million" in text:
        return {"tool_calls": [tool_call("update_lead", budget_aed=2000000)]}
    if "marina" in text:
        return {"tool_calls": [tool_call("update_lead", preferred_area="Dubai Marina")]}
    if "apartment" in text:
        return {"tool_calls": [tool_call("update_lead", property_type="apartment")]}
    if "two months" in text:
        return {"tool_calls": [tool_call("update_lead", timeline="within two months", interest_level="hot",
                                         status="qualified")]}
    if "bye" in text or "goodbye" in text or "hang up" in text:
        return {"tool_calls": [tool_call("end_call", summary="Call finished.")], "content": "Goodbye!"}
    return {"content": "Hi, this is Sara from Bayari Investments. Is this a good time?"}

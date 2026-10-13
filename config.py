"""All settings in one place. Override any of them with environment variables."""

import os

# Where Ollama runs. On the server itself: http://localhost:11434
# From your laptop: http://<server-ip>:11434  (Ollama must be started with OLLAMA_HOST=0.0.0.0)
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

# The three models under test (step 1, 2, 3 of the plan). Must be pulled in Ollama first.
MODELS = os.getenv("MODELS", "qwen3:8b,qwen3:14b,qwen3:30b-a3b").split(",")

# The model that plays the customer in "simulated" mode. Keep it FIXED across runs,
# so every agent model is tested against the same kind of caller.
CALLER_MODEL = os.getenv("CALLER_MODEL", "qwen3:14b")

# A fixed "now" so that "call me tomorrow at 4pm" always has one correct answer.
SIMULATED_NOW = os.getenv("SIMULATED_NOW", "2026-10-14T10:00")  # a Wednesday

COMPANY_NAME = os.getenv("COMPANY_NAME", "Bayari Investments")
AGENT_NAME = os.getenv("AGENT_NAME", "Sara")
CURRENCY = "AED"

AGENT_TEMPERATURE = float(os.getenv("AGENT_TEMPERATURE", "0.2"))
CALLER_TEMPERATURE = float(os.getenv("CALLER_TEMPERATURE", "0.7"))
NUM_CTX = int(os.getenv("NUM_CTX", "8192"))          # context window per conversation
MAX_TURNS = int(os.getenv("MAX_TURNS", "16"))        # agent replies per call before we stop it
MAX_TOOL_ROUNDS = int(os.getenv("MAX_TOOL_ROUNDS", "6"))
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "600"))  # slow hardware is fine, we test quality

LEADS_SEED = os.getenv("LEADS_SEED", "data/leads.csv")
SCENARIOS_FILE = os.getenv("SCENARIOS_FILE", "scenarios/scenarios.json")
RUNS_DIR = os.getenv("RUNS_DIR", "runs")

# A model is "good enough" when it reaches both targets.
TARGET_FIELD_ACCURACY = 0.95
TARGET_SCENARIO_PASS_RATE = 0.90

import os
from pathlib import Path

# Swappable without a code change via LLM_JUDGE_MODEL. Switched from
# qwen3guard-gen:4b to a general instruct model - see PROGRESS_LOG.md for why
# (qwen3guard-gen ignored our system prompt/schema entirely and missed
# paraphrase-disguised attacks on the 500-row external evaluation).
MODEL_NAME = os.getenv("LLM_JUDGE_MODEL", "deepseek-r1:7b")
OLLAMA_HOST = os.getenv("OLLAMA_HOST")  # None -> ollama client's own default
REQUEST_TIMEOUT_SECONDS = float(os.getenv("LLM_JUDGE_TIMEOUT_SECONDS", "30"))

# Where inference runs. "ollama" = local Ollama (the documented default);
# "groq" = Groq's OpenAI-compatible cloud API, used when the local machine
# can't hold the 8B model. Groq serves the same Llama-3.1-8B-Instruct weights
# (unquantized) as `llama-3.1-8b-instant`. See PROGRESS_LOG.md.
LLM_JUDGE_BACKEND = os.getenv("LLM_JUDGE_BACKEND", "ollama").lower()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
# gpt-oss models on Groq emit chain-of-thought unless told otherwise. "low" keeps
# token use (and free-tier rate-limit pressure) down; the few-shot example in
# prompt.py carries most of the reasoning load. Ignored by non-gpt-oss models.
GROQ_REASONING_EFFORT = os.getenv("GROQ_REASONING_EFFORT", "low")

LLM_JUDGE_PATH = Path(os.path.dirname(__file__))
PROMPTS_PATH = LLM_JUDGE_PATH / "prompts"

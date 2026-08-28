import os

# Swappable without a code change via LLM_JUDGE_MODEL. Switched from
# qwen3guard-gen:4b to a general instruct model - see PROGRESS_LOG.md for why
# (qwen3guard-gen ignored our system prompt/schema entirely and missed
# paraphrase-disguised attacks on the 500-row external evaluation).
MODEL_NAME = os.getenv("LLM_JUDGE_MODEL", "llama3.1:8b-instruct-q4_K_M")
OLLAMA_HOST = os.getenv("OLLAMA_HOST")  # None -> ollama client's own default
REQUEST_TIMEOUT_SECONDS = float(os.getenv("LLM_JUDGE_TIMEOUT_SECONDS", "30"))

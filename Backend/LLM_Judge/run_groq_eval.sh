#!/usr/bin/env bash
# Evaluate the LLM Judge against a frozen sample via the Groq API.
#
#   export GROQ_API_KEY=gsk_...        # once per shell
#   ./run_groq_eval.sh                 # resumable - just run it again to continue
#
# Free tier (openai/gpt-oss-20b): ~1000 requests/day, ~10/min. When the daily
# cap is hit the run starts logging ERROR rows - Ctrl-C and re-run tomorrow;
# it skips everything already done and retries only the errored rows.
# Metrics are printed at the end and also work on a partial CSV:
#   ../venv/bin/python run_eval.py results.csv --metrics-only
set -euo pipefail
cd "$(dirname "$0")"

: "${GROQ_API_KEY:?set GROQ_API_KEY first: export GROQ_API_KEY=gsk_...}"

export LLM_JUDGE_BACKEND=groq
export GROQ_MODEL="${GROQ_MODEL:-openai/gpt-oss-20b}"
export GROQ_REASONING_EFFORT="${GROQ_REASONING_EFFORT:-low}"
export LLM_JUDGE_SELF_CONSISTENCY="${LLM_JUDGE_SELF_CONSISTENCY:-0}"   # single pass
export LLM_JUDGE_TIMEOUT_SECONDS="${LLM_JUDGE_TIMEOUT_SECONDS:-180}"

SAMPLE="${SAMPLE:-eval_sample_50k.parquet}"
OUT="${OUT:-results_groq.csv}"
RPM="${RPM:-8}"          # requests/minute cap; raise on the paid tier
WORKERS="${WORKERS:-4}"

../venv/bin/python run_eval.py "$SAMPLE" --out "$OUT" --workers "$WORKERS" --rpm "$RPM"

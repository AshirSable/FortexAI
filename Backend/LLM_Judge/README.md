# LLM Judge (Stage 4)

Final stage of FortexAI's prompt-injection detection cascade. Content that earlier,
cheaper stages have already flagged as suspicious is passed here to a local LLM
(`llama3.1:8b-instruct-q4_K_M` via Ollama) for a structured verdict: `malicious`,
`benign`, or `uncertain`.

The judge sends the model a system prompt describing the task plus one worked
few-shot example, and forces a JSON-schema response (`schema.Verdict`). It fails
safe to `malicious` on any error rather than silently passing content through.
`CLAUDE.md` has the full design rationale and history (including why the module
switched off its original `qwen3guard-gen:4b` model — read Section 5 before changing
`judge.py`, `prompt.py`, or `guardrails.py`).

## Status

| File | Purpose | Status |
|---|---|---|
| `schema.py` | Pydantic `Verdict` model (`verdict`, `confidence`, `reason`) | done |
| `prompt.py` | Builds the chat messages: `SYSTEM_PROMPT` + one worked few-shot turn + the flagged content | done |
| `config.py` | `LLM_JUDGE_MODEL` / `OLLAMA_HOST` / `LLM_JUDGE_TIMEOUT_SECONDS`, all env-var overridable | done |
| `judge.py` | `evaluate_prompt(text, context)` — Ollama call with forced JSON schema + `temperature=0`; fails safe to `malicious` (confidence 0.0) on timeout / connection error / Ollama error / schema-invalid output | done |
| `guardrails.py` | Recommended entry point. Input isolation, hidden base64/ROT13 payload decoding, two-pass self-consistency check (on by default) | done |
| `evaluate.py` | Runs `guardrails.evaluate_with_guardrails()` over `data/labeled_test_set.json`, reports accuracy/false-positive/false-negative breakdown, writes `evaluation_results.csv` (git-ignored) | done |
| `evaluate_parquet.py` | Same, for larger external datasets in parquet form (`text` + `label` columns); dedupes, class-balanced sample, writes `parquet_evaluation_results.csv` (git-ignored) | done |
| `data/labeled_test_set.json` | 100 hand-written labeled prompts (45 malicious / 45 benign / 10 ambiguous) across ~19 injection categories | done |
| `canary_test.py` + `data/canary_set.json` | Regression suite — 12 already-confirmed cases checked one at a time; `known_issue: true` entries (2) are reported but don't fail the run | done |
| `tests/test_judge.py` | 8 unit tests for `judge.py`, Ollama call mocked | done |

## Accuracy

The last committed measurement — **93/100 overall, 45/45 benign (zero false
positives), 0 dangerous false negatives** — was taken against the module's
*previous* model (`qwen3guard-gen:4b`). The current `llama3.1:8b-instruct-q4_K_M` +
few-shot configuration has **not** had a fresh evaluation committed yet
(`evaluation_results.csv` is regenerated, not tracked).

Run `python evaluate.py` against a live model for current numbers, and see
`CLAUDE.md` Section 5 for the full root-cause history behind every design choice.

## Setup

Dependencies live in the shared `Backend/requirements.txt` (this module doesn't
maintain its own). From `Backend/`:

```bash
source venv/bin/activate
pip install -r requirements.txt
```

This installs `ollama` (Python client) and `pydantic` alongside the rest of the
backend's dependencies. `pandas` / `pyarrow` (used by `evaluate_parquet.py`) are
in there too.

### Install Ollama

```bash
brew install ollama
brew services start ollama        # or: ollama serve
```

Verify the server is up:

```bash
curl http://localhost:11434/api/version
```

### Pull the judge model

```bash
ollama pull llama3.1:8b-instruct-q4_K_M
```

~4.7 GB download. Everyone on the team should pull this exact tag so results are
comparable — `CLAUDE.md` Section 5 shows how sensitive this module is to the exact
model build. Confirm it's registered:

```bash
ollama list   # should show llama3.1:8b-instruct-q4_K_M
```

## Configuration

All of `config.py` is overridable via environment variables so the model can be
swapped with no code change:

| Variable | Default | Purpose |
|---|---|---|
| `LLM_JUDGE_MODEL` | `llama3.1:8b-instruct-q4_K_M` | Model name passed to Ollama |
| `OLLAMA_HOST` | unset (client's own default, `http://localhost:11434`) | Ollama server to call |
| `LLM_JUDGE_TIMEOUT_SECONDS` | `30` | Per-request timeout before failing safe |

## Usage

```python
import judge
import guardrails

# Direct call - no input isolation, no hidden-payload decoding, no self-consistency.
verdict = judge.evaluate_prompt(flagged_text, context=optional_context)

# Hardened call (recommended for production) - delimits flagged_text as data,
# decodes and checks any hidden base64/ROT13 payloads, and by default requires
# two independently-framed passes to agree before returning a confident verdict.
verdict = guardrails.evaluate_with_guardrails(flagged_text, context=optional_context)

# self_consistency=False skips the second pass (faster, single model call,
# more sensitive to superficial framing) if you need lower latency.
verdict = guardrails.evaluate_with_guardrails(flagged_text, self_consistency=False)
```

`verdict` is always a `schema.Verdict` — `judge.evaluate_prompt` never raises for
timeouts, connection errors, or malformed model output; it fails safe to
`verdict="malicious", confidence=0.0` with a `reason` explaining what went wrong, so
a broken judge escalates for manual review instead of silently waving content
through as benign.

## Running tests

```bash
source ../venv/bin/activate            # from Backend/LLM_Judge/
python -m pytest tests/test_judge.py -v
```

Tests mock the Ollama call entirely and don't require a running model.

To measure actual judgment quality (requires a running model):

```bash
python evaluate.py        # 100-prompt labeled set -> evaluation_results.csv
python canary_test.py     # regression suite; exits non-zero on any real regression
python evaluate_parquet.py <file.parquet> [--sample-size 500] [--seed 42]
```

# LLM Judge (Stage 4)

Final stage of FortexAI's prompt-injection detection cascade. Content that earlier,
cheaper stages have already flagged as suspicious is passed here to a local LLM
(`qwen3guard-gen:4b` via Ollama) for a structured verdict: `malicious`, `benign`, or
`uncertain`.

The model has its own chat template baked into the GGUF file that overrides any
system prompt or JSON-schema output we ask for, so `judge.py` lets it answer in its
own native safety-classification format (`Safety: Safe/Unsafe/Controversial` +
`Categories: ...`) and parses that in Python - "Jailbreak" is the model's own
category name for instruction-hijacking, which maps to `malicious`. Full
investigation in `CLAUDE.md` section 5.

## Status

| File | Purpose | Status |
|---|---|---|
| `schema.py` | Pydantic `Verdict` model (`verdict`, `confidence`, `reason`) | done |
| `prompt.py` | Builds the plain user message sent to the model (no system prompt, no few-shot injection into the live call) | done |
| `config.py` | `MODEL_NAME` / `OLLAMA_HOST` / timeout, all env-var overridable | done |
| `judge.py` | `evaluate_prompt(text, context)` - calls Ollama, parses the model's native `Safety:`/`Categories:` output, maps `Categories: Jailbreak` to `malicious`; fails safe to `malicious` on timeout/error/unparseable output | done |
| `guardrails.py` | Recommended entry point. Input isolation, hidden base64/ROT13 payload decoding, two-pass self-consistency check (on by default) | done |
| `tests/test_judge.py` | Unit tests for `judge.py`, Ollama call mocked | done |
| `data/few_shot_examples.json` | Labeled reference examples for `evaluate.py` (no longer injected into the live prompt) | done (placeholders) |
| `throwaway_ollama_test.py` | One-off script confirming Ollama + the model return a valid parsed `Verdict` | done, delete after use |
| `evaluate.py` | Runs `guardrails.evaluate_with_guardrails()` over `data/labeled_test_set.json`, reports accuracy/false-positive/false-negative breakdown, writes `evaluation_results.csv` | done |
| `data/labeled_test_set.json` | 100 hand-written labeled prompts (45 malicious / 45 benign / 10 ambiguous) across a range of injection styles | done |
| `canary_test.py` | Known-bad prompts that must always be flagged | stub |
| `data/canary_set.json` | Canary prompts | placeholder (empty) |

## Current accuracy

Latest run of `python evaluate.py` against the 100 labeled prompts (2026-08-08):

| | Result |
|---|---|
| Overall | 93/100 (93%) |
| Benign correctly left alone | 45/45 (100%) - zero false positives |
| Malicious caught cleanly | 43/45 (95.6%) |
| Dangerous false negatives (malicious silently called benign) | **0** |

The 2 remaining malicious prompts land on `uncertain` (escalated for review), not a
silent `benign`. Full root-cause history and reasoning for every fix live in
`CLAUDE.md` section 5 - read that before changing `judge.py` or `guardrails.py`.
Per-prompt results are in `evaluation_results.csv`.

## Setup

Dependencies live in the shared `Backend/requirements.txt` (this module doesn't
maintain its own). From `Backend/`:

```bash
source venv/bin/activate
pip install -r requirements.txt
```

This installs `ollama` (Python client) and `pydantic` alongside the rest of the
backend's dependencies.

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

`qwen3guard-gen:4b` is **not** in Ollama's own registry - only Qwen3Guard's 0.6B
variant is (`sileader/qwen3guard:0.6b`). The 4B generative variant
(`Qwen/Qwen3Guard-Gen-4B`) is only published on Hugging Face, so we pull a
community GGUF quantization from there and alias it to the name the rest of
this module expects:

```bash
ollama pull hf.co/mradermacher/Qwen3Guard-Gen-4B-GGUF:Q4_K_M
ollama cp hf.co/mradermacher/Qwen3Guard-Gen-4B-GGUF:Q4_K_M qwen3guard-gen:4b
```

~2.7GB download. Everyone on the team should run these same two commands so
`qwen3guard-gen:4b` resolves consistently everywhere.

Confirm it's registered:

```bash
ollama list   # should show qwen3guard-gen:4b
```

### Confirm the model returns a valid, parseable verdict

From `Backend/LLM_Judge/` (with the venv active):

```bash
python throwaway_ollama_test.py
```

Runs `judge.evaluate_prompt()` on one sample flagged prompt and prints the
resulting `Verdict`, or a clear error if Ollama/the model isn't reachable. Delete
this script once you've confirmed setup works - it's not part of the module.

## Configuration

All of `config.py` is overridable via environment variables so the model can be
swapped (e.g. dev's 0.6B/4B GGUF -> a promoted production model) with no code
change:

| Variable | Default | Purpose |
|---|---|---|
| `LLM_JUDGE_MODEL` | `qwen3guard-gen:4b` | Model name passed to Ollama |
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

`verdict` is always a `schema.Verdict` - `judge.evaluate_prompt` never raises for
timeouts, connection errors, or malformed model output; it fails safe to
`verdict="malicious", confidence=0.0` with a `reason` explaining what went wrong,
so a broken judge escalates for manual review instead of silently waving content
through as benign.

## Running tests

```bash
source ../venv/bin/activate   # from Backend/LLM_Judge/
python -m pytest tests/test_judge.py -v
```

Use `python -m pytest` (not the bare `pytest` executable) run from
`Backend/LLM_Judge/` - there's no `tests/__init__.py`, so pytest needs the current
working directory on `sys.path` to resolve `import judge`. Tests mock the Ollama
call entirely and don't require a running model.

To measure actual judgment quality (requires a running model), run
`python evaluate.py` - see "Current accuracy" above.

import logging
import time

import httpx
import ollama
from pydantic import ValidationError

from config import (
    GROQ_API_KEY,
    GROQ_BASE_URL,
    GROQ_MODEL,
    GROQ_REASONING_EFFORT,
    LLM_JUDGE_BACKEND,
    MODEL_NAME,
    OLLAMA_HOST,
    REQUEST_TIMEOUT_SECONDS,
)
from prompt import build_messages
from schema import Verdict, VerdictLabel

logger = logging.getLogger(__name__)

_client = ollama.Client(host=OLLAMA_HOST, timeout=REQUEST_TIMEOUT_SECONDS)

_GROQ_MAX_RETRIES = 10


class JudgeUnavailable(RuntimeError):
    """The backend could not be reached / kept rate-limiting. Distinct from a
    real verdict so an evaluation harness can retry instead of recording a
    fail-safe 'malicious' as if the model had actually classified the text."""


def _fail_safe(reason: str) -> Verdict:
    # A judge that can't be trusted to have actually looked at the content
    # must never resolve to "benign" - fail toward escalation instead, at
    # zero confidence since this isn't a real classification.
    return Verdict(verdict=VerdictLabel.MALICIOUS, confidence=0.0, reason=reason)


def _chat_via_ollama(messages: list[dict]) -> str:
    response = _client.chat(
        model=MODEL_NAME,
        messages=messages,
        format=Verdict.model_json_schema(),
        options={"temperature": 0},
    )
    return response["message"]["content"]


def _chat_via_groq(messages: list[dict]) -> str:
    """Call Groq's OpenAI-compatible chat endpoint. JSON mode (response_format
    json_object) + the prompt's explicit schema instruction stand in for
    Ollama's `format=` schema forcing - the pydantic parse in evaluate_prompt
    is the real guarantee either way. Retries on 429 / 5xx with backoff since
    the free tier is rate limited."""
    if not GROQ_API_KEY:
        raise ollama.RequestError("LLM_JUDGE_BACKEND=groq but GROQ_API_KEY is not set")

    payload = {
        "model": GROQ_MODEL,
        "messages": messages,
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    if "gpt-oss" in GROQ_MODEL and GROQ_REASONING_EFFORT:
        payload["reasoning_effort"] = GROQ_REASONING_EFFORT
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
    url = f"{GROQ_BASE_URL}/chat/completions"

    last_exc: Exception | None = None
    for attempt in range(_GROQ_MAX_RETRIES):
        try:
            resp = httpx.post(url, json=payload, headers=headers, timeout=REQUEST_TIMEOUT_SECONDS)
        except httpx.HTTPError as e:
            last_exc = e
            time.sleep(min(2 ** attempt, 30))
            continue

        if resp.status_code == 429 or resp.status_code >= 500:
            retry_after = resp.headers.get("retry-after")
            wait = float(retry_after) if retry_after else min(2 ** attempt, 60)
            logger.warning("Groq %s; retrying in %.1fs (attempt %d)", resp.status_code, wait, attempt + 1)
            time.sleep(wait)
            continue

        if resp.status_code != 200:
            raise ollama.ResponseError(f"Groq returned {resp.status_code}: {resp.text[:200]}")

        return resp.json()["choices"][0]["message"]["content"]

    raise ollama.ResponseError(f"Groq unreachable after {_GROQ_MAX_RETRIES} retries: {last_exc}")


def evaluate_prompt(text: str, context: str | None = None, *, strict: bool = False) -> Verdict:
    """Classify `text` (optionally with `context`) using the configured backend
    (local Ollama by default, or Groq when LLM_JUDGE_BACKEND=groq).

    Always returns a Verdict - a timeout, connection failure, or output that
    doesn't parse resolves to a fail-safe "malicious" verdict for manual
    escalation rather than an implicit benign.

    `strict=True` (for evaluation harnesses) instead raises `JudgeUnavailable`
    when the backend can't be reached, so a rate-limit / network failure is
    retried rather than silently scored as a real "malicious" classification.
    A malformed model *response* still fails safe either way - that came from
    the model.
    """
    messages = build_messages(text, context=context)
    backend_label = GROQ_MODEL if LLM_JUDGE_BACKEND == "groq" else MODEL_NAME
    try:
        if LLM_JUDGE_BACKEND == "groq":
            raw_content = _chat_via_groq(messages)
        else:
            raw_content = _chat_via_ollama(messages)
    except (httpx.TimeoutException, httpx.ConnectError) as e:
        logger.warning("Judge call to %s timed out/unreachable: %s", backend_label, e)
        if strict:
            raise JudgeUnavailable(f"{type(e).__name__}: {e}") from e
        return _fail_safe(f"Judge call failed ({type(e).__name__}); escalating for manual review.")
    except (ollama.ResponseError, ollama.RequestError) as e:
        logger.warning("Judge call to %s errored: %s", backend_label, e)
        if strict:
            raise JudgeUnavailable(f"{type(e).__name__}: {e}") from e
        return _fail_safe(f"Judge call errored ({type(e).__name__}); escalating for manual review.")

    try:
        return Verdict.model_validate_json(raw_content)
    except ValidationError as e:
        logger.warning("Judge output failed schema validation: %s | raw=%r", e, raw_content)
        return _fail_safe("Judge output was malformed/unparseable; escalating for manual review.")

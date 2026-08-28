import logging

import httpx
import ollama
from pydantic import ValidationError

from config import MODEL_NAME, OLLAMA_HOST, REQUEST_TIMEOUT_SECONDS
from prompt import build_messages
from schema import Verdict, VerdictLabel

logger = logging.getLogger(__name__)

_client = ollama.Client(host=OLLAMA_HOST, timeout=REQUEST_TIMEOUT_SECONDS)


def _fail_safe(reason: str) -> Verdict:
    # A judge that can't be trusted to have actually looked at the content
    # must never resolve to "benign" - fail toward escalation instead, at
    # zero confidence since this isn't a real classification.
    return Verdict(verdict=VerdictLabel.MALICIOUS, confidence=0.0, reason=reason)


def evaluate_prompt(text: str, context: str | None = None) -> Verdict:
    """Classify `text` (optionally with `context`) using the local Ollama model.

    Always returns a Verdict - a timeout, connection failure, or output that
    doesn't parse resolves to a fail-safe "malicious" verdict for manual
    escalation rather than an implicit benign.
    """
    try:
        response = _client.chat(
            model=MODEL_NAME,
            messages=build_messages(text, context=context),
            format=Verdict.model_json_schema(),
            options={"temperature": 0},
        )
    except (httpx.TimeoutException, httpx.ConnectError) as e:
        logger.warning("Judge call to %s timed out/unreachable: %s", MODEL_NAME, e)
        return _fail_safe(f"Judge call failed ({type(e).__name__}); escalating for manual review.")
    except (ollama.ResponseError, ollama.RequestError) as e:
        logger.warning("Judge call to %s errored: %s", MODEL_NAME, e)
        return _fail_safe(f"Judge call errored ({type(e).__name__}); escalating for manual review.")

    raw_content = response["message"]["content"]
    try:
        return Verdict.model_validate_json(raw_content)
    except ValidationError as e:
        logger.warning("Judge output failed schema validation: %s | raw=%r", e, raw_content)
        return _fail_safe("Judge output was malformed/unparseable; escalating for manual review.")

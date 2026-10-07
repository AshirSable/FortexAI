# POST /v1/screen - the public endpoint that customers call with an API key.
import logging
import os
import time
import uuid

from app.type_store import Phase, PhaseInput, Verdict
from auth.database import get_db
from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from gateway import state
from gateway.keys import hash_key
from gateway.models import ApiKey, ScreenEvent, utc_now

logger = logging.getLogger(__name__)
router = APIRouter(tags=["screen"])

# same bar main.py uses before writing a verdict back to semantic search
CONFIRM_CONFIDENCE_THRESHOLD = 0.90

VERDICT_MAP = {
    Verdict.attack: "blocked",
    Verdict.benign: "passed",
    Verdict.undetermined: "flagged",
}


class ScreenBody(BaseModel):
    prompt: str
    end_user: str | None = None


def find_key(db, authorization):
    # header looks like: "Bearer ftx_live_abc..."
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing API key")
    token = authorization[len("Bearer "):].strip()
    key = db.query(ApiKey).filter(ApiKey.key_hash == hash_key(token)).first()
    if key is None or key.revoked:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return key


def run_pipeline(prompt):
    # returns (verdict, phase, confidence, latency_ms)
    start = time.perf_counter()
    try:
        result = state.pipeline.run(PhaseInput(text=prompt))
    except Exception as e:
        logger.warning("pipeline crashed: %s", e)
        result = None

    if result is None or result.is_err():
        # fail closed: a broken pipeline blocks the prompt
        ms = (time.perf_counter() - start) * 1000
        return "blocked", "error", 0.0, ms

    success = result.unwrap()
    verdict = VERDICT_MAP[success.verdict]
    write_back(prompt, success)
    return verdict, success.at_phase.name, success.confidence, success.latency_ms


def write_back(prompt, success):
    # teach semantic search from confident answers of the later stages
    if os.environ.get("FORTEX_DISABLE_WRITEBACK") == "1":
        return
    if state.semantic_search is None:
        return
    if success.at_phase == Phase.semantic_search:
        return
    if success.confidence < CONFIRM_CONFIDENCE_THRESHOLD:
        return
    try:
        label = "attack" if success.verdict == Verdict.attack else "benign"
        state.semantic_search.confirm(prompt, label)
    except Exception as e:
        logger.warning("semantic search write-back failed: %s", e)


@router.post("/v1/screen")
def screen_prompt(
    body: ScreenBody,
    authorization: str | None = Header(default=None),
    db: Session = Depends(get_db),
):
    key = find_key(db, authorization)

    if key.status == "stopped":
        # switch is OFF: let it through without running detection
        verdict, phase, confidence, latency_ms, screened = "bypassed", "none", 0.0, 0.0, False
    else:
        if state.pipeline is None:
            raise HTTPException(status_code=503, detail="detection pipeline is not ready yet")
        verdict, phase, confidence, latency_ms = run_pipeline(body.prompt)
        screened = True

    # blocked is the only verdict that stops the prompt
    allowed = verdict != "blocked"

    key.last_used_at = utc_now()
    db.add(
        ScreenEvent(
            key_id=key.id,
            user_id=key.user_id,
            prompt_preview=body.prompt[:200],
            end_user=body.end_user,
            verdict=verdict,
            phase=phase,
            confidence=confidence,
            latency_ms=latency_ms,
            screened=screened,
        )
    )
    db.commit()

    return {
        "request_id": "req_" + uuid.uuid4().hex[:12],
        "allowed": allowed,
        "screened": screened,
        "verdict": verdict,
        "phase": phase,
        "confidence": confidence,
        "latency_ms": latency_ms,
    }

# Dashboard numbers. All endpoints are scoped to the logged-in user.
from datetime import timedelta

from auth.database import get_db
from auth.models import User
from auth.router import get_current_user
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from gateway.models import ScreenEvent, utc_now

router = APIRouter(prefix="/metrics", tags=["metrics"])

PERIODS = {
    "1h": timedelta(hours=1),
    "24h": timedelta(hours=24),
    "7d": timedelta(days=7),
    "30d": timedelta(days=30),
}
# (bucket size, number of buckets) for the trend chart
BUCKETS = {
    "1h": (timedelta(minutes=5), 12),
    "24h": (timedelta(hours=1), 24),
    "7d": (timedelta(days=1), 7),
    "30d": (timedelta(days=1), 30),
}
HIGH_RISK_MIN_BLOCKED = 3


def get_since(period):
    if period not in PERIODS:
        raise HTTPException(status_code=422, detail="period must be 1h, 24h, 7d or 30d")
    return utc_now() - PERIODS[period]


def user_events(db, user, since, key_id):
    query = db.query(ScreenEvent).filter(
        ScreenEvent.user_id == user.id, ScreenEvent.created_at >= since
    )
    if key_id is not None:
        query = query.filter(ScreenEvent.key_id == key_id)
    return query


@router.get("/summary")
def summary(
    period: str = "24h",
    key_id: int | None = None,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    events = user_events(db, user, get_since(period), key_id).all()

    counts = {"blocked": 0, "flagged": 0, "passed": 0, "bypassed": 0}
    phases = {}
    blocked_per_user = {}
    latencies = []
    for e in events:
        counts[e.verdict] = counts.get(e.verdict, 0) + 1
        if e.screened:
            phases[e.phase] = phases.get(e.phase, 0) + 1
            latencies.append(e.latency_ms)
        if e.verdict == "blocked" and e.end_user:
            blocked_per_user[e.end_user] = blocked_per_user.get(e.end_user, 0) + 1

    screened_total = counts["blocked"] + counts["flagged"] + counts["passed"]
    malicious_pct = 0.0
    if screened_total > 0:
        malicious_pct = round((counts["blocked"] + counts["flagged"]) / screened_total * 100, 1)
    avg_latency = round(sum(latencies) / len(latencies), 1) if latencies else 0.0

    high_risk = [u for u, n in blocked_per_user.items() if n >= HIGH_RISK_MIN_BLOCKED]

    return {
        "period": period,
        "total": len(events),
        "blocked": counts["blocked"],
        "flagged": counts["flagged"],
        "passed": counts["passed"],
        "bypassed": counts["bypassed"],
        "malicious_pct": malicious_pct,
        "avg_latency_ms": avg_latency,
        "high_risk_users": len(high_risk),
        "high_risk_user_list": sorted(high_risk),
        "phases": phases,
    }


@router.get("/trend")
def trend(
    period: str = "24h",
    key_id: int | None = None,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    size, count = BUCKETS[period] if period in BUCKETS else (None, None)
    if size is None:
        raise HTTPException(status_code=422, detail="period must be 1h, 24h, 7d or 30d")

    # buckets end "now"; bucket 0 is the oldest
    end = utc_now()
    start = end - size * count
    events = user_events(db, user, start, key_id).all()

    buckets = []
    for i in range(count):
        bucket_start = start + size * i
        buckets.append({"start": bucket_start.isoformat() + "Z", "attempts": 0, "blocked": 0})
    for e in events:
        i = int((e.created_at - start) / size)
        i = max(0, min(count - 1, i))
        buckets[i]["attempts"] += 1
        if e.verdict == "blocked":
            buckets[i]["blocked"] += 1
    return {"period": period, "buckets": buckets}


@router.get("/events")
def events_list(
    period: str = "30d",
    key_id: int | None = None,
    verdict: str | None = None,
    phase: str | None = None,
    q: str | None = None,
    limit: int = 100,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    query = user_events(db, user, get_since(period), key_id)
    if verdict:
        query = query.filter(ScreenEvent.verdict == verdict)
    if phase:
        query = query.filter(ScreenEvent.phase == phase)
    if q:
        like = "%" + q + "%"
        query = query.filter(
            ScreenEvent.prompt_preview.ilike(like) | ScreenEvent.end_user.ilike(like)
        )
    rows = query.order_by(ScreenEvent.created_at.desc(), ScreenEvent.id.desc()).limit(min(limit, 500)).all()
    return [
        {
            "id": e.id,
            "key_id": e.key_id,
            "time": e.created_at.isoformat() + "Z",
            "prompt_preview": e.prompt_preview,
            "end_user": e.end_user,
            "verdict": e.verdict,
            "phase": e.phase,
            "confidence": e.confidence,
            "latency_ms": e.latency_ms,
            "screened": e.screened,
        }
        for e in rows
    ]

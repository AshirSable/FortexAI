# Per-key limits for POST /v1/screen.
#   - requests per minute: sliding window kept in memory (resets on restart)
#   - requests per day / per month: counted from the screen_events table (survives restarts)
# Set a limit to 0 to turn it off.
import os
import threading
import time
from collections import defaultdict, deque
from datetime import timedelta

from fastapi import HTTPException
from sqlalchemy import func

from gateway.models import ScreenEvent, utc_now

PER_MINUTE = int(os.environ.get("FORTEX_RATE_LIMIT_PER_MIN", "60"))
PER_DAY = int(os.environ.get("FORTEX_DAILY_QUOTA", "10000"))
PER_MONTH = int(os.environ.get("FORTEX_MONTHLY_QUOTA", "200000"))

_hits = defaultdict(deque)  # key_id -> timestamps of calls in the last minute
_lock = threading.Lock()


def _too_many(detail, retry_after):
    return HTTPException(
        status_code=429,
        detail=detail,
        headers={"Retry-After": str(max(1, int(retry_after)))},
    )


def _day_start():
    return utc_now().replace(hour=0, minute=0, second=0, microsecond=0)


def _month_start():
    return _day_start().replace(day=1)


def _next_month_start():
    start = _month_start()
    return (start.replace(year=start.year + 1, month=1) if start.month == 12
            else start.replace(month=start.month + 1))


def _count_since(db, key_id, since):
    return (
        db.query(func.count(ScreenEvent.id))
        .filter(ScreenEvent.key_id == key_id, ScreenEvent.created_at >= since)
        .scalar()
    )


def get_usage(db, key_id):
    """Usage numbers the dashboard shows for one key."""
    now = utc_now()
    return {
        "daily_used": _count_since(db, key_id, _day_start()),
        "daily_limit": PER_DAY,
        "daily_resets_in_seconds": int((_day_start() + timedelta(days=1) - now).total_seconds()),
        "monthly_used": _count_since(db, key_id, _month_start()),
        "monthly_limit": PER_MONTH,
        "per_minute_limit": PER_MINUTE,
        "monthly_resets_in_seconds": int((_next_month_start() - now).total_seconds()),
    }


def check_rate_limit(db, key_id):
    """Raises 429 if this key is over its per-minute, daily or monthly limit; otherwise counts the call."""
    if PER_DAY > 0 and _count_since(db, key_id, _day_start()) >= PER_DAY:
        seconds_left = (_day_start() + timedelta(days=1) - utc_now()).total_seconds()
        raise _too_many(f"Daily limit of {PER_DAY} requests reached for this key", seconds_left)

    if PER_MONTH > 0 and _count_since(db, key_id, _month_start()) >= PER_MONTH:
        seconds_left = (_next_month_start() - utc_now()).total_seconds()
        raise _too_many(f"Monthly limit of {PER_MONTH} requests reached for this key", seconds_left)

    if PER_MINUTE > 0:
        now = time.monotonic()
        with _lock:
            window = _hits[key_id]
            while window and now - window[0] >= 60:
                window.popleft()
            if len(window) >= PER_MINUTE:
                raise _too_many(
                    f"Rate limit of {PER_MINUTE} requests per minute exceeded", 60 - (now - window[0])
                )
            window.append(now)

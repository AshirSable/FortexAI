# Database tables for API keys and screening events.
# They use the same Base as the auth tables, so init_db() creates them too.
from datetime import datetime, timezone

from auth.database import Base
from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String


def utc_now():
    # naive UTC time - SQLite does not keep timezones
    return datetime.now(timezone.utc).replace(tzinfo=None)


class ApiKey(Base):
    __tablename__ = "api_keys"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True, nullable=False)
    name = Column(String, nullable=False)
    key_prefix = Column(String, nullable=False)  # first chars, safe to show
    key_hash = Column(String, unique=True, index=True, nullable=False)
    status = Column(String, default="active", nullable=False)  # active / stopped
    created_at = Column(DateTime, default=utc_now)
    last_used_at = Column(DateTime, nullable=True)
    revoked = Column(Boolean, default=False, nullable=False)


class ScreenEvent(Base):
    __tablename__ = "screen_events"

    id = Column(Integer, primary_key=True, index=True)
    key_id = Column(Integer, ForeignKey("api_keys.id"), index=True, nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), index=True, nullable=False)
    created_at = Column(DateTime, default=utc_now, index=True)
    prompt_preview = Column(String, nullable=False)
    end_user = Column(String, nullable=True)
    verdict = Column(String, nullable=False)  # blocked / passed / flagged / bypassed
    phase = Column(String, nullable=False)
    confidence = Column(Float, default=0.0)  # 0 to 1
    latency_ms = Column(Float, default=0.0)
    screened = Column(Boolean, default=True, nullable=False)

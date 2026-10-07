# Endpoints for managing API keys (logged-in user only).
import hashlib
import secrets

from auth.database import get_db
from auth.models import User
from auth.router import get_current_user
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from gateway.models import ApiKey
from gateway.ratelimit import get_usage

router = APIRouter(prefix="/keys", tags=["keys"])


def hash_key(key):
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def key_to_dict(k):
    return {
        "id": k.id,
        "name": k.name,
        "key_prefix": k.key_prefix,
        "status": k.status,
        "created_at": k.created_at.isoformat() + "Z" if k.created_at else None,
        "last_used_at": k.last_used_at.isoformat() + "Z" if k.last_used_at else None,
    }


def get_own_key(db, user, key_id):
    key = (
        db.query(ApiKey)
        .filter(ApiKey.id == key_id, ApiKey.user_id == user.id, ApiKey.revoked == False)  # noqa: E712
        .first()
    )
    if key is None:
        raise HTTPException(status_code=404, detail="Key not found")
    return key


class CreateKeyRequest(BaseModel):
    name: str = "New Key"


class StatusRequest(BaseModel):
    status: str  # "active" or "stopped"


@router.post("", status_code=201)
def create_key(
    body: CreateKeyRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    full_key = "ftx_live_" + secrets.token_urlsafe(24)
    key = ApiKey(
        user_id=user.id,
        name=body.name.strip() or "New Key",
        key_prefix=full_key[:13],
        key_hash=hash_key(full_key),
    )
    db.add(key)
    db.commit()
    db.refresh(key)
    result = key_to_dict(key)
    result["key"] = full_key  # the only time the full key is shown
    return result


@router.get("")
def list_keys(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    keys = (
        db.query(ApiKey)
        .filter(ApiKey.user_id == user.id, ApiKey.revoked == False)  # noqa: E712
        .order_by(ApiKey.created_at.desc(), ApiKey.id.desc())
        .all()
    )
    return [{**key_to_dict(k), "usage": get_usage(db, k.id)} for k in keys]


@router.patch("/{key_id}/status")
def set_status(
    key_id: int,
    body: StatusRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    if body.status not in ("active", "stopped"):
        raise HTTPException(status_code=422, detail="status must be active or stopped")
    key = get_own_key(db, user, key_id)
    key.status = body.status
    db.commit()
    return key_to_dict(key)


@router.delete("/{key_id}")
def revoke_key(
    key_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    key = get_own_key(db, user, key_id)
    key.revoked = True
    db.commit()
    return {"detail": "Key revoked"}

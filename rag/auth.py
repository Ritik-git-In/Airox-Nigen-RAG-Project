"""File-based user accounts with securely hashed passwords.

Passwords are never stored in plain text. Each account keeps a random per-user
salt and a PBKDF2-HMAC-SHA256 hash; verification is constant-time. The store is
a JSON file under the (git-ignored) ``data/`` directory, so credentials never
leave the machine and are never committed.

This is intentionally dependency-free (standard library only). For a large,
multi-tenant production system you'd move this to a real database with a
battle-tested hasher (bcrypt/argon2), but PBKDF2 with a high iteration count is
solid for a self-hosted app.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import time
from pathlib import Path

from . import config, security

ITERATIONS = 200_000
MIN_PASSWORD_LEN = 8


def _users_file() -> Path:
    config.ensure_dirs()
    return config.DATA_DIR / "users.json"


def _load() -> dict:
    f = _users_file()
    if not f.exists():
        return {}
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except Exception:
        return {}  # corrupt file -> treat as empty rather than crash


def _save(users: dict) -> None:
    f = _users_file()
    tmp = f.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(users, indent=2), encoding="utf-8")
    tmp.replace(f)  # atomic-ish replace so a crash can't truncate the store


def _hash(password: str, salt_hex: str) -> str:
    dk = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), ITERATIONS
    )
    return dk.hex()


def user_exists(email: str) -> bool:
    return email.strip().lower() in _load()


def create_user(email: str, password: str) -> tuple[bool, str]:
    """Register a new account. Returns ``(ok, message)``."""
    email = email.strip().lower()
    if not security.is_valid_email(email):
        return False, "Please enter a valid email address."
    if len(password) < MIN_PASSWORD_LEN:
        return False, f"Password must be at least {MIN_PASSWORD_LEN} characters."
    users = _load()
    if email in users:
        return False, "An account with this email already exists. Try signing in."
    salt_hex = secrets.token_hex(16)
    users[email] = {
        "salt": salt_hex,
        "hash": _hash(password, salt_hex),
        "iterations": ITERATIONS,
        "created_at": int(time.time()),
    }
    _save(users)
    return True, "Account created."


def verify_user(email: str, password: str) -> bool:
    """Check an email/password pair (constant-time)."""
    email = email.strip().lower()
    rec = _load().get(email)
    if not rec:
        # Still do a dummy hash so timing doesn't reveal whether the email exists.
        _hash(password, secrets.token_hex(16))
        return False
    expected = rec.get("hash", "")
    actual = _hash(password, rec.get("salt", ""))
    return hmac.compare_digest(actual, expected)

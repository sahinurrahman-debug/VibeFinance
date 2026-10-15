"""Authentication primitives: password hashing, session tokens, input validation, rate limiting.

* Passwords are never stored: only `scrypt$N$r$p$salt$hash` (memory-hard, per-user random salt).
* Session cookies carry a random 256-bit token; the database stores only an HMAC-SHA256 of it
  (keyed with VF_SECRET_KEY), so a leaked database cannot be replayed as live sessions.
* Verification is constant-time and runs a dummy hash for unknown emails, so response time does not
  reveal which emails have accounts.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import os
import re
import secrets
import threading
import time
from collections import defaultdict, deque

from . import config

_EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9\-]+(\.[A-Za-z0-9\-]+)*\.[A-Za-z]{2,}$")
_MAXMEM = 128 * 1024 * 1024


def _b64(b: bytes) -> str:
    return base64.b64encode(b).decode()


def hash_password(password: str) -> str:
    n, r, p = config.SCRYPT_N, 8, 1
    salt = os.urandom(16)
    dk = hashlib.scrypt(password.encode(), salt=salt, n=n, r=r, p=p, dklen=32, maxmem=_MAXMEM)
    return f"scrypt${n}${r}${p}${_b64(salt)}${_b64(dk)}"


def _verify(password: str, stored: str) -> bool:
    try:
        scheme, n, r, p, salt, want = stored.split("$")
        if scheme != "scrypt":
            return False
        dk = hashlib.scrypt(password.encode(), salt=base64.b64decode(salt), n=int(n), r=int(r), p=int(p),
                            dklen=32, maxmem=_MAXMEM)
        return hmac.compare_digest(dk, base64.b64decode(want))
    except Exception:                                     # noqa: BLE001 - malformed hash must never raise to the caller
        return False


_DUMMY = None


def verify_password(password: str, stored: str | None) -> bool:
    """Constant-effort check: unknown accounts still pay for one scrypt computation."""
    global _DUMMY
    if stored is None:
        if _DUMMY is None:
            _DUMMY = hash_password("not-a-real-password")
        _verify(password, _DUMMY)
        return False
    return _verify(password, stored)


def new_token() -> str:
    return secrets.token_urlsafe(32)


def hash_token(token: str) -> str:
    return hmac.new(config.SECRET_KEY.encode(), token.encode(), hashlib.sha256).hexdigest()


def new_id() -> str:
    return secrets.token_hex(12)


# ----------------------------------------------------------------------------- validation
def normalise_email(email: str) -> str:
    return email.strip().lower()


def email_problem(email: str) -> str | None:
    e = normalise_email(email)
    if len(e) > 254 or not _EMAIL_RE.match(e):
        return "Enter a valid email address."
    return None


def password_problem(password: str) -> str | None:
    if len(password) < config.PASSWORD_MIN_LEN:
        return f"Password must be at least {config.PASSWORD_MIN_LEN} characters."
    if len(password) > 128:
        return "Password must be 128 characters or fewer."
    if len(set(password)) < 4:
        return "Choose a less repetitive password."
    if not (re.search(r"[A-Za-z]", password) and re.search(r"\d", password)):
        return "Password must contain at least one letter and one number."
    return None


# ----------------------------------------------------------------------------- rate limiting
class RateLimiter:
    """Sliding-window limiter, in-process. Fine for one instance; use Redis if you scale out."""

    def __init__(self):
        self._hits: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def _trim(self, q: deque, window: float, now: float):
        while q and now - q[0] > window:
            q.popleft()

    def check(self, key: str, limit: int, window: float) -> tuple[bool, int]:
        """Return (allowed, retry_after_seconds) WITHOUT recording a hit."""
        now = time.monotonic()
        with self._lock:
            q = self._hits[key]
            self._trim(q, window, now)
            if len(q) >= limit:
                return False, max(1, int(window - (now - q[0])) + 1)
        return True, 0

    def hit(self, key: str, window: float):
        now = time.monotonic()
        with self._lock:
            q = self._hits[key]
            self._trim(q, window, now)
            q.append(now)

    def consume(self, key: str, limit: int, window: float) -> tuple[bool, int]:
        ok, retry = self.check(key, limit, window)
        if ok:
            self.hit(key, window)
        return ok, retry

    def reset(self, key: str):
        with self._lock:
            self._hits.pop(key, None)


LIMITER = RateLimiter()

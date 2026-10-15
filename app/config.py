"""Central configuration.

* Values come from real environment variables first, then from an optional `.env` file in the project
  root (never committed; see `.env.example`). Real environment variables always win.
* `VF_ENV` selects the profile: `development` (default), `test` or `production`.
  Production refuses to start without a strong `VF_SECRET_KEY` and turns on secure cookies, HSTS and
  same-origin enforcement. Development generates a local secret into `data/` (git-ignored) so nothing
  secret ever lives in the repository.
"""
from __future__ import annotations

import os
import secrets
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    """Tiny .env reader (KEY=VALUE, '#' comments, optional quotes). Does not override real env vars."""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
            v = v[1:-1]
        os.environ.setdefault(k, v)


_load_dotenv(ROOT / ".env")


def _bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    return default if v is None or v == "" else v.strip().lower() in ("1", "true", "yes", "on")


# ----------------------------------------------------------------------------- environment profile
ENV = os.environ.get("VF_ENV", "development").strip().lower()
if ENV not in ("development", "test", "production"):
    raise RuntimeError(f"VF_ENV must be development, test or production (got {ENV!r})")
IS_PROD = ENV == "production"
IS_TEST = ENV == "test"

DATA_DIR = Path(os.environ.get("VF_DATA_DIR", ROOT / "data"))
LOG_DIR = Path(os.environ.get("VF_LOG_DIR", ROOT / "logs"))
DB_PATH = Path(os.environ.get("VF_DB", DATA_DIR / "vibefinance.db"))
LOG_LEVEL = os.environ.get("VF_LOG_LEVEL", "INFO" if IS_PROD else "DEBUG" if ENV == "development" else "WARNING").upper()

# ----------------------------------------------------------------------------- security
SECRET_KEY_ENV = os.environ.get("VF_SECRET_KEY", "")


def _dev_secret() -> str:
    """Stable per-machine secret for development, generated on first run and kept out of git."""
    f = DATA_DIR / "dev_secret.key"
    if not f.exists():
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(secrets.token_hex(32))
    return f.read_text().strip()


SECRET_KEY = SECRET_KEY_ENV or ("test-only-secret-key-not-for-real-use-0123456789" if IS_TEST else None)

COOKIE_NAME = os.environ.get("VF_COOKIE_NAME", "vf_session")
COOKIE_SECURE = _bool("VF_COOKIE_SECURE", IS_PROD)            # Secure flag: HTTPS only (default on in production)
COOKIE_SAMESITE = "lax"
SESSION_DAYS = float(os.environ.get("VF_SESSION_DAYS", "7"))
GUEST_HOURS = float(os.environ.get("VF_GUEST_HOURS", "24"))
MAX_GUESTS = int(os.environ.get("VF_MAX_GUESTS", "500"))
TRUST_PROXY = _bool("VF_TRUST_PROXY", IS_PROD)                # honour X-Forwarded-For (Render, Heroku, ...)
ALLOWED_ORIGINS = [o.strip().rstrip("/") for o in os.environ.get("VF_ALLOWED_ORIGINS", "").split(",") if o.strip()]
ENABLE_DOCS = _bool("VF_ENABLE_DOCS", not IS_PROD)            # /docs and /openapi.json
HSTS = IS_PROD

SCRYPT_N = int(os.environ.get("VF_SCRYPT_N", "1024" if IS_TEST else "16384"))
PASSWORD_MIN_LEN = 10

# ----------------------------------------------------------------------------- rate limits (count, window seconds)
LIMITS = {
    "login": (5, 15 * 60),          # failed attempts per IP+email
    "register": (10, 60 * 60),      # per IP
    "guest": (20, 60 * 60),         # per IP
    "analysis": (30, 60),           # per user
    "heavy": (6, 60),               # per user: benchmarks, live data
}
if IS_TEST:
    LIMITS = {**LIMITS, "analysis": (1000, 60), "heavy": (1000, 60), "guest": (1000, 3600), "register": (1000, 3600)}


def validate() -> None:
    """Fail fast on unsafe configuration. Called once at server start-up."""
    global SECRET_KEY
    if IS_PROD:
        if len(SECRET_KEY_ENV) < 32:
            raise RuntimeError("VF_SECRET_KEY must be set to a random string of at least 32 characters in production "
                               "(generate one with: python -c \"import secrets; print(secrets.token_hex(32))\").")
        if not COOKIE_SECURE:
            raise RuntimeError("VF_COOKIE_SECURE must not be disabled in production.")
    if SECRET_KEY is None:
        SECRET_KEY = _dev_secret()


# ----------------------------------------------------------------------------- data and engine
SEED = int(os.environ.get("VF_SEED", "42"))

# Simulated network latency for each feed (milliseconds). Scaled by VF_IO_SCALE.
# Set VF_IO_SCALE=0 for tests. The latencies exist so that parallel-vs-sequential
# agent execution can be measured honestly (pure-CPU agents finish in <1 ms).
IO_SCALE = float(os.environ.get("VF_IO_SCALE", "1.0"))
FEED_LATENCY_MS = {"prices": 70, "sentiment": 95, "filings": 80, "portfolio": 25}

# Orchestrator
AGENT_TIMEOUT_S = float(os.environ.get("VF_AGENT_TIMEOUT", "1.2"))
MAX_WORKERS = int(os.environ.get("VF_WORKERS", "48"))

# Backtest
FORWARD_DAYS = 30            # calendar days (= forward return horizon)
SIGNAL_THRESHOLD = 0.15      # |score| below this is "neutral" and not scored

DEGRADATIONS = {
    "price_feed_stale": "NSE price feed lags: the latest 3 sessions are missing",
    "price_feed_down": "NSE price feed unavailable (only last cached close exists)",
    "price_corrupt": "Price feed returns NaNs and a glitch spike in recent bars",
    "sentiment_feed_down": "News/sentiment feed down: fall back to a 3-day-old cached snapshot",
    "filings_unavailable": "Filing index unreachable: no disclosures can be retrieved",
    "slow_agent": "Sentiment agent hangs beyond the orchestrator timeout",
}

VECTOR_BACKEND = os.environ.get("VF_VECTOR", "auto")      # auto | qdrant | qdrant-memory | numpy
QDRANT_PATH = Path(os.environ.get("VF_QDRANT", DATA_DIR / "qdrant"))

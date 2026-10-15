"""Central configuration. Everything is overridable with environment variables."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
LOG_DIR = ROOT / "logs"
DB_PATH = Path(os.environ.get("VF_DB", DATA_DIR / "vibefinance.db"))

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

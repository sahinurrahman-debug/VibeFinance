"""Historical back-test of the three market-signal dimensions against 30-day forward returns.

Method: every 5 sessions, from session 130 onward, compute each dimension's signal using
ONLY data available at that date (the same `signals.py` functions the live agents call),
then compare sign(score) with the realised return over the next 30 calendar days.
Only non-neutral signals (|score| >= 0.15) are scored. No parameter is fitted on this
data, so the figures are out-of-sample in the weak sense that thresholds were fixed a priori.
Caveats shown in the UI: overlapping 30-day windows make samples serially correlated
(effective n is smaller than reported), and the data are simulated.

The per-(date, stock) signal grid is kept (`_grid`) so the strategy back-test and the confidence
calibration reuse exactly the same signals instead of recomputing them.
"""
from __future__ import annotations

import datetime as dt
import math

import numpy as np

from . import config
from . import signals as sg
from .market import MarketData, SYMBOLS

ENS_W = {"momentum": 0.40, "volume": 0.25, "sentiment": 0.35}
BUCKETS = [(0.0, 0.5), (0.5, 0.65), (0.65, 1.01)]


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - m) / d, (c + m) / d)


def _summ(rows: list[tuple[float, float, float]], base_up: float) -> dict:
    sc = [r for r in rows if abs(r[0]) >= config.SIGNAL_THRESHOLD]
    n = len(sc)
    hits = sum(1 for s, c, f in sc if np.sign(s) == np.sign(f))
    lo, hi = _wilson(hits, n)
    bull = [f for s, c, f in sc if s > 0]
    bear = [f for s, c, f in sc if s < 0]
    cal = []
    for a, b in BUCKETS:
        sub = [r for r in sc if a <= r[1] < b]
        k = sum(1 for s, c, f in sub if np.sign(s) == np.sign(f))
        cal.append({"range": f"{a:.2f}-{min(b, 1.0):.2f}", "lo": a, "hi": b, "n": len(sub), "hit_rate": (k / len(sub)) if sub else None,
                    "smoothed": (k + 1) / (len(sub) + 2)})
    return {"n_signals": n, "n_total": len(rows), "hit_rate": hits / n if n else None, "ci95": [lo, hi],
            "avg_fwd_bullish": float(np.mean(bull)) if bull else None,
            "avg_fwd_bearish": float(np.mean(bear)) if bear else None,
            "base_rate_up": base_up, "calibration": cal}


def ensemble(parts: dict) -> tuple[float, float]:
    """parts: {dim: (score, confidence)} -> (composite, confidence) using fixed ensemble weights."""
    wc = {k: ENS_W[k] * parts[k][1] for k in parts}
    comp = sum(wc[k] * parts[k][0] for k in parts) / sum(wc.values())
    spread = float(np.sqrt(sum(wc[k] * (parts[k][0] - comp) ** 2 for k in parts) / sum(wc.values())))
    conf = sum(wc.values()) / sum(ENS_W.values()) * (1 - 0.4 * min(1.0, spread / 0.8))
    return float(comp), float(conf)


def run_backtest(data: MarketData, step: int = 5, start: int = 130) -> dict:
    n = len(data.dates)
    recs: dict[str, list] = {"momentum": [], "volume": [], "sentiment": [], "ensemble": []}
    flat: list[dict] = []                                  # one record per (date, stock) for calibration
    grid: dict[int, dict[str, dict]] = {}
    ups, tot = 0, 0
    for sym in SYMBOLS:
        close, vol = data.close[sym], data.volume[sym]
        for t in range(start, n - 1, step):
            m = sg.momentum_signal(close[: t + 1])
            v = sg.volume_signal(close[: t + 1], vol[: t + 1])
            hl = data.headlines_between(sym, data.dates[t] - dt.timedelta(days=7), data.dates[t])
            s = sg.sentiment_signal(hl, data.dates[t], float(data.fii[t - 4: t + 1].sum()), float(data.pcr[sym][t]))
            parts = {"momentum": (m.score, m.confidence), "volume": (v.score, v.confidence), "sentiment": (s.score, s.confidence)}
            grid.setdefault(t, {})[sym] = parts
            target = data.dates[t] + dt.timedelta(days=config.FORWARD_DAYS)
            if target > data.dates[-1]:
                continue
            j = next(k for k in range(t + 1, n) if data.dates[k] >= target)
            fwd = float(close[j] / close[t] - 1)
            ups += fwd > 0
            tot += 1
            for k, (sc, cf) in parts.items():
                recs[k].append((sc, cf, fwd))
            comp, conf = ensemble(parts)
            recs["ensemble"].append((comp, conf, fwd))
            flat.append({"t": t, "sym": sym, "score": comp, "conf": conf, "fwd": fwd, "parts": parts})
    base = ups / tot if tot else 0.5
    out = {k: _summ(v, base) for k, v in recs.items()}
    out["_meta"] = {"horizon_days": config.FORWARD_DAYS, "sample_step_sessions": step, "n_obs": tot,
                    "universe": len(SYMBOLS), "threshold": config.SIGNAL_THRESHOLD, "base_rate_up": base,
                    "period": f"{data.dates[start].isoformat()} to {data.dates[-1].isoformat()}",
                    "caveats": ["Simulated data with mild built-in predictive structure: validates pipeline mechanics, not real-market alpha.",
                                "Overlapping 30-day windows -> serially correlated samples; effective n is smaller than reported.",
                                "Filings agent is not back-tested (corpus is a single as-of snapshot)."]}
    out["_records"] = flat
    out["_grid"] = grid
    return out


def public(bt: dict) -> dict:
    """Back-test summary without the internal per-observation arrays (safe to serialise)."""
    return {k: v for k, v in bt.items() if k not in ("_records", "_grid")}

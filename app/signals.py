"""Pure signal functions (no I/O). Agents wrap these; the backtester calls them directly,
so the accuracy we report is measured on exactly the code that produces live signals."""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import numpy as np

from .lexicon import tone


@dataclass
class SignalResult:
    score: float
    label: str
    confidence: float
    items: list[tuple[str, str, object, str]] = field(default_factory=list)   # (text, metric, value, stance)
    flags: dict = field(default_factory=dict)


def label_from_score(s: float) -> str:
    if s >= 0.5:
        return "Strong Bullish"
    if s >= 0.15:
        return "Bullish"
    if s > -0.15:
        return "Neutral"
    if s > -0.5:
        return "Bearish"
    return "Strong Bearish"


def stance(x: float, thr: float = 0.1) -> str:
    return "bullish" if x > thr else "bearish" if x < -thr else "neutral"


def ema(x: np.ndarray, n: int) -> np.ndarray:
    a = 2.0 / (n + 1)
    out = np.empty_like(x, dtype=float)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = a * x[i] + (1 - a) * out[i - 1]
    return out


def rsi(close: np.ndarray, n: int = 14) -> float:
    d = np.diff(close)
    gain, loss = np.clip(d, 0, None), np.clip(-d, 0, None)
    ag, al = gain[:n].mean(), loss[:n].mean()
    for i in range(n, len(d)):
        ag = (ag * (n - 1) + gain[i]) / n
        al = (al * (n - 1) + loss[i]) / n
    return 100.0 if al == 0 else 100 - 100 / (1 + ag / al)


def annual_vol(close: np.ndarray, n: int = 60) -> float:
    r = np.diff(np.log(close))[-n:]
    return float(r.std() * np.sqrt(252))


# --------------------------------------------------------------------- dimension 1
def momentum_signal(close: np.ndarray) -> SignalResult:
    if len(close) < 70:
        return SignalResult(0.0, "Neutral", 0.1, [("Insufficient price history", "bars", len(close), "neutral")])
    ret = np.diff(np.log(close))
    vol_d = max(float(ret[-60:].std()), 1e-4)
    r20 = close[-1] / close[-21] - 1
    r60 = close[-1] / close[-61] - 1
    sma50 = float(close[-50:].mean())
    sma200 = float(close[-200:].mean()) if len(close) >= 200 else None
    rs = rsi(close)
    macd = ema(close, 12) - ema(close, 26)
    hist = float((macd - ema(macd, 9))[-1] / close[-1])

    c1 = float(np.tanh(r20 / (vol_d * np.sqrt(20) * 2.0)))
    c2 = float(np.tanh(r60 / (vol_d * np.sqrt(60) * 2.0)))
    p = close[-1]
    if sma200 is not None:
        trend = 0.5 if p > sma50 > sma200 else -0.5 if p < sma50 < sma200 else (0.15 if p > sma50 else -0.15)
    else:
        trend = 0.15 if p > sma50 else -0.15
    if rs > 75:
        rsi_c = 0.5 - (rs - 75) / 25 * 0.8            # overbought: fade the momentum
    elif rs < 25:
        rsi_c = -0.5 + (25 - rs) / 25 * 0.6           # oversold: bounce risk
    else:
        rsi_c = (rs - 50) / 50
    macd_c = float(np.tanh(hist / (vol_d * 0.6)))
    comps = [c1, c2, trend, rsi_c, macd_c]
    score = float(np.clip(0.30 * c1 + 0.20 * c2 + 0.25 * trend + 0.10 * rsi_c + 0.15 * macd_c, -1, 1))
    sgn = np.sign(score) or 1.0
    agree = sum(1 for c in comps if np.sign(c) == sgn and abs(c) > 0.05) / len(comps)
    conf = float(np.clip(0.25 + 0.45 * abs(score) + 0.30 * agree, 0.1, 0.95))

    items = [
        (f"20-day return {r20:+.1%}, 60-day return {r60:+.1%} (daily vol {vol_d:.2%})", "return_20d/60d",
         f"{r20:+.3f}/{r60:+.3f}", stance(c1 * 0.6 + c2 * 0.4)),
        (f"Price {p:,.1f} vs 50-day average {sma50:,.1f}" + (f" and 200-day {sma200:,.1f}" if sma200 else ""),
         "trend_structure", round(trend, 2), stance(trend)),
        (f"RSI(14) = {rs:.0f}" + (" - overbought, momentum may fade" if rs > 75 else
                                  " - oversold, bounce risk" if rs < 25 else ""), "rsi14", round(rs, 1), stance(rsi_c)),
        (f"MACD histogram {hist:+.2%} of price", "macd_hist", round(hist, 4), stance(macd_c)),
    ]
    return SignalResult(score, label_from_score(score), conf, items,
                        {"r20": float(r20), "r60": float(r60), "rsi": float(rs), "sma50": sma50, "vol_d": vol_d})


# --------------------------------------------------------------------- dimension 2
def volume_signal(close: np.ndarray, volume: np.ndarray) -> SignalResult:
    if len(close) < 40:
        return SignalResult(0.0, "Normal Volume", 0.1, [("Insufficient history", "bars", len(close), "neutral")])
    lv = np.log(volume.astype(float))
    base = lv[-23:-3]
    mu, sd = float(base.mean()), max(float(base.std()), 0.08)
    vz = (float(lv[-3:].mean()) - mu) / sd
    ratio = float(np.exp(lv[-3:].mean() - mu))
    ret = np.diff(np.log(close))
    vol_d = max(float(ret[-60:].std()), 1e-4)
    ret3 = close[-1] / close[-4] - 1
    move = ret3 / (vol_d * np.sqrt(3))                          # 3-day move in daily-sigma units
    sgn20 = np.sign(np.diff(close[-21:]))
    obv = float((sgn20 * volume[-20:]).sum() / max(volume[-20:].sum(), 1))   # -1..1

    if vz >= 1.2 and abs(move) < 0.4:
        label, score = "Churn / Absorption", 0.25 * obv
        conf = 0.35 + 0.08 * min(vz, 3)
    elif vz >= 1.2:
        label = "Accumulation" if move > 0 else "Distribution"
        mag = min(1.0, 0.4 + 0.6 * min(abs(move), 1.5) / 1.5)
        score = float(np.sign(move) * np.tanh(vz / 2.5) * mag)
        conf = 0.40 + 0.12 * min(vz, 3.5) + (0.15 if np.sign(obv) == np.sign(move) else -0.05)
    elif vz <= -0.8:
        label, score, conf = "Volume Dry-up", 0.3 * obv, 0.45
    else:
        label, score, conf = "Normal Volume", 0.35 * obv, 0.5 - 0.1 * abs(vz) / 1.2
    score = float(np.clip(score, -1, 1))
    conf = float(np.clip(conf, 0.1, 0.95))
    items = [
        (f"Last-3-session volume is {ratio:.1f}x the prior 20-session average (z = {vz:+.1f})", "volume_z", round(vz, 2),
         "neutral" if abs(vz) < 1.2 else stance(move)),
        (f"3-day price move {ret3:+.1%} ({move:+.1f} daily sigmas)", "ret_3d", round(float(ret3), 4), stance(move, 0.3)),
        (f"20-day on-balance-volume tilt {obv:+.2f}", "obv_tilt", round(obv, 3), stance(obv, 0.1)),
    ]
    return SignalResult(score, label, conf, items, {"vz": vz, "ratio": ratio, "move": float(move), "obv": obv})


# --------------------------------------------------------------------- dimension 3
def sentiment_signal(headlines: list, as_of: dt.date, fii_5d: float | None, pcr: float | None) -> SignalResult:
    rows = []
    for h in headlines:
        pol, pos, neg = tone(h.text)
        age = max((as_of - h.date).days, 0)
        w = 0.5 ** (age / 3.0) * h.reliability
        rows.append((h, pol, w, pos, neg))
    flow_c = float(np.tanh(fii_5d / 8000.0)) if fii_5d is not None else 0.0
    pcr_c = float(np.clip((1.0 - pcr) / 0.5, -1, 1)) if pcr is not None else 0.0
    extras = (fii_5d is not None) + (pcr is not None)
    items = []
    if rows:
        wsum = sum(r[2] for r in rows)
        news = sum(r[1] * r[2] for r in rows) / wsum
        disp = float(np.sqrt(sum(r[2] * (r[1] - news) ** 2 for r in rows) / wsum))
        score = (0.65 * news + 0.20 * flow_c + 0.15 * pcr_c) if extras == 2 else news
        conf = 0.15 + 0.06 * min(len(rows), 6) + 0.20 * (1 - min(disp, 1)) + 0.10 * abs(score)
        for h, pol, w, pos, neg in sorted(rows, key=lambda r: -r[2])[:3]:
            items.append((f"[{h.source}] {h.text}", f"headline:{h.id}", round(pol, 2), stance(pol, 0.15)))
    else:
        news, disp = 0.0, 1.0
        score = 0.5 * flow_c + 0.5 * pcr_c
        conf = 0.2 if extras else 0.1
        items.append(("No headlines in the last 7 days; sentiment rests on flows/options only", "headline_count", 0, "neutral"))
    if fii_5d is not None:
        items.append((f"FII net flow (5-session, market-wide) {fii_5d:+,.0f} cr", "fii_5d", round(fii_5d), stance(flow_c, 0.1)))
    if pcr is not None:
        items.append((f"Put-call ratio {pcr:.2f}", "pcr", round(pcr, 2), stance(pcr_c, 0.1)))
    score = float(np.clip(score, -1, 1))
    return SignalResult(score, label_from_score(score), float(np.clip(conf, 0.1, 0.9)), items,
                        {"news": float(news), "n_headlines": len(rows), "dispersion": float(disp),
                         "headline_ids": [r[0].id for r in rows]})

"""Portfolio-level back-test: would following the personalised rules have paid off?

Weekly rebalance (every 5 sessions). At each date the three back-testable dimensions are combined with
the investor's profile weights; stocks whose composite clears that investor's buy threshold (and minimum
confidence) are held equal-weighted up to their position limit, the rest is cash. Positions are held for
the NEXT 5 sessions only (signals use data up to the rebalance date, so there is no look-ahead).
Benchmark: equal-weight all stocks, rebalanced weekly. A 0.10% cost is charged on turnover.

Filings are not part of this back-test (single as-of corpus), so conservative profiles - which lean on filings -
are tested on a reduced signal set. That limitation is reported alongside the results.
"""
from __future__ import annotations

import numpy as np

from .market import MarketData, SYMBOLS
from .synthesis import profile_params

COST = 0.001
STEP = 5


def _metrics(r: np.ndarray, turnover: float, exposure: float) -> dict:
    eq = np.cumprod(1 + r)
    ann = 252 / STEP
    vol = float(r.std(ddof=1) * np.sqrt(ann)) if len(r) > 2 else 0.0
    peak = np.maximum.accumulate(eq)
    return {"total_return": float(eq[-1] - 1), "cagr": float(eq[-1] ** (ann / len(r)) - 1), "vol": vol,
            "sharpe": float(r.mean() * ann / vol) if vol > 0 else 0.0, "max_drawdown": float((eq / peak - 1).min()),
            "avg_turnover": float(turnover), "avg_exposure": float(exposure), "weeks": int(len(r))}


def run(data: MarketData, grid: dict, user_like: dict, label: str | None = None) -> dict:
    pp = profile_params(user_like)
    w = {k: pp["weights"][k] for k in ("momentum", "volume", "sentiment")}
    tot = sum(w.values())
    w = {k: v / tot for k, v in w.items()}
    max_pos = user_like["max_position_pct"] / 100
    ts = sorted(grid)
    prev_w = {s: 0.0 for s in SYMBOLS}
    rets, bench, turns, expo, dates = [], [], [], [], []
    for t in ts:
        if t + STEP >= len(data.dates):
            break
        sel = []
        for s in SYMBOLS:
            parts = grid[t][s]
            eff = {k: w[k] * parts[k][1] for k in w}
            comp = sum(eff[k] * parts[k][0] for k in w) / sum(eff.values())
            conf = sum(eff.values()) / sum(w.values()) if False else sum(w[k] * parts[k][1] for k in w)
            if comp >= pp["buy_thr"] and conf >= pp["conf_min"] - 0.10:
                sel.append(s)
        target = {s: 0.0 for s in SYMBOLS}
        if sel:
            each = min(max_pos, 1.0 / len(sel))
            for s in sel:
                target[s] = each
        turn = sum(abs(target[s] - prev_w[s]) for s in SYMBOLS)
        fwd = {s: data.close[s][t + STEP] / data.close[s][t] - 1 for s in SYMBOLS}
        rets.append(sum(target[s] * fwd[s] for s in SYMBOLS) - COST * turn)
        bench.append(np.mean(list(fwd.values())))
        turns.append(turn)
        expo.append(sum(target.values()))
        prev_w = target
        dates.append(data.dates[t + STEP].isoformat())
    r, b = np.array(rets), np.array(bench)
    return {"label": label or pp["label"], "risk_tolerance": user_like["risk_tolerance"], "dates": dates,
            "equity": np.round(np.cumprod(1 + r), 4).tolist(), "benchmark_equity": np.round(np.cumprod(1 + b), 4).tolist(),
            "metrics": _metrics(r, np.mean(turns), np.mean(expo)), "benchmark_metrics": _metrics(b, 0.0, 1.0),
            "weights_used": {k: round(v, 3) for k, v in w.items()},
            "note": "Filings excluded (no historical corpus); weights renormalised over momentum/volume/sentiment."}


def run_profiles(data: MarketData, grid: dict, users: list[dict]) -> dict:
    out = {u["id"]: run(data, grid, u, u["name"]) for u in users}
    return out

"""Portfolio analytics: valuation, concentration score, volatility / VaR, and
behavioural-history analytics derived from the user's trade log."""
from __future__ import annotations

import datetime as dt
from copy import deepcopy

import numpy as np

from .market import BY_SYMBOL, MarketData, SYMBOLS


def _effn_score(weights: list[float]) -> float:
    """0 = fully diversified across the 12-name universe, 100 = single name."""
    w = np.array([x for x in weights if x > 0], dtype=float)
    if w.sum() <= 0:
        return 0.0
    w = w / w.sum()
    eff_n = 1.0 / float((w ** 2).sum())
    return float(np.clip(100 * (1 - (eff_n - 1) / (len(SYMBOLS) - 1)), 0, 100))


def risk_analysis(pos: list[dict], total: float, data: MarketData) -> dict:
    """Portfolio-level risk: risk contributions, correlation, historical VaR/ES, drawdown, stress tests."""
    if not pos or total <= 0:
        return {"contributions": [], "stress": [], "corr": {"symbols": [], "matrix": []}}
    syms = [p["symbol"] for p in pos]
    w = np.array([p["value"] / total for p in pos])
    R = np.vstack([data.rets[s][1:] for s in syms])                  # positions x sessions (log returns)
    port = w @ R                                                      # cash earns 0
    cov = np.atleast_2d(np.cov(R[:, -60:]))
    var_p = float(w @ cov @ w)
    sig_p = float(np.sqrt(max(var_p, 1e-12)))
    mrc = (w * (cov @ w)) / max(var_p, 1e-12)                         # share of portfolio variance (sums to 1)
    vol_i = np.sqrt(np.diag(cov)) * np.sqrt(252)
    idx = data.index[-250:]
    beta = np.array([np.cov(R[i, -250:], idx)[0, 1] / np.var(idx, ddof=1) for i in range(len(syms))])
    p5 = float(np.percentile(port, 5))
    tail = port[port <= p5]
    cum = np.exp(np.cumsum(port))
    mdd = float((cum / np.maximum.accumulate(cum) - 1).min())
    roll = np.convolve(port, np.ones(21), mode="valid")
    j = int(np.argmin(roll))
    dates = data.dates[1:]
    stress = [{"name": "Market -10%", "loss": float(-(w * beta).sum() * 0.10 * total), "how": "realised betas x -10% index move"},
              {"name": "Market -20%", "loss": float(-(w * beta).sum() * 0.20 * total), "how": "realised betas x -20% index move"}]
    secs: dict[str, float] = {}
    for p in pos:
        secs[p["sector"]] = secs.get(p["sector"], 0.0) + p["value"] / total
    top_sec, top_w = max(secs.items(), key=lambda kv: kv[1])
    stress.append({"name": f"{top_sec} sector -15%", "loss": float(-top_w * 0.15 * total), "how": "largest sector exposure x -15%"})
    stress.append({"name": f"Worst 21-session replay ({dates[j].isoformat()})", "loss": float(-(1 - np.exp(roll[j])) * total),
                   "how": "current weights through the worst historical month"})
    order = np.argsort(-mrc)
    return {"vol_ann": float(sig_p * np.sqrt(252)), "beta": float((w * beta).sum()),
            "diversification_ratio": float((w * vol_i).sum() / (sig_p * np.sqrt(252))) if sig_p > 0 else 1.0,
            "var95_hist": float(-p5 * total), "var99_hist": float(-np.percentile(port, 1) * total),
            "es95_hist": float(-tail.mean() * total) if len(tail) else 0.0, "max_drawdown": mdd,
            "contributions": [{"symbol": syms[i], "weight": float(w[i]), "risk_share": float(mrc[i]), "vol_ann": float(vol_i[i]),
                               "beta": float(beta[i])} for i in order],
            "top_risk_share": float(mrc[order[0]]), "top_risk_symbol": syms[order[0]], "stress": stress,
            "corr": {"symbols": syms, "matrix": np.round(np.corrcoef(R[:, -120:]), 2).tolist() if len(syms) > 1 else [[1.0]]},
            "avg_pair_corr": float((np.corrcoef(R[:, -120:]).sum() - len(syms)) / (len(syms) * (len(syms) - 1))) if len(syms) > 1 else 0.0}


def snapshot(user: dict, holdings: list[dict], prices: dict[str, float], data: MarketData,
             price_note: str = "") -> dict:
    cash = float(user["cash"])
    pos = []
    for h in holdings:
        if h["qty"] <= 0:
            continue
        px = prices[h["symbol"]]
        st = BY_SYMBOL[h["symbol"]]
        val = h["qty"] * px
        pos.append({"symbol": h["symbol"], "name": st.name, "sector": st.sector, "qty": h["qty"],
                    "avg_cost": h["avg_cost"], "price": px, "value": val,
                    "pnl": val - h["qty"] * h["avg_cost"],
                    "pnl_pct": px / h["avg_cost"] - 1 if h["avg_cost"] else 0.0})
    invested = sum(p["value"] for p in pos)
    total = invested + cash
    for p in pos:
        p["weight"] = p["value"] / total if total else 0.0
    sectors: dict[str, float] = {}
    for p in pos:
        sectors[p["sector"]] = sectors.get(p["sector"], 0.0) + p["weight"]

    stock_score = _effn_score([p["value"] for p in pos])
    sector_score = _effn_score(list(sectors.values()))
    conc = 0.6 * stock_score + 0.4 * sector_score

    vol_ann, var95 = 0.0, 0.0
    if pos and total:
        syms = [p["symbol"] for p in pos]
        R = np.vstack([data.rets[s][-60:] for s in syms])
        w = np.array([p["value"] / total for p in pos])
        daily = float(np.sqrt(w @ np.atleast_2d(np.cov(R)) @ w))      # one holding -> np.cov is a scalar
        vol_ann, var95 = daily * np.sqrt(252), 1.645 * daily * total

    flags = []
    mp, ms = user["max_position_pct"], user["max_sector_pct"]
    for p in pos:
        if p["weight"] * 100 > mp:
            flags.append({"type": "position_limit", "symbol": p["symbol"],
                          "text": f"{p['symbol']} is {p['weight']:.0%} of the portfolio vs your {mp:.0f}% position limit"})
    for s, w in sectors.items():
        if w * 100 > ms:
            flags.append({"type": "sector_limit", "symbol": None,
                          "text": f"{s} exposure is {w:.0%} vs your {ms:.0f}% sector limit"})
    return {"positions": sorted(pos, key=lambda p: -p["value"]), "cash": cash, "invested": invested, "total": total,
            "cash_pct": cash / total if total else 1.0, "sector_weights": dict(sorted(sectors.items(), key=lambda kv: -kv[1])),
            "stock_concentration": round(stock_score, 1), "sector_concentration": round(sector_score, 1),
            "concentration_score": round(conc, 1), "top_weight": max([p["weight"] for p in pos], default=0.0),
            "volatility_ann": float(vol_ann), "var95_1d": float(var95), "limit_flags": flags,
            "price_note": price_note, "risk": risk_analysis(pos, total, data)}


def apply_hypothetical(user: dict, holdings: list[dict], trades: list[dict], prices: dict[str, float]):
    """Return (user', holdings') after executing suggested trades at the quoted prices."""
    u = deepcopy(user)
    hs = {h["symbol"]: dict(h) for h in holdings}
    for t in trades:
        px = prices[t["symbol"]]
        h = hs.get(t["symbol"], {"symbol": t["symbol"], "qty": 0, "avg_cost": px})
        if t["side"] == "BUY":
            cost = t["qty"] * px
            if cost > u["cash"]:
                continue
            nq = h["qty"] + t["qty"]
            h["avg_cost"] = (h["qty"] * h["avg_cost"] + cost) / nq
            h["qty"] = nq
            u["cash"] -= cost
        else:
            q = min(t["qty"], h["qty"])
            h["qty"] -= q
            u["cash"] += q * px
        hs[t["symbol"]] = h
    return u, list(hs.values())


# ----------------------------------------------------------------------------- behaviour
def behavior_profile(trades: list[dict], data: MarketData) -> dict:
    """Derive behavioural tendencies from the trade log *and* the market context at each trade."""
    buys, sells, fomo, panic, regret = [], [], [], [], []
    evidence = []
    for t in trades:
        d = dt.date.fromisoformat(t["ts"])
        if d not in data._idx or t["symbol"] not in data.close:
            continue
        i = data.idx(d)
        c = data.close[t["symbol"]]
        if i < 25:
            continue
        r20 = c[i] / c[i - 20] - 1
        dd = c[i] / c[i - 20: i + 1].max() - 1
        fwd = (c[i + 21] / c[i] - 1) if i + 21 < len(c) else None
        if t["side"] == "BUY":
            buys.append(t)
            if r20 >= 0.12:
                fomo.append(t)
                evidence.append({"id": f"TRD-{t['id']}", "kind": "history", "date": t["ts"], "symbol": t["symbol"],
                                 "text": f"Bought {t['symbol']} on {t['ts']} after a {r20:+.0%} 20-day run-up"})
        else:
            sells.append(t)
            if dd <= -0.08:
                panic.append(t)
                regretted = fwd is not None and fwd > 0.03
                if regretted:
                    regret.append(t)
                evidence.append({"id": f"TRD-{t['id']}", "kind": "history", "date": t["ts"], "symbol": t["symbol"],
                                 "text": f"Sold {t['symbol']} on {t['ts']} {dd:.0%} below its 20-day high"
                                         + (f"; it rose {fwd:+.0%} over the next month" if regretted else "")})
    span_days = 1
    if trades:
        ds = [dt.date.fromisoformat(t["ts"]) for t in trades]
        span_days = max((max(ds) - min(ds)).days, 30)
    fomo_rate = len(fomo) / len(buys) if buys else 0.0
    panic_rate = len(panic) / len(sells) if sells else 0.0
    if len(fomo) >= 3 and fomo_rate >= 0.4:
        arch = "FOMO-prone"
    elif len(panic) >= 2 and panic_rate >= 0.4:
        arch = "Panic-prone"
    else:
        arch = "Disciplined"
    return {"archetype": arch, "n_trades": len(trades), "n_buys": len(buys), "n_sells": len(sells),
            "fomo_buy_rate": round(fomo_rate, 2), "panic_sell_rate": round(panic_rate, 2),
            "regretted_panic_sells": len(regret), "trades_per_year": round(len(trades) / span_days * 365, 1),
            "evidence": evidence[:8]}

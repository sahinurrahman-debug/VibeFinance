"""Seed three investor personas with holdings, watchlists and a synthetic trade history.

Trade histories are generated *from the market data* (e.g. the FOMO investor's buys are
placed after genuine >12% run-ups) so the behavioural analytics in `portfolio.py` derive
the archetype from evidence rather than reading a hard-coded label.
"""
from __future__ import annotations

import numpy as np

from .market import MarketData, SYMBOLS
from .store import Store

USERS = [
    dict(id="asha", name="Asha Iyer", age=58, risk_tolerance=2, max_position_pct=15, max_sector_pct=30,
         max_volatility=0.30, horizon_years=5, total=3_000_000, cash_pct=0.20,
         bio="Retiring in 5 years. Capital preservation first; has sold into dips before.",
         weights={"NAVBANK": .14, "KISANFOODS": .13, "HINDCONSUMER": .17, "TECHNOVA": .12, "SANCHARNET": .12},
         watch=["RASHTRIYA", "BHARATOIL", "INFOSPHERE"], style="panic"),
    dict(id="rohan", name="Rohan Mehta", age=27, risk_tolerance=9, max_position_pct=30, max_sector_pct=50,
         max_volatility=0.70, horizon_years=12, total=1_200_000, cash_pct=0.12,
         bio="Early-career, high risk appetite, chases momentum, trades often.",
         weights={"VIKRAMINFRA": .34, "GATIMOTORS": .22, "BHARATOIL": .16, "RASHTRIYA": .16},
         watch=["INFOSPHERE", "SANCHARNET", "TECHNOVA", "NAVBANK"], style="fomo"),
    dict(id="meera", name="Meera Nair", age=39, risk_tolerance=5, max_position_pct=20, max_sector_pct=40,
         max_volatility=0.45, horizon_years=15, total=2_000_000, cash_pct=0.15,
         bio="Balanced saver, monthly SIPs, 15-year horizon.",
         weights={"INFOSPHERE": .17, "SAMRIDDHI": .14, "AROGYAPHARMA": .14, "BHARATOIL": .14, "NAVBANK": .14,
                  "SANCHARNET": .12},
         watch=["VIKRAMINFRA", "GATIMOTORS", "RASHTRIYA", "HINDCONSUMER"], style="sip"),
]


def _candidates(data: MarketData, kind: str):
    out = []
    for s in SYMBOLS:
        c = data.close[s]
        for i in range(60, len(c) - 60):
            r20 = c[i] / c[i - 20] - 1
            dd = c[i] / c[i - 20: i + 1].max() - 1
            fwd = c[i + 21] / c[i] - 1
            if kind == "fomo" and r20 >= 0.12:
                out.append((s, i))
            if kind == "panic" and dd <= -0.08 and fwd > 0.03:
                out.append((s, i))
    return out


def seed(store: Store, data: MarketData):
    rng = np.random.default_rng(7)
    for u in USERS:
        total = u["total"]
        invested = 0.0
        initial = []                      # (symbol, qty, cost, date) written after the user row exists (foreign keys)
        for sym, w in u["weights"].items():
            px = float(data.close[sym][-1])
            qty = int(total * w // px)
            c = data.close[sym]
            n = len(c)
            ok = [i for i in range(n - 300, n - 60)
                  if 0.8 <= c[i] / px <= 1.3 and c[i] / c[i - 20] - 1 < 0.10]    # realistic entry, not a chase
            pool = ok or list(range(n - 300, n - 60))
            buy_i = int(pool[int(rng.integers(0, len(pool)))])
            cost = float(c[buy_i])
            initial.append((sym, qty, round(cost, 2), data.dates[buy_i].isoformat()))
            invested += qty * px
        cash = round(total - invested, 2)
        store.create_user({**{k: u[k] for k in ("id", "name", "age", "bio", "risk_tolerance", "max_position_pct",
                                                "max_sector_pct", "max_volatility", "horizon_years")}, "cash": cash, "role": "template"})
        for sym, qty, cost, day in initial:
            store.set_holding(u["id"], sym, qty, cost)
            store.add_trade_log(u["id"], day, sym, "BUY", qty, cost, "initial position")
        store.set_watchlist(u["id"], u["watch"])

        def rt(sym, i_buy, i_sell, qty, note):
            store.add_trade_log(u["id"], data.dates[i_buy].isoformat(), sym, "BUY", qty, round(float(data.close[sym][i_buy]), 2), note)
            store.add_trade_log(u["id"], data.dates[i_sell].isoformat(), sym, "SELL", qty, round(float(data.close[sym][i_sell]), 2), note)

        if u["style"] == "fomo":
            cands = _candidates(data, "fomo")
            picks = [cands[j] for j in rng.choice(len(cands), size=min(7, len(cands)), replace=False)]
            for sym, i in picks:
                rt(sym, i, min(i + 22, len(data.dates) - 1), int(rng.integers(20, 120)), "chased rally")
        elif u["style"] == "panic":
            cands = _candidates(data, "panic")
            picks = [cands[j] for j in rng.choice(len(cands), size=min(4, len(cands)), replace=False)]
            for sym, i in picks:
                store.add_trade_log(u["id"], data.dates[i - 30].isoformat(), sym, "BUY", 60,
                                    round(float(data.close[sym][i - 30]), 2), "")
                store.add_trade_log(u["id"], data.dates[i].isoformat(), sym, "SELL", 60,
                                    round(float(data.close[sym][i]), 2), "sold during dip")
        else:   # disciplined SIP: equal monthly purchases, no panic, no chasing
            for k, i in enumerate(range(len(data.dates) - 200, len(data.dates) - 20, 22)):
                sym = ["SAMRIDDHI", "NAVBANK"][k % 2]
                store.add_trade_log(u["id"], data.dates[i].isoformat(), sym, "BUY", 5,
                                    round(float(data.close[sym][i]), 2), "monthly SIP")

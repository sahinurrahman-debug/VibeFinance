"""Persistent investor sessions: what changed since the last time this user looked.

Each analysis stores a snapshot (portfolio composition, behavioural metrics, every action
recommended). The timeline diffs consecutive snapshots - positions opened / closed / resized,
cash and concentration movement, recommendation flips, behavioural drift - and joins them with
the user's recorded decisions on earlier advice.
"""
from __future__ import annotations


def make_snapshot(user: dict, snap: dict, behavior: dict, recs: list[dict], alert_ids: list[str],
                  decisions: list[dict]) -> dict:
    acc = sum(1 for d in decisions if d["choice"] == "accepted")
    return {
        "positions": {p["symbol"]: {"qty": p["qty"], "weight": round(p["weight"], 4), "value": round(p["value"], 2)}
                      for p in snap["positions"]},
        "cash": round(snap["cash"], 2), "total": round(snap["total"], 2),
        "concentration": snap["concentration_score"], "risk_tolerance": user["risk_tolerance"],
        "behavior": {"archetype": behavior["archetype"], "fomo_buy_rate": behavior["fomo_buy_rate"],
                     "panic_sell_rate": behavior["panic_sell_rate"], "n_trades": behavior["n_trades"],
                     "trades_per_year": behavior["trades_per_year"], "decisions": len(decisions), "accepted": acc},
        "actions": {r["symbol"]: {"action": r["action"], "confidence": r["confidence"], "composite": r["composite"]} for r in recs},
        "alert_ids": alert_ids,
    }


def diff(prev: dict, cur: dict) -> dict:
    pp, cp = prev["positions"], cur["positions"]
    pos = []
    for s in sorted(set(pp) | set(cp)):
        a, b = pp.get(s), cp.get(s)
        if a is None:
            pos.append({"symbol": s, "change": "opened", "qty": b["qty"]})
        elif b is None:
            pos.append({"symbol": s, "change": "closed", "qty": -a["qty"]})
        elif a["qty"] != b["qty"]:
            pos.append({"symbol": s, "change": "resized", "qty": b["qty"] - a["qty"]})
    flips = [{"symbol": s, "from": prev["actions"][s]["action"], "to": v["action"]}
             for s, v in cur["actions"].items() if s in prev["actions"] and prev["actions"][s]["action"] != v["action"]]
    pb, cb = prev["behavior"], cur["behavior"]
    return {"positions": pos, "cash_delta": round(cur["cash"] - prev["cash"], 2),
            "total_delta": round(cur["total"] - prev["total"], 2),
            "concentration_delta": round(cur["concentration"] - prev["concentration"], 1),
            "recommendation_changes": flips,
            "behavior_changes": [] if pb["archetype"] == cb["archetype"] and pb["n_trades"] == cb["n_trades"] else
            [f"archetype {pb['archetype']} -> {cb['archetype']}" if pb["archetype"] != cb["archetype"] else
             f"{cb['n_trades'] - pb['n_trades']} new trade(s) logged; FOMO-buy rate {pb['fomo_buy_rate']:.0%} -> {cb['fomo_buy_rate']:.0%}, "
             f"panic-sell rate {pb['panic_sell_rate']:.0%} -> {cb['panic_sell_rate']:.0%}"],
            "new_alerts": [a for a in cur["alert_ids"] if a not in prev["alert_ids"]],
            "cleared_alerts": [a for a in prev["alert_ids"] if a not in cur["alert_ids"]],
            "rules_changed": prev["risk_tolerance"] != cur["risk_tolerance"]}


def decision_stats(decisions: list[dict]) -> dict:
    by: dict[str, dict] = {}
    for d in decisions:
        b = by.setdefault(d["action"], {"accepted": 0, "ignored": 0, "other": 0})
        b[d["choice"] if d["choice"] in b else "other"] += 1
    n = len(decisions)
    acc = sum(1 for d in decisions if d["choice"] == "accepted")
    return {"total": n, "accepted": acc, "accept_rate": round(acc / n, 3) if n else None, "by_action": by}


def build(snapshots: list[dict], decisions: list[dict], sessions: list[dict]) -> dict:
    """snapshots oldest-first. Returns timeline entries newest-first with diffs against the previous snapshot."""
    entries = []
    for i, sn in enumerate(snapshots):
        e = {"session_id": sn["session_id"], "ts": sn["ts"], "total": sn["payload"]["total"],
             "concentration": sn["payload"]["concentration"], "archetype": sn["payload"]["behavior"]["archetype"],
             "n_alerts": len(sn["payload"]["alert_ids"]), "diff": diff(snapshots[i - 1]["payload"], sn["payload"]) if i else None}
        e["decisions"] = [d for d in decisions if d.get("session_id") == sn["session_id"]]
        entries.append(e)
    return {"entries": entries[::-1], "decision_stats": decision_stats(decisions),
            "behavior_series": [{"ts": s["ts"], **s["payload"]["behavior"]} for s in snapshots]}

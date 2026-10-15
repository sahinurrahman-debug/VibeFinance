"""Personalised alerts for holdings and the watchlist.

Every alert is produced by a per-user rule (stored in SQLite, editable in the UI), carries the
rule that fired and why it matters to *this* investor, and is tagged new / ongoing against the
previous analysis session so the user is told what changed rather than shown the same list again.
"""
from __future__ import annotations

DEFAULT_RULES = {
    "min_confidence": 0.55,           # signal alerts need at least this confidence
    "volume_z": 2.0,                  # |volume z-score| for an unusual-volume alert
    "daily_move_pct": 4.0,            # |1-day move| for a price-move alert
    "adverse_filings": 2.0,           # number of adverse filing passages
    "max_risk_share_pct": 40.0,       # one holding explaining more than this share of portfolio risk
    "max_concentration": 60.0,        # portfolio concentration score
    "max_loss_10pct_mkt_pct": 8.0,    # loss in a -10% market as % of portfolio
}
RULE_HELP = {
    "min_confidence": "Minimum confidence for buy/sell signal alerts (0-1)",
    "volume_z": "Volume anomaly threshold (z-score)",
    "daily_move_pct": "Daily price move that triggers an alert (%)",
    "adverse_filings": "Adverse filing passages needed to alert",
    "max_risk_share_pct": "Max share of portfolio risk from one holding (%)",
    "max_concentration": "Max portfolio concentration score (0-100)",
    "max_loss_10pct_mkt_pct": "Max tolerated loss in a -10% market (% of portfolio)",
}
SEV = {"high": 0, "medium": 1, "low": 2}


def _a(kind, symbol, severity, text, why, scope=None, sub=""):
    return {"id": f"{kind}:{symbol or '-'}:{sub}", "kind": kind, "symbol": symbol, "severity": severity, "text": text,
            "why": why, "scope": scope}


def evaluate(user: dict, rules: dict, snap: dict, recs: list[dict], watch: list[str], signals: list[dict],
             prev_ids: list[str] | None = None) -> dict:
    held = {p["symbol"] for p in snap["positions"]}
    sig = {s["symbol"]: s for s in signals}
    out: list[dict] = []
    for f in snap["limit_flags"]:
        out.append(_a("limit", f["symbol"], "high", f["text"], "Your position/sector limit from your risk profile",
                      "holding", f["type"]))
    risk = snap.get("risk") or {}
    if risk.get("top_risk_share", 0) * 100 > rules["max_risk_share_pct"] and len(snap["positions"]) > 1:
        out.append(_a("risk_share", risk["top_risk_symbol"], "high",
                      f"{risk['top_risk_symbol']} drives {risk['top_risk_share']:.0%} of portfolio risk "
                      f"(vs {rules['max_risk_share_pct']:.0f}% rule) though it is only "
                      f"{next(c['weight'] for c in risk['contributions'] if c['symbol'] == risk['top_risk_symbol']):.0%} of value",
                      "Risk contribution, not just weight, counts", "holding"))
    if snap["concentration_score"] > rules["max_concentration"]:
        out.append(_a("concentration", None, "medium", f"Concentration score {snap['concentration_score']:.0f} exceeds your {rules['max_concentration']:.0f} rule",
                      "Few names / sectors carry the portfolio", "portfolio"))
    for st in risk.get("stress", [])[:1]:
        pct = -st["loss"] / snap["total"] * 100 if snap["total"] else 0
        if pct > rules["max_loss_10pct_mkt_pct"]:
            out.append(_a("stress", None, "medium", f"A 10% market fall would cost ~{pct:.1f}% of the portfolio (rule {rules['max_loss_10pct_mkt_pct']:.0f}%)",
                          "Portfolio beta is high for your tolerance", "portfolio"))
    for r in recs:
        s = r["symbol"]
        scope = "holding" if s in held else "watchlist" if s in watch else None
        if scope is None:
            continue
        tag = f"{s} ({scope})"
        conf_ok = r["confidence"] >= rules["min_confidence"]
        if r["action"] in ("TRIM", "SELL") and s in held and conf_ok:
            out.append(_a("signal_sell", s, "high", f"{tag}: {r['action']} - composite {r['composite']:+.2f}, confidence {r['confidence']:.0%}",
                          f"Confidence above your {rules['min_confidence']:.0%} rule", scope))
        elif r["action"] in ("BUY", "ACCUMULATE") and r["sizing"] and conf_ok:
            out.append(_a("signal_buy", s, "medium", f"{tag}: {r['action']} setup - composite {r['composite']:+.2f}, confidence {r['confidence']:.0%}",
                          f"Clears your buy threshold {r['thresholds']['buy_thr']:.2f} and risk limits", scope))
        if r["conflicts"]:
            out.append(_a("conflict", s, "medium", f"{tag}: agents disagree - {r['conflicts'][0]['text']}",
                          "Disagreement lowers confidence; review before acting", scope))
        vol = r["agents"]["volume"]
        if vol["status"] in ("ok", "degraded") and abs(vol["data"].get("vz", 0)) >= rules["volume_z"] and vol["label"] in ("Accumulation", "Distribution"):
            out.append(_a("volume", s, "high" if vol["label"] == "Distribution" and s in held else "medium",
                          f"{tag}: {vol['label'].lower()} volume {vol['data']['ratio']:.1f}x normal (z {vol['data']['vz']:+.1f})",
                          f"Volume z-score beyond your {rules['volume_z']:.1f} rule", scope))
        fil = r["agents"]["filings"]
        if fil["status"] == "ok" and fil["data"]["n_adverse"] >= rules["adverse_filings"]:
            out.append(_a("filing", s, "medium", f"{tag}: {fil['data']['n_adverse']} adverse passages in recent filings",
                          f"At least {rules['adverse_filings']:.0f} adverse passages", scope))
        chg = sig.get(s, {}).get("chg_1d")
        if chg is not None and abs(chg) * 100 >= rules["daily_move_pct"]:
            out.append(_a("move", s, "medium", f"{tag}: moved {chg:+.1%} today", f"Beyond your {rules['daily_move_pct']:.0f}% daily-move rule", scope))
    seen, uniq = set(), []
    for a in out:
        if a["id"] not in seen:
            seen.add(a["id"])
            uniq.append(a)
    prev = set(prev_ids or [])
    for a in uniq:
        a["is_new"] = bool(prev) and a["id"] not in prev
    resolved = sorted(prev - seen)
    uniq.sort(key=lambda a: (SEV[a["severity"]], not a["is_new"]))
    return {"alerts": uniq, "resolved_ids": resolved, "ids": [a["id"] for a in uniq]}

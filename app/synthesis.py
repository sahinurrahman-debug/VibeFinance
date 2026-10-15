"""Synthesis layer: merges the six agent outputs into one personalised, cited recommendation.

Pipeline per (user, stock):
  1. weight the four signal agents by profile x agent confidence -> composite score
  2. measure agreement; detect and explain cross-agent conflicts
  3. calibrate confidence (agreement, coverage, grounding, empirical back-test reliability)
  4. map to an action with profile-dependent thresholds
  5. apply the risk gate (volatility / position / sector limits) and behavioural nudges
  6. size the trade and emit the full reasoning chain with source references
"""
from __future__ import annotations

import math

import numpy as np

from .contracts import AgentOutput, Citation
from .calibration import counterfactuals
from .market import get_stock

SIGNALS = ("momentum", "volume", "sentiment", "filings")
W_CONS = {"momentum": 0.15, "volume": 0.10, "sentiment": 0.15, "filings": 0.60}
W_AGGR = {"momentum": 0.38, "volume": 0.22, "sentiment": 0.15, "filings": 0.25}
NICE = {"momentum": "Momentum", "volume": "Volume", "sentiment": "Sentiment", "filings": "Filings (RAG)"}


def profile_params(u: dict) -> dict:
    tol = int(u["risk_tolerance"])
    t = (tol - 1) / 8
    w = {a: (1 - t) * W_CONS[a] + t * W_AGGR[a] for a in SIGNALS}
    s = sum(w.values())
    buy = 0.45 - 0.035 * tol
    sell = -(0.10 + 0.03 * tol)
    return {"tolerance": tol, "label": "Conservative" if tol <= 3 else "Moderate" if tol <= 6 else "Aggressive",
            "weights": {a: w[a] / s for a in SIGNALS}, "buy_thr": buy, "strong_buy": buy + 0.25,
            "sell_thr": sell, "strong_sell": sell - 0.20, "conf_min": 0.62 - 0.03 * tol}


def _top_refs(o: AgentOutput, n: int = 3) -> list[str]:
    return list(dict.fromkeys(e.source_id for e in o.evidence if e.source_id))[:n]


def synthesize(user: dict, symbol: str, outs: dict[str, AgentOutput], price: float, snapshot: dict,
               calibrator, degr=()) -> dict:
    st = get_stock(symbol)
    pp = profile_params(user)
    W = pp["weights"]
    sig = {a: outs[a] for a in SIGNALS}
    usable = {a: o for a, o in sig.items() if o.usable}
    risk, beh = outs["risk"], outs["behavior"]
    pos = next((p for p in snapshot["positions"] if p["symbol"] == symbol), None)
    held = pos is not None
    warnings = [w for o in outs.values() for w in o.warnings]

    # 1. composite --------------------------------------------------------
    eff = {a: W[a] * o.confidence for a, o in usable.items()}
    tot = sum(eff.values())
    composite = sum(eff[a] * usable[a].score for a in usable) / tot if tot > 0 else 0.0
    contribs = [{"agent": a, "weight": round(W[a], 3), "score": round(o.score, 3), "confidence": round(o.confidence, 3),
                 "contribution": round(eff[a] * o.score / tot, 3) if tot else 0.0, "status": o.status,
                 "label": o.label} for a, o in sig.items() if a in usable]
    contribs += [{"agent": a, "weight": round(W[a], 3), "score": 0.0, "confidence": 0.0, "contribution": 0.0,
                  "status": o.status, "label": o.label} for a, o in sig.items() if a not in usable]
    spread = math.sqrt(sum(eff[a] * (usable[a].score - composite) ** 2 for a in usable) / tot) if tot else 0.0

    # 2. conflicts --------------------------------------------------------
    conflicts = []
    names = list(usable)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            oa, ob = usable[a], usable[b]
            if oa.score * ob.score < 0 and min(abs(oa.score), abs(ob.score)) > 0.25 and min(oa.confidence, ob.confidence) >= 0.4:
                bull, bear = (a, b) if oa.score > 0 else (b, a)
                conflicts.append({"bullish": bull, "bearish": bear,
                                  "text": f"{NICE[bull]} is {usable[bull].label} ({usable[bull].score:+.2f}) while "
                                          f"{NICE[bear]} is {usable[bear].label} ({usable[bear].score:+.2f})"})
    explanation = ""
    if conflicts:
        c = conflicts[0]
        heavier = max((c["bullish"], c["bearish"]), key=lambda x: W[x])
        explanation = (f"{c['text']}. For the {pp['label'].lower()} profile {NICE[heavier]} carries the larger weight "
                       f"({W[heavier]:.0%}), so the composite lands at {composite:+.2f}. "
                       f"Disagreement lowers confidence rather than being averaged away.")

    # 3. confidence -------------------------------------------------------
    base_conf = sum(W[a] * o.confidence for a, o in usable.items())          # missing agents contribute 0
    agreement = 1 - min(1.0, spread / 0.6)
    conf = base_conf * (0.7 + 0.3 * agreement) * (0.8 if conflicts else 1.0)
    grounded = "filings" in usable
    if not grounded:
        conf = min(conf, 0.60)
        warnings.append("Not grounded in filings: confidence capped at 60% and position size halved")
    conf = float(np.clip(conf, 0.0, 0.95))
    lo, hi = max(-1.0, composite - spread), min(1.0, composite + spread)
    cal = calibrator.lookup(conf) if calibrator else None
    pred = calibrator.predict(conf) if calibrator else None
    cfs = counterfactuals(usable, W, pp["buy_thr"], pp["sell_thr"], composite) if len(usable) > 1 else []
    decisive = [c for c in cfs if c["decisive"]]
    dq = 1 - sum(W[a] for a in usable) / sum(W.values())
    dq += 0.5 * sum(W[a] for a, o in usable.items() if o.status == "degraded") / sum(W.values())
    ci_w = (pred["hi"] - pred["lo"]) if pred else 0.0
    u_score = 0.5 * min(1.0, spread / 0.5) + 0.3 * min(1.0, dq * 2) + 0.2 * min(1.0, ci_w / 0.3)
    uncertainty = {"level": "High" if u_score > 0.5 else "Medium" if u_score > 0.25 else "Low", "score": round(u_score, 3),
                   "disagreement": round(spread, 3), "missing_or_degraded_data": round(min(1.0, dq), 3),
                   "calibration_band": round(ci_w, 3),
                   "drivers": [d for d, on in (("agents disagree", spread > 0.3), ("missing or degraded agents", dq > 0.1),
                                               ("wide historical calibration band", ci_w > 0.15)) if on]}
    if decisive:
        d0 = decisive[0]
        explanation += (" " if explanation else "") + (
            f"Counterfactual: without {NICE[d0['without']]} the composite would be {d0['composite']:+.2f} "
            f"(a {d0['zone']} verdict), so that agent is decisive.")

    # 4/5. action + gates -------------------------------------------------
    notes = []
    gate = risk.data.get("suitability") if risk.usable else None
    conf_ok = conf >= pp["conf_min"]
    action = "HOLD" if held else "WATCH"
    size_mult = 1.0 if grounded else 0.5
    if composite >= pp["buy_thr"] and conf_ok:
        base = "BUY" if (composite >= pp["strong_buy"] and conf >= pp["conf_min"] + 0.10 and grounded) else "ACCUMULATE"
        if gate is None:
            notes.append("Price/volatility data unavailable, so no trade can be sized: watch only")
        elif gate == "breach":
            notes.append("Signal is positive but fails your risk limits: " + "; ".join(risk.data["reasons"]))
            if held and risk.data["weight_pct"] > user["max_position_pct"]:
                notes.append("Consider trimming back to your position limit")
        else:
            action = base
            if gate == "stretch":
                size_mult *= 0.6
                notes.append("Close to one of your limits: position size reduced")
    elif composite >= pp["buy_thr"] and not conf_ok:
        notes.append(f"Composite {composite:+.2f} clears your buy threshold ({pp['buy_thr']:.2f}) but confidence "
                     f"{conf:.0%} is below your minimum {pp['conf_min']:.0%}: wait for confirmation")
    elif composite <= pp["sell_thr"] and conf >= pp["conf_min"] - 0.10:
        if held:
            action = "SELL" if composite <= pp["strong_sell"] else "TRIM"
        else:
            action = "AVOID"
    if action in ("BUY", "ACCUMULATE") and beh.usable and beh.data.get("fomo_risk"):
        size_mult *= beh.data["size_multiplier"]
        if action == "BUY":
            action = "ACCUMULATE"
        notes.append("Behavioural guard: " + beh.data["nudges"][0] + " (entry split into 3 tranches)")
    if action in ("TRIM", "SELL") and beh.usable and beh.data.get("panic_risk") and composite > pp["strong_sell"]:
        action = "HOLD"
        notes.append("Behavioural guard: " + beh.data["nudges"][0] + "; signals are not bearish enough to justify exiting")
    if conflicts and action in ("BUY", "ACCUMULATE") and user["risk_tolerance"] <= 4:
        action = "WATCH" if not held else "HOLD"
        notes.append("Conservative profile does not act while agents disagree")

    # 6. sizing -----------------------------------------------------------
    sizing = None
    cash, total = snapshot["cash"], snapshot["total"]
    if action in ("BUY", "ACCUMULATE") and gate in ("fits", "stretch") and price > 0:
        room = max(0.0, min(risk.data["pos_room_pct"], risk.data["sector_room_pct"]))
        intensity = float(np.clip(composite / 0.6, 0.3, 1.0))
        pct = room * intensity * conf * (1.0 if action == "BUY" else 0.6) * size_mult
        amount = min(pct / 100 * total, cash * 0.9)
        qty = int(amount // price)
        if qty >= 1:
            sizing = {"side": "BUY", "qty": qty, "amount": round(qty * price, 2), "target_pct": round(qty * price / total * 100, 2),
                      "tranches": 3 if (beh.usable and beh.data.get("fomo_risk")) else 1}
        else:
            notes.append("Suggested size rounds to zero shares (limited cash/headroom)")
    elif action in ("TRIM", "SELL") and held:
        frac = 1.0 if action == "SELL" else 0.4
        qty = max(1, int(pos["qty"] * frac))
        sizing = {"side": "SELL", "qty": qty, "amount": round(qty * price, 2), "target_pct": round(-qty * price / total * 100, 2),
                  "tranches": 1}
    elif held and risk.usable and risk.data["weight_pct"] > user["max_position_pct"] and action == "HOLD":
        excess = (risk.data["weight_pct"] - user["max_position_pct"]) / 100 * total
        q = int(excess // price)
        if q >= 1:
            notes.append(f"Rebalance idea: sell ~{q} shares (₹{q * price:,.0f}) to return to your position limit")

    # reasoning chain -----------------------------------------------------
    chain = []
    px_ref = next((c.source_id for o in outs.values() for c in o.citations if c.source_id.startswith("PX:")), None)
    chain.append({"step": "Ingest", "text": f"{symbol} last close ₹{price:,.2f}. Data-quality gate: "
                  + ("; ".join(dict.fromkeys(warnings)) if warnings else "all feeds healthy"), "refs": [px_ref] if px_ref else []})
    for a in SIGNALS:
        o = sig[a]
        if o.usable:
            chain.append({"step": NICE[a] + " agent", "text": f"{o.summary} (score {o.score:+.2f}, confidence {o.confidence:.0%})"
                          + (" [degraded]" if o.status == "degraded" else ""), "refs": _top_refs(o)})
        else:
            chain.append({"step": NICE[a] + " agent", "text": f"{o.status.upper()}: {o.summary}", "refs": []})
    wtxt = ", ".join(f"{NICE[a]} {W[a]:.0%}" for a in SIGNALS)
    chain.append({"step": "Weighting", "text": f"{pp['label']} profile (risk tolerance {pp['tolerance']}/10) weights: {wtxt}. "
                  f"Confidence-weighted composite = {composite:+.2f} (range {lo:+.2f} to {hi:+.2f})", "refs": []})
    chain.append({"step": "Agreement check", "text": explanation or
                  f"Agents broadly agree (spread {spread:.2f}); agreement factor {agreement:.2f}", "refs": []})
    chain.append({"step": "Confidence", "text": f"{conf:.0%} = weighted agent confidence {base_conf:.0%} x agreement"
                  + (" x 0.8 conflict penalty" if conflicts else "") + ("" if grounded else " (capped: no filing grounding)")
                  + (f". Calibrated probability the direction is right over 30 days: {pred['p']:.0%} (90% band {pred['lo']:.0%}-{pred['hi']:.0%}, "
                     f"fitted on {pred['n']} back-test signals). Uncertainty: {uncertainty['level']}"
                     + (f" ({', '.join(uncertainty['drivers'])})" if uncertainty['drivers'] else "") if pred else ""), "refs": []})
    chain.append({"step": "Risk gate", "text": risk.summary, "refs": _top_refs(risk, 2)})
    chain.append({"step": "Behaviour check", "text": beh.summary, "refs": _top_refs(beh, 3)})
    dec = f"{action}" + (f": {sizing['side']} {sizing['qty']} shares (~₹{sizing['amount']:,.0f})" if sizing else "")
    chain.append({"step": "Decision", "text": dec + (". " + " ".join(notes) if notes else ""), "refs": []})

    cits: dict[str, dict] = {}
    for o in outs.values():
        for c in o.citations:
            cits.setdefault(c.source_id, c.__dict__.copy())

    headline = {"BUY": "Build a position", "ACCUMULATE": "Add gradually", "HOLD": "Hold; no change", "WATCH": "Watch; do not buy yet",
                "TRIM": "Reduce exposure", "SELL": "Exit position", "AVOID": "Avoid"}[action]
    return {"symbol": symbol, "name": st.name, "sector": st.sector, "price": price, "user_id": user["id"],
            "profile_label": pp["label"], "action": action, "headline": headline, "composite": round(composite, 3),
            "confidence": round(conf, 3), "interval": [round(lo, 3), round(hi, 3)], "calibration": cal, "calibrated": pred, "uncertainty": uncertainty,
            "counterfactuals": cfs,
            "held": held, "position": pos, "sizing": sizing, "notes": notes, "grounded": grounded,
            "conflicts": conflicts, "disagreement": explanation, "contributions": contribs,
            "agents": {a: o.to_dict() for a, o in outs.items()}, "reasoning_chain": chain, "citations": cits,
            "warnings": list(dict.fromkeys(warnings)), "thresholds": {k: round(v, 3) for k, v in pp.items() if isinstance(v, float)}}

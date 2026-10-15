"""The six specialised agents. Each takes an `AgentContext` and returns an `AgentOutput`.

Signal agents (market-wide, user-independent):  momentum, volume, sentiment, filings (RAG)
Constraint agents (user-specific):               risk (portfolio fit), behavior (trade-history nudges)
"""
from __future__ import annotations

import datetime as dt
import time
from dataclasses import dataclass, field

import numpy as np

from . import config
from .contracts import AgentOutput, Citation, Evidence, failed_output
from .lexicon import sentences, tone
from .market import FeedUnavailable, get_stock, MarketData, MarketFeed, validate_prices
from .rag import RagIndex
from .runcache import RunCache
from . import signals as sg


@dataclass
class AgentContext:
    symbol: str
    feed: MarketFeed
    rag: RagIndex
    data: MarketData
    degr: frozenset = frozenset()
    user: dict | None = None
    snapshot: dict | None = None
    behavior: dict | None = None
    past_decisions: list = field(default_factory=list)
    cache: RunCache | None = None

    def cached(self, key, fn):
        return self.cache.get(key, fn) if self.cache is not None else fn()


def _px_citation(symbol: str, res, last_close: float) -> Citation:
    d = res.as_of.isoformat()
    return Citation(f"PX:{symbol}:{d}", "market_data", f"{symbol} price & volume series ({res.source})", d,
                    f"Last close ₹{last_close:,.2f} as of {d}", url=getattr(res, "url", ""))


def _load_prices(ctx: AgentContext):
    """Fetch + data-quality gate. Returns (series, feed_result, warnings, confidence multiplier)."""
    def fetch():
        res = ctx.feed.prices(ctx.symbol, ctx.degr)
        clean, warns, mult = validate_prices(res.data)
        warns = list(res.warnings) + warns
        if res.stale_days:
            mult *= 0.6
        return clean, res, warns, mult

    clean, res, warns, mult = ctx.cached(("px", ctx.symbol, tuple(sorted(ctx.degr))), fetch)
    return clean, res, list(warns), mult


class MomentumAgent:
    name, kind = "momentum", "signal"

    def run(self, ctx: AgentContext) -> AgentOutput:
        try:
            ps, res, warns, mult = _load_prices(ctx)
        except FeedUnavailable as e:
            return failed_output(self.name, ctx.symbol, self.kind, "failed", f"Price feed unavailable ({e}); no momentum view.")
        sig = sg.momentum_signal(ps.close)
        cit = _px_citation(ctx.symbol, res, float(ps.close[-1]))
        ev = [Evidence(t, m, v, cit.source_id, s) for t, m, v, s in sig.items]
        return AgentOutput(self.name, ctx.symbol, self.kind, "degraded" if warns else "ok", sig.label, sig.score,
                           sig.confidence * mult,
                           f"{sig.label}: 20d {sig.flags['r20']:+.1%}, 60d {sig.flags['r60']:+.1%}, RSI {sig.flags['rsi']:.0f}.",
                           ev, [cit], sig.flags, warns)


class VolumeAgent:
    name, kind = "volume", "signal"

    def run(self, ctx: AgentContext) -> AgentOutput:
        try:
            ps, res, warns, mult = _load_prices(ctx)
        except FeedUnavailable as e:
            return failed_output(self.name, ctx.symbol, self.kind, "failed", f"Price feed unavailable ({e}); no volume view.")
        sig = sg.volume_signal(ps.close, ps.volume)
        cit = _px_citation(ctx.symbol, res, float(ps.close[-1]))
        ev = [Evidence(t, m, v, cit.source_id, s) for t, m, v, s in sig.items]
        return AgentOutput(self.name, ctx.symbol, self.kind, "degraded" if warns else "ok", sig.label, sig.score,
                           sig.confidence * mult,
                           f"{sig.label}: volume {sig.flags['ratio']:.1f}x normal (z {sig.flags['vz']:+.1f}).",
                           ev, [cit], sig.flags, warns)


class SentimentAgent:
    name, kind = "sentiment", "signal"

    def run(self, ctx: AgentContext) -> AgentOutput:
        if "slow_agent" in ctx.degr:
            time.sleep(config.AGENT_TIMEOUT_S * 2.5)          # simulated hang; orchestrator times it out
        warns, mult, status = [], 1.0, "ok"
        try:
            res = ctx.cached(("sent", ctx.symbol, tuple(sorted(ctx.degr))), lambda: ctx.feed.sentiment(ctx.symbol, ctx.degr))
        except FeedUnavailable:
            res = ctx.feed.cached_sentiment(ctx.symbol)       # graceful fallback to last-good snapshot
            warns, mult, status = list(res.warnings), 0.5, "degraded"
        b = res.data
        sig = sg.sentiment_signal(b["headlines"], b["as_of"], b["fii_5d"], b["pcr"])
        d = b["as_of"].isoformat()
        cits, ev = [], []
        by_id = {h.id: h for h in b["headlines"]}
        for text, metric, value, stance in sig.items:
            if metric.startswith("headline:"):
                h = by_id[metric.split(":", 1)[1]]
                cid = h.id
                cits.append(Citation(h.id, "news", h.text, h.date.isoformat(), f"{h.source} (reliability {h.reliability:.1f})",
                                     url=getattr(h, "url", "")))
            elif metric == "fii_5d":
                cid = f"FII:{d}"
                cits.append(Citation(cid, "flow", "FII/DII net flow disclosure (simulated)", d, text))
            elif metric == "pcr":
                cid = f"OPT:{ctx.symbol}:{d}"
                cits.append(Citation(cid, "flow", f"{ctx.symbol} options chain put-call ratio (simulated)", d, text))
            else:
                cid = f"NEWS:{ctx.symbol}:{d}"
                cits.append(Citation(cid, "news", "News feed (no headlines in window)", d, text))
            ev.append(Evidence(text, metric.split(":")[0], value, cid, stance))
        seen, uniq = set(), []
        for c in cits:
            if c.source_id not in seen:
                seen.add(c.source_id)
                uniq.append(c)
        return AgentOutput(self.name, ctx.symbol, self.kind, status, sig.label, sig.score, sig.confidence * mult,
                           f"{sig.label}: {sig.flags['n_headlines']} headline(s), news tone {sig.flags['news']:+.2f}.",
                           ev, uniq, {**sig.flags, "from_cache": res.from_cache}, warns)


class FilingsAgent:
    """RAG agent: every claim is a verbatim quote from a retrieved filing passage."""
    name, kind = "filings", "signal"
    RECENCY_TAU = 120.0      # retrieval prefers recent passages (days)
    AGE_TAU = 90.0           # evidence weight decays with filing age (days)
    QUERIES = [
        "earnings guidance revenue growth margin outlook order book",
        "risk litigation regulatory probe penalty promoter pledge debt downgrade",
        "management commentary demand capex deleveraging dividend",
    ]

    def run(self, ctx: AgentContext) -> AgentOutput:
        sym = ctx.symbol
        if "filings_unavailable" in ctx.degr or f"filings_missing:{sym}" in ctx.degr:
            return failed_output(self.name, sym, self.kind, "abstained",
                                 "No filings could be retrieved for this company; the system makes no document-grounded claim.",
                                 ["Filing index unavailable - recommendation will not be grounded in disclosures"])
        sector = get_stock(sym).sector

        def retrieve():
            ctx.feed._io("filings")
            pool: dict[str, object] = {}
            for q in self.QUERIES:
                for h in ctx.rag.search(q, symbol=sym, sector=sector, k=3, recency=self.RECENCY_TAU):
                    if h.chunk.chunk_id not in pool or h.score > pool[h.chunk.chunk_id].score:
                        pool[h.chunk.chunk_id] = h
            return pool

        pool = ctx.cached(("filings", sym), retrieve)
        hits = sorted(pool.values(), key=lambda h: -h.score)[:6]
        if not hits:
            return failed_output(self.name, sym, self.kind, "abstained", "No relevant filing passages found.",
                                 ["No documents matched this company"])
        asof = getattr(ctx.feed, "as_of", None) or ctx.data.dates[-1]
        rows, ev, cits = [], [], []

        def quote_of(c):
            sents = sentences(c.text)
            best = max(sents, key=lambda s: abs(tone(s)[0]) * 10 + len(tone(s)[1]) + len(tone(s)[2]))
            return best if tone(best)[1] or tone(best)[2] else sents[0]

        for h in hits:
            c = h.chunk
            t, pos, neg = tone(c.text)
            age = (asof - dt.date.fromisoformat(c.date)).days
            w = max(h.score, 0.05) * float(np.exp(-age / self.AGE_TAU))
            rows.append((t, w, age))
            quote = quote_of(c)
            verified = ctx.rag.verify_quote(c.doc_id, quote)
            stance = sg.stance(tone(quote)[0], 0.15)
            ev.append(Evidence(f"{c.doc_type} - \"{quote}\"", "tone", round(t, 2), c.chunk_id, stance))
            cits.append(Citation(c.chunk_id, "filing", f"{c.title} - {c.heading}", c.date, quote, round(h.score, 3),
                                 c.url or f"/api/docs/{c.doc_id}", verified))
        tones = np.array([r[0] for r in rows])
        ws = np.array([r[1] for r in rows])
        score = float(np.clip(1.2 * (tones * ws).sum() / ws.sum(), -1, 1))
        consistency = 1 - min(1.0, float(np.sqrt((ws * (tones - (tones * ws).sum() / ws.sum()) ** 2).sum() / ws.sum())))
        newest = min(r[2] for r in rows)
        conf = 0.2 + 0.06 * len(rows) + 0.20 * consistency + 0.12 * abs(score)
        if newest > 90:
            conf *= 0.8
        trend = ctx.cached(("trend", sym), lambda: self._trend(ctx, sym))
        if trend:                                  # time structure from the multi-quarter corpus
            have = {c.source_id for c in cits}
            for c in (trend["first_chunk"], trend["last_chunk"]):
                if c.chunk_id not in have:
                    q = quote_of(c)
                    cits.append(Citation(c.chunk_id, "filing", f"{c.title} - {c.heading}", c.date, q, None,
                                         c.url or f"/api/docs/{c.doc_id}", ctx.rag.verify_quote(c.doc_id, q)))
                    have.add(c.chunk_id)
            ev.append(Evidence(f"Disclosure tone trend: {trend['first_period']} {trend['first']:+.2f} -> {trend['last_period']} "
                               f"{trend['last']:+.2f} ({trend['direction']})", "tone_trend", round(trend["delta"], 2),
                               trend["last_chunk"].chunk_id, "bullish" if trend["delta"] > 0.15 else "bearish" if trend["delta"] < -0.15 else "neutral"))
        label = ("Strongly Supportive Disclosures" if score >= 0.5 else "Supportive Disclosures" if score >= 0.15 else
                 "Mixed Disclosures" if score > -0.15 else "Adverse Disclosures" if score > -0.5 else
                 "Strongly Adverse Disclosures")
        n_neg = int((tones < -0.15).sum())
        n_pos = int((tones > 0.15).sum())
        tr = f" Trend: {trend['direction']}." if trend else ""
        return AgentOutput(self.name, sym, self.kind, "ok", label, score, float(np.clip(conf, 0.1, 0.85)),
                           f"{label}: {len(rows)} passages retrieved ({n_pos} supportive, {n_neg} adverse); "
                           f"newest filing {newest} days old.{tr}", ev, cits,
                           {"n_passages": len(rows), "newest_age_days": newest, "n_adverse": n_neg, "n_supportive": n_pos,
                            "verified": all(c.verified for c in cits),
                            "trend": {k: v for k, v in (trend or {}).items() if not k.endswith("_chunk")}}, [])

    @staticmethod
    def _trend(ctx: AgentContext, sym: str):
        """Average tone of earnings/disclosure passages per quarter -> improving or deteriorating?"""
        by: dict[str, list] = {}
        for c in ctx.rag.chunks:
            if c.symbol == sym and c.doc_type in ("Earnings Transcript", "SEBI Filing") and c.period:
                by.setdefault((c.period, c.date), []).append(c)
        if len(by) < 6:
            return None
        periods = {}
        for (per, date), cs in by.items():
            periods.setdefault(per, {"date": date, "chunks": []})["chunks"].extend(cs)
        order = sorted(periods.items(), key=lambda kv: kv[1]["date"])
        means = [(per, float(np.mean([tone(c.text)[0] for c in v["chunks"]])), v["chunks"][0]) for per, v in order]
        first_per, first, first_c = means[0]
        last_per, last, last_c = means[-1]
        base = float(np.mean([m[1] for m in means[:4]]))      # older year as the reference level
        delta = last - base
        return {"first_period": first_per, "first": first, "last_period": last_per, "last": last, "baseline": base,
                "delta": delta, "direction": "improving" if delta > 0.15 else "deteriorating" if delta < -0.15 else "stable",
                "first_chunk": first_c, "last_chunk": last_c}


class RiskAgent:
    """User-specific suitability gate: volatility, position headroom, sector headroom."""
    name, kind = "risk", "constraint"

    def run(self, ctx: AgentContext) -> AgentOutput:
        u, snap = ctx.user, ctx.snapshot
        try:
            ps, res, warns, mult = _load_prices(ctx)
        except FeedUnavailable as e:
            return failed_output(self.name, ctx.symbol, self.kind, "failed",
                                 f"Cannot assess volatility: price feed unavailable ({e}); position sizing disabled.")
        ctx.feed._io("portfolio")
        st = get_stock(ctx.symbol)
        vol = sg.annual_vol(ps.close)
        pos = next((p for p in snap["positions"] if p["symbol"] == ctx.symbol), None)
        w = pos["weight"] * 100 if pos else 0.0
        sec_w = snap["sector_weights"].get(st.sector, 0.0) * 100
        pos_room, sec_room = u["max_position_pct"] - w, u["max_sector_pct"] - sec_w
        reasons = []
        if vol > u["max_volatility"]:
            reasons.append(f"annualised volatility {vol:.0%} exceeds your {u['max_volatility']:.0%} limit")
        if pos_room < 1.0:
            reasons.append(f"position already {w:.1f}% vs {u['max_position_pct']:.0f}% limit")
        if sec_room < 1.0:
            reasons.append(f"{st.sector} exposure {sec_w:.1f}% vs {u['max_sector_pct']:.0f}% limit")
        if reasons:
            suit, score = "breach", -0.7
        elif vol > 0.8 * u["max_volatility"] or pos_room < 2.5 or sec_room < 2.5:
            suit, score = "stretch", 0.0
            reasons.append("close to one of your limits")
        else:
            suit, score = "fits", 0.6
        pcit = Citation(f"PF:{u['id']}:{res.as_of.isoformat()}", "portfolio", f"{u['name']}'s portfolio & risk limits",
                        res.as_of.isoformat(), f"total ₹{snap['total']:,.0f}, cash {snap['cash_pct']:.0%}")
        xcit = _px_citation(ctx.symbol, res, float(ps.close[-1]))
        ev = [Evidence(f"{ctx.symbol} 60-day annualised volatility {vol:.0%} vs your limit {u['max_volatility']:.0%}",
                       "volatility", round(vol, 3), xcit.source_id, "bearish" if vol > u["max_volatility"] else "neutral"),
              Evidence(f"Current weight {w:.1f}% (limit {u['max_position_pct']:.0f}%); {st.sector} sector {sec_w:.1f}% "
                       f"(limit {u['max_sector_pct']:.0f}%)", "weights", round(w, 2), pcit.source_id, "neutral")]
        label = {"fits": "Fits Profile", "stretch": "Stretch", "breach": "Breaches Limits"}[suit]
        return AgentOutput(self.name, ctx.symbol, self.kind, "degraded" if warns else "ok", label, score,
                           0.9 * mult, f"{label} for {u['name']}" + (": " + "; ".join(reasons) if reasons else "."),
                           ev, [pcit, xcit], {"suitability": suit, "vol": vol, "weight_pct": w, "pos_room_pct": pos_room,
                                              "sector_room_pct": sec_room, "reasons": reasons, "sector": st.sector}, warns)


class BehaviorAgent:
    """Reads the user's trade log and past decisions and flags behavioural traps for *this* stock."""
    name, kind = "behavior", "constraint"

    def run(self, ctx: AgentContext) -> AgentOutput:
        u, b = ctx.user, ctx.behavior
        warns = []
        r20 = dd = None
        try:
            ps, res, warns, _ = _load_prices(ctx)
            c = ps.close
            r20, dd = float(c[-1] / c[-21] - 1), float(c[-1] / c[-21:].max() - 1)
        except FeedUnavailable:
            warns.append("Price feed unavailable: stock-specific behavioural triggers cannot be evaluated")
        held = any(p["symbol"] == ctx.symbol for p in ctx.snapshot["positions"])
        fomo = r20 is not None and r20 >= 0.10 and (b["archetype"] == "FOMO-prone" or b["fomo_buy_rate"] >= 0.3)
        panic = held and dd is not None and dd <= -0.06 and (b["archetype"] == "Panic-prone" or b["panic_sell_rate"] >= 0.3)
        nudges, mult = [], 1.0
        if fomo:
            nudges.append(f"{ctx.symbol} is up {r20:+.0%} in 20 days; {b['fomo_buy_rate']:.0%} of your past buys came after similar run-ups - stage entries")
            mult = 0.5
        if panic:
            nudges.append(f"{ctx.symbol} is {dd:.0%} below its 20-day high; you sold {b['regretted_panic_sells']} stock(s) in similar dips that later recovered")
        decs = [d for d in ctx.past_decisions if d["symbol"] == ctx.symbol][:3]
        hid = f"HIST:{u['id']}"
        cits = [Citation(hid, "history", f"{u['name']}'s trade log ({b['n_trades']} trades) and behavioural analysis",
                         "", f"archetype {b['archetype']}; FOMO-buy rate {b['fomo_buy_rate']:.0%}; panic-sell rate {b['panic_sell_rate']:.0%}")]
        ev = [Evidence(f"Archetype from trade log: {b['archetype']} (FOMO-buy {b['fomo_buy_rate']:.0%}, panic-sell {b['panic_sell_rate']:.0%}, "
                       f"{b['trades_per_year']:.0f} trades/yr)", "archetype", b["archetype"], hid,
                       "bearish" if b["archetype"] != "Disciplined" else "neutral")]
        for e in b["evidence"][:3]:
            cits.append(Citation(e["id"], "history", e["text"], e["date"], e["text"]))
            ev.append(Evidence(e["text"], "past_trade", e["symbol"], e["id"], "neutral"))
        for d in decs:
            cid = f"DEC-{d['id']}"
            cits.append(Citation(cid, "history", f"Earlier {d['action']} recommendation: you chose '{d['choice']}'", "",
                                 d.get("note", "")))
            ev.append(Evidence(f"Previously you {d['choice']} a {d['action']} recommendation on {ctx.symbol}", "decision", d["choice"], cid, "neutral"))
        flagged = fomo or panic
        label = "Behavioural Risk" if flagged else "No Behavioural Flag"
        return AgentOutput(self.name, ctx.symbol, self.kind, "degraded" if warns else "ok", label,
                           -0.5 if flagged else 0.4, 0.75 if r20 is not None else 0.4,
                           ("; ".join(nudges) if nudges else f"No behavioural trap triggered for {ctx.symbol}."),
                           ev, cits, {"fomo_risk": bool(fomo), "panic_risk": bool(panic), "size_multiplier": mult,
                                      "nudges": nudges, "r20": r20, "drawdown": dd, "archetype": b["archetype"]}, warns)


SIGNAL_AGENTS = {"momentum": MomentumAgent(), "volume": VolumeAgent(), "sentiment": SentimentAgent(),
                 "filings": FilingsAgent()}
USER_AGENTS = {"risk": RiskAgent(), "behavior": BehaviorAgent()}
ALL_AGENTS = {**SIGNAL_AGENTS, **USER_AGENTS}

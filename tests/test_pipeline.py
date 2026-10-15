import numpy as np
import pytest

from app import signals as sg
from app.agents import SIGNAL_AGENTS, AgentContext
from app.contracts import AgentOutput, Evidence
from app.market import MarketData, SYMBOLS

ACTIONS = {"BUY", "ACCUMULATE", "HOLD", "WATCH", "TRIM", "SELL", "AVOID"}


# ---------------------------------------------------------------- contract
def test_contract_rejects_uncited_evidence():
    o = AgentOutput("x", "S", "signal", "ok", "Bullish", 0.5, 0.5, "s", [Evidence("claim", source_id="")], [])
    assert any("uncited" in p for p in o.validate())


def test_contract_rejects_out_of_range():
    o = AgentOutput("x", "S", "signal", "failed", "No signal", 3.0, 0.5, "s")
    assert o.validate()


def test_data_is_deterministic():
    a, b = MarketData(), MarketData()
    assert np.array_equal(a.close["BHARATOIL"], b.close["BHARATOIL"])
    assert [h.id for h in a.headlines["NAVBANK"]] == [h.id for h in b.headlines["NAVBANK"]]


# ---------------------------------------------------------------- agents
def test_every_signal_agent_meets_contract_for_every_stock(engine):
    for s in SYMBOLS:
        ctx = AgentContext(s, engine.feed, engine.rag, engine.data)
        for name, ag in SIGNAL_AGENTS.items():
            o = ag.run(ctx)
            assert not o.validate(), (s, name, o.validate())
            assert o.usable and o.evidence


def test_scenarios_produce_designed_conflicts(engine):
    r = engine.analyze("meera", persist=False)
    by = {x["symbol"]: x for x in r["recommendations"]}
    assert by["VIKRAMINFRA"]["conflicts"], "price-vs-filings conflict expected"
    assert by["INFOSPHERE"]["conflicts"], "momentum-vs-filings conflict expected"
    assert not by["BHARATOIL"]["conflicts"]


def test_signals_have_no_lookahead(engine):
    d = engine.data
    t = 300
    base = sg.momentum_signal(d.close["TECHNOVA"][: t + 1]).score
    fut = d.close["TECHNOVA"].copy()
    fut[t + 1:] *= 3.0                                   # wreck the future
    assert sg.momentum_signal(fut[: t + 1]).score == base


# ---------------------------------------------------------------- RAG
def test_rag_retrieves_relevant_and_filters_by_company(engine):
    hits = engine.rag.search("promoter pledge show-cause notice penalty", symbol="VIKRAMINFRA", k=3)
    assert hits and all(h.chunk.symbol == "VIKRAMINFRA" for h in hits)
    text = hits[0].chunk.text.lower()
    assert "pledge" in text or "show-cause" in text


def test_rag_attribution_verification(engine):
    d = engine.rag.doc("GATIMOTORS-LODR-R30")
    assert engine.rag.verify_quote(d.doc_id, d.sections[0][1][:40])
    assert not engine.rag.verify_quote(d.doc_id, "Gati Motors announced a record profit")


def test_filings_agent_quotes_are_verified(engine):
    r = engine.analyze("asha", persist=False)
    for rec in r["recommendations"]:
        fil = [c for c in rec["citations"].values() if c["kind"] == "filing"]
        assert fil and all(c["verified"] for c in fil)


# ---------------------------------------------------------------- personalisation
def test_same_inputs_different_outputs(engine):
    out = engine.compare("VIKRAMINFRA")
    acts = {x["user"]["id"]: x["recommendation"]["action"] for x in out["results"]}
    assert len(set(acts.values())) >= 2, acts
    comps = {round(x["recommendation"]["composite"], 3) for x in out["results"]}
    assert len(comps) == 3                              # profile weights differ -> composites differ
    assert set(out["market_inputs"]) == {"momentum", "volume", "sentiment", "filings"}


def test_profile_edit_changes_recommendation(engine):
    store = engine.store
    original = store.get_user("meera")
    try:
        store.update_user("meera", {"risk_tolerance": 1})
        low = engine.compare("BHARATOIL")
        store.update_user("meera", {"risk_tolerance": 10})
        high = engine.compare("BHARATOIL")
        m_low = [x for x in low["results"] if x["user"]["id"] == "meera"][0]["recommendation"]
        m_high = [x for x in high["results"] if x["user"]["id"] == "meera"][0]["recommendation"]
        assert m_low["thresholds"]["buy_thr"] > m_high["thresholds"]["buy_thr"]
        assert m_low["contributions"][3]["weight"] > m_high["contributions"][3]["weight"]   # filings weight
    finally:
        store.update_user("meera", {"risk_tolerance": original["risk_tolerance"]})


def test_behavioural_archetypes_derived_from_trades(engine):
    arch = {u: engine.analyze(u, persist=False)["behavior"]["archetype"] for u in ("asha", "rohan", "meera")}
    assert arch == {"asha": "Panic-prone", "rohan": "FOMO-prone", "meera": "Disciplined"}


def test_fomo_guard_reduces_aggression(engine):
    r = engine.analyze("rohan", persist=False)
    for rec in r["recommendations"]:
        if rec["agents"]["behavior"]["data"]["fomo_risk"]:
            assert rec["action"] != "BUY"


def test_position_limits_are_respected(engine):
    for uid in ("asha", "rohan", "meera"):
        r = engine.analyze(uid, persist=False)
        u = r["user"]
        for rec in r["recommendations"]:
            if rec["sizing"] and rec["sizing"]["side"] == "BUY":
                cur = rec["position"]["weight"] * 100 if rec["position"] else 0
                assert cur + rec["sizing"]["target_pct"] <= u["max_position_pct"] + 0.5, (uid, rec["symbol"])


# ---------------------------------------------------------------- degraded data
@pytest.mark.parametrize("degr", [["price_feed_stale"], ["price_feed_down"], ["price_corrupt"], ["sentiment_feed_down"],
                                  ["filings_unavailable"], ["slow_agent"],
                                  ["price_feed_down", "sentiment_feed_down", "filings_unavailable"]])
def test_pipeline_survives_degradation_and_stays_cited(engine, degr):
    r = engine.analyze("rohan", degr, persist=False)
    assert len(r["recommendations"]) == 12
    for rec in r["recommendations"]:
        assert rec["action"] in ACTIONS
        assert rec["citations"], f"{rec['symbol']} has no citations under {degr}"
        assert 0 <= rec["confidence"] <= 0.95
    assert r["metrics"]["citation_coverage"] == 1.0


def test_sentiment_outage_falls_back_to_cache(engine):
    r = engine.analyze("asha", ["sentiment_feed_down"], persist=False)
    o = r["recommendations"][0]["agents"]["sentiment"]
    assert o["status"] == "degraded" and o["data"]["from_cache"]


def test_missing_filings_abstain_and_cap_confidence(engine):
    r = engine.analyze("meera", ["filings_unavailable"], persist=False)
    for rec in r["recommendations"]:
        assert rec["agents"]["filings"]["status"] == "abstained"
        assert not rec["grounded"] and rec["confidence"] <= 0.60
        assert rec["action"] != "BUY"


def test_per_stock_missing_filing(engine):
    r = engine.analyze("meera", ["filings_missing:NAVBANK"], persist=False)
    by = {x["symbol"]: x for x in r["recommendations"]}
    assert by["NAVBANK"]["agents"]["filings"]["status"] == "abstained"
    assert by["BHARATOIL"]["agents"]["filings"]["status"] == "ok"


def test_price_outage_disables_sizing(engine):
    r = engine.analyze("meera", ["price_feed_down"], persist=False)
    assert all(rec["sizing"] is None or rec["sizing"]["side"] == "SELL" for rec in r["recommendations"])
    assert all(rec["agents"]["momentum"]["status"] == "failed" for rec in r["recommendations"])


def test_slow_agent_times_out_without_blocking(engine):
    r = engine.analyze("asha", ["slow_agent"], persist=False)
    assert all(rec["agents"]["sentiment"]["status"] == "timeout" for rec in r["recommendations"])
    assert r["metrics"]["e2e_ms"] < 3000


def test_corrupt_prices_are_cleaned(engine):
    r = engine.analyze("asha", ["price_corrupt"], persist=False)
    mom = r["recommendations"][0]["agents"]["momentum"]
    assert mom["status"] == "degraded" and mom["warnings"]


def test_crashing_agent_is_isolated(engine, monkeypatch):
    from app.agents import MomentumAgent

    def boom(self, ctx):
        raise RuntimeError("bug")

    monkeypatch.setattr(MomentumAgent, "run", boom)
    r = engine.analyze("asha", persist=False)
    assert len(r["recommendations"]) == 12
    assert r["recommendations"][0]["agents"]["momentum"]["status"] == "failed"


# ---------------------------------------------------------------- evaluation / persistence
def test_backtest_is_sane(engine):
    bt = engine.backtest
    for k in ("momentum", "volume", "sentiment", "ensemble"):
        assert bt[k]["n_signals"] > 100
        assert 0 <= bt[k]["hit_rate"] <= 1
    assert bt["_meta"]["horizon_days"] == 30


def test_session_metrics_have_at_least_three_measures(engine):
    m = engine.analyze("asha", persist=True)["metrics"]
    for key in ("e2e_ms", "parallel_speedup", "concentration_score_before", "signal_accuracy_30d", "citation_coverage"):
        assert m[key] is not None
    assert m["parallel_speedup"] >= 1


def test_sessions_and_decisions_persist(engine):
    r = engine.analyze("meera", persist=True)
    rec = r["recommendations"][0]
    engine.store.save_decision("meera", rec["rec_id"], rec["symbol"], rec["action"], "accepted")
    assert engine.store.sessions("meera")[0]["id"] == r["session_id"]
    assert engine.store.decisions("meera")[0]["choice"] == "accepted"
    # the behaviour agent sees the earlier decision on the next run
    r2 = engine.analyze("meera", persist=False)
    b = [x for x in r2["recommendations"] if x["symbol"] == rec["symbol"]][0]["agents"]["behavior"]
    assert any(e["metric"] == "decision" for e in b["evidence"])


def test_trade_updates_portfolio(engine):
    before = engine.store.get_user("meera")["cash"]
    engine.store.apply_trade("meera", "2026-10-15", "KISANFOODS", "BUY", 10, 468.0)
    assert engine.store.get_user("meera")["cash"] == pytest.approx(before - 4680)
    with pytest.raises(ValueError):
        engine.store.apply_trade("meera", "2026-10-15", "KISANFOODS", "SELL", 10**6, 468.0)


# ---------------------------------------------------------------- HTTP API
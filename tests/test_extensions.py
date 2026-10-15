"""Tests for the seven extensions: live sources, vector DB, portfolio risk & alerts, backtests,
calibration, persistent sessions, and parallel-execution optimisation."""
import datetime as dt
import json
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from app import strategy
from app.agents import AgentContext
from app.contracts import AgentOutput, Evidence, Citation
from app.engine import run_agents
from app.rag import RagIndex
from app.rag_eval import evaluate


# ====================================================================== Ext 7: parallel execution
def test_shared_cache_cuts_feed_calls(engine):
    engine.use_cache = False
    c0 = sum(engine.feed.calls.values())
    engine.analyze("asha", persist=False)
    no_cache = sum(engine.feed.calls.values()) - c0
    engine.use_cache = True
    c0 = sum(engine.feed.calls.values())
    r = engine.analyze("asha", persist=False)
    cached = sum(engine.feed.calls.values()) - c0
    assert cached < no_cache * 0.7
    assert r["metrics"]["cache"]["hits"] > 0 and r["metrics"]["feed_calls"]["prices"] == 12


def test_queue_time_does_not_trigger_timeouts():
    class Slow:
        name, kind = "slow", "signal"

        def run(self, ctx):
            time.sleep(0.05)
            return AgentOutput("slow", ctx.symbol, "signal", "ok", "Neutral", 0.0, 0.5, "ok",
                               [Evidence("e", source_id="X")], [Citation("X", "market_data", "x")])

    pool = ThreadPoolExecutor(max_workers=2)
    ctx = AgentContext("S", None, None, None)
    tasks = [((i, "slow"), Slow(), ctx) for i in range(12)]          # 12 x 50ms on 2 workers = 300ms of queueing
    res, wall = run_agents(tasks, pool=pool, timeout_s=0.15)
    assert wall > 250 and all(o.status == "ok" for o in res.values())


def test_hung_task_still_times_out():
    class Hang:
        name, kind = "hang", "signal"

        def run(self, ctx):
            time.sleep(1.0)

    pool = ThreadPoolExecutor(max_workers=2)
    res, wall = run_agents([((0, "h"), Hang(), AgentContext("S", None, None, None))], pool=pool, timeout_s=0.1)
    assert res[(0, "h")].status == "timeout" and wall < 600


def test_load_benchmark_shape(engine):
    rows = engine.load_benchmark(concurrency=(1, 3), per_level=3)
    assert [r["concurrent_users"] for r in rows] == [1, 3]
    assert all(r["p95_ms"] >= r["p50_ms"] > 0 and r["throughput_per_s"] > 0 for r in rows)
    pool = engine.pool_sweep(sizes=(2, 8))
    assert len(pool) == 2


# ====================================================================== Ext 5: calibration & uncertainty
def test_calibration_is_monotone_banded_and_out_of_sample(engine):
    cal = engine.calibrator
    ps = [cal.predict(c) for c in (0.2, 0.35, 0.5, 0.65, 0.8)]
    assert all(a["p"] <= b["p"] + 1e-9 for a, b in zip(ps, ps[1:]))
    assert all(p["lo"] <= p["p"] <= p["hi"] for p in ps)
    rep = cal.report
    assert rep["n_test"] > 100 and rep["ece_calibrated"] < rep["ece_raw"]
    assert rep["brier_calibrated"] <= rep["brier_raw"] + 0.005          # not worse than the raw number


def test_recommendation_carries_calibration_uncertainty_counterfactuals(engine):
    r = engine.analyze("asha", persist=False)
    v = {x["symbol"]: x for x in r["recommendations"]}["VIKRAMINFRA"]
    assert 0 < v["calibrated"]["p"] < 1 and v["uncertainty"]["level"] in ("Low", "Medium", "High")
    dec = [c for c in v["counterfactuals"] if c["decisive"]]
    assert dec and dec[0]["without"] == "filings"                      # removing the dissenting agent flips the verdict
    assert "Counterfactual" in v["disagreement"]


def test_uncertainty_rises_when_data_is_missing(engine):
    ok = engine.analyze("meera", persist=False)["recommendations"]
    bad = engine.analyze("meera", ["filings_unavailable", "sentiment_feed_down"], persist=False)["recommendations"]
    u_ok = np.mean([r["uncertainty"]["score"] for r in ok])
    u_bad = np.mean([r["uncertainty"]["score"] for r in bad])
    assert u_bad > u_ok


# ====================================================================== Ext 3: portfolio risk & alerts
def test_risk_contributions_sum_to_one_and_expose_hidden_concentration(engine):
    k = engine.analyze("rohan", persist=False)["portfolio"]["risk"]
    assert sum(c["risk_share"] for c in k["contributions"]) == pytest.approx(1.0, abs=1e-6)
    top = k["contributions"][0]
    assert top["symbol"] == "VIKRAMINFRA" and top["risk_share"] > top["weight"] * 1.4
    assert all(s["loss"] < 0 for s in k["stress"]) and k["es95_hist"] >= k["var95_hist"] > 0


def test_alert_rules_are_personal_and_alerts_track_new_and_resolved(engine):
    uid = "meera"
    first = engine.analyze(uid)
    second = engine.analyze(uid)
    assert not any(a["is_new"] for a in second["alerts"])
    n_before = len(second["alerts"])
    # silence move/volume alerts through the user's own rules -> they resolve; a new rule set is persisted
    engine.store.set_alert_rules(uid, {"daily_move_pct": 50, "volume_z": 99}, __import__("app.alerts", fromlist=["x"]).DEFAULT_RULES)
    third = engine.analyze(uid)
    assert len(third["alerts"]) < n_before and third["alerts_meta"]["resolved_ids"]
    assert third["alerts_meta"]["rules"]["daily_move_pct"] == 50
    assert all(a["why"] for a in third["alerts"])


def test_concentration_alert_depends_on_the_users_threshold(engine):
    from app.alerts import DEFAULT_RULES
    engine.store.set_alert_rules("rohan", {"max_concentration": 99}, DEFAULT_RULES)
    quiet = engine.analyze("rohan", persist=False)
    assert not any(a["kind"] == "concentration" for a in quiet["alerts"])
    engine.store.set_alert_rules("rohan", {"max_concentration": 10}, DEFAULT_RULES)
    loud = engine.analyze("rohan", persist=False)
    assert any(a["kind"] == "concentration" for a in loud["alerts"])
    engine.store.set_alert_rules("rohan", {"max_concentration": DEFAULT_RULES["max_concentration"]}, DEFAULT_RULES)


def test_watchlist_names_get_alerts(engine):
    r = engine.analyze("asha", persist=False)
    assert any(a["symbol"] in r["watchlist"] and "watchlist" in a["text"] for a in r["alerts"])


# ====================================================================== Ext 6: persistent sessions
def test_timeline_diffs_portfolio_and_recommendation_changes(engine):
    uid = "rohan"
    engine.analyze(uid)
    engine.store.apply_trade(uid, "2026-10-15", "KISANFOODS", "BUY", 20, float(engine.feed.last_close("KISANFOODS")), "test")
    a = engine.analyze(uid)
    d = engine.timeline(uid)
    last = d["entries"][0]
    assert last["session_id"] == a["session_id"]
    moves = {p["symbol"]: p["change"] for p in last["diff"]["positions"]}
    assert moves.get("KISANFOODS") == "opened" and last["diff"]["cash_delta"] < 0
    assert last["diff"]["behavior_changes"]                                  # new trade logged -> behavioural note
    engine.store.set_watchlist(uid, ["TECHNOVA"])
    assert d["decision_stats"]["total"] >= 0 and d["alert_rules"]


def test_decisions_link_to_the_session_that_made_the_advice(engine):
    uid = "meera"
    r = engine.analyze(uid)
    rec = r["recommendations"][0]
    engine.store.save_decision(uid, rec["rec_id"], rec["symbol"], rec["action"], "ignored")
    entry = [e for e in engine.timeline(uid)["entries"] if e["session_id"] == r["session_id"]][0]
    assert any(x["choice"] == "ignored" for x in entry["decisions"])
    assert engine.timeline(uid)["decision_stats"]["by_action"][rec["action"]]["ignored"] >= 1


# ====================================================================== Ext 4: strategy back-test
def test_strategy_backtest_personalised_and_causal(engine):
    st = engine.strategy()["runs"]
    assert set(st) == {"asha", "meera", "rohan"}
    n = len(st["asha"]["equity"])
    assert n > 50 and len(st["asha"]["benchmark_equity"]) == n and st["asha"]["dates"][0] > "2025"
    assert st["asha"]["metrics"]["avg_exposure"] < st["rohan"]["metrics"]["avg_exposure"]      # profile changes behaviour
    # causality: signals at t are formed from data <= t and held over [t, t+5]; the first equity point cannot depend on the future
    grid = engine.backtest["_grid"]
    t0 = min(grid)
    r = strategy.run(engine.data, grid, {"risk_tolerance": 9, "max_position_pct": 30})
    assert r["dates"][0] == engine.data.dates[t0 + strategy.STEP].isoformat()


# ====================================================================== Ext 2: vector DB & larger corpus
def test_corpus_is_large_and_time_structured(engine):
    s = engine.rag.stats()
    assert s["passages"] > 450 and s["documents"] > 250 and s["companies"] == 12
    assert s["date_range"][0] < "2024-12" and s["date_range"][1] > "2026-10"


def test_qdrant_persists_vectors_and_falls_back_when_locked():
    d = tempfile.mkdtemp()
    first = RagIndex(backend="qdrant", qdrant_path=os.path.join(d, "q"))
    assert first.stats()["backend"].startswith("Qdrant") and not first.reused
    second = RagIndex(backend="qdrant", qdrant_path=os.path.join(d, "q"))          # same folder, same process -> locked
    assert second.stats()["backend"].startswith("numpy") and "Qdrant unavailable" in second.stats()["note"]
    a = [h.chunk.chunk_id for h in first.search("promoter pledge", symbol="VIKRAMINFRA", k=5)]
    b = [h.chunk.chunk_id for h in second.search("promoter pledge", symbol="VIKRAMINFRA", k=5)]
    assert a == b                                                                   # both backends agree
    first.backend.close()
    third = RagIndex(backend="qdrant", qdrant_path=os.path.join(d, "q"))
    assert third.reused and third.stats()["persisted"]                              # vectors reloaded from disk
    assert [h.chunk.chunk_id for h in third.search("promoter pledge", symbol="VIKRAMINFRA", k=5)] == a
    third.backend.close()


def test_payload_filters(engine):
    r = engine.rag
    hits = r.search("dividend", symbol="NAVBANK", sector="Banking", k=20, doc_types=["SEBI Filing"])
    assert hits and all(h.chunk.doc_type == "SEBI Filing" and h.chunk.symbol in ("NAVBANK", None) for h in hits)
    recent = r.search("dividend", symbol="NAVBANK", sector="Banking", k=20, date_from=dt.date(2026, 6, 1))
    assert recent and all(h.chunk.date >= "2026-06-01" for h in recent)


def test_dense_retrieval_helps_on_paraphrases(engine):
    ev = evaluate(engine.rag)
    para, lex = ev["sets"]["paraphrase"], ev["sets"]["lexical"]
    assert para["vector"]["recall@3"] >= para["bm25"]["recall@3"]
    assert para["hybrid"]["mrr"] >= para["bm25"]["mrr"]
    assert lex["hybrid"]["recall@3"] >= 0.85
    hit = engine.rag.search("watchdog questioned undisclosed dealings with affiliates", k=3, mode="vector")
    assert hit[0].chunk.chunk_id == "VIKRAMINFRA-LODR-R30#0"                        # no shared keywords, still found


def test_filings_agent_reports_disclosure_trend(engine):
    r = engine.analyze("asha", persist=False)
    by = {x["symbol"]: x["agents"]["filings"]["data"]["trend"] for x in r["recommendations"]}
    assert by["VIKRAMINFRA"]["direction"] == "deteriorating" and by["INFOSPHERE"]["direction"] == "improving"


def test_search_api_modes():
    from fastapi.testclient import TestClient
    from app.server import app
    c = TestClient(app)
    j = c.get("/api/rag/search", params={"q": "show-cause notice", "mode": "hybrid", "symbol": "VIKRAMINFRA"}).json()
    assert j["hits"] and j["backend"] and all(h["symbol"] in ("VIKRAMINFRA", None) for h in j["hits"])
    assert c.get("/api/rag/search", params={"q": "x", "mode": "nope"}).status_code == 400
    assert c.get("/api/rag/stats").json()["eval"]["n_queries"] == 14


# ====================================================================== Ext 1: live sources (mocked HTTP)
class FakeResp:
    def __init__(self, status=200, payload=None, text=""):
        self.status_code, self._p, self.content = status, payload, text.encode()

    def json(self):
        return self._p


def _yahoo(n=120, trend=0.002):
    end = dt.date(2026, 10, 14)
    days = [end - dt.timedelta(days=i) for i in range(n * 2) if (end - dt.timedelta(days=i)).weekday() < 5][:n][::-1]
    rng = np.random.default_rng(1)
    close = 100 * np.exp(np.cumsum(trend + rng.normal(0, 0.01, n)))
    vol = (1e6 * (1 + rng.random(n))).tolist()
    vol[-1] = 0                                                        # in-progress bar: must be dropped
    ts = [int(dt.datetime(d.year, d.month, d.day, 4).timestamp()) for d in days]
    return {"chart": {"result": [{"meta": {"longName": "Test Co Ltd", "gmtoffset": 19800, "regularMarketPrice": float(close[-1])},
                                  "timestamp": ts, "events": {"dividends": {"1": {"amount": 6.0, "date": ts[-5]}}},
                                  "indicators": {"quote": [{"close": close.tolist(), "volume": vol}]}}]}}


NSE = [{"an_dt": "08-Oct-2026 15:41:52", "desc": "Dividend", "attchmntText": "Test Co Ltd has informed the Exchange that Board declared Interim Dividend of Rs 30 per share and record growth in profit",
        "attchmntFile": "https://nsearchives.nseindia.com/corporate/x1.pdf"},
       {"an_dt": "01-Oct-2026 10:00:00", "desc": "Financial Results", "attchmntText": "Outcome of Board Meeting: results for the quarter, beat estimates with strong demand and robust order book",
        "attchmntFile": "https://nsearchives.nseindia.com/corporate/x2.pdf"},
       {"an_dt": "20-Sep-2026 10:00:00", "desc": "Analysts/Institutional Investor Meet/Con. Call Updates", "attchmntText": "Schedule of earnings conference call with analysts for the quarter",
        "attchmntFile": "https://nsearchives.nseindia.com/corporate/x3.pdf"}]
RSS = """<?xml version="1.0"?><rss><channel>{items}</channel></rss>"""
ITEM = "<item><title>{t} - Mint</title><link>https://news.example/{i}</link><pubDate>{d}</pubDate><source>Mint</source></item>"


def _rss():
    d = dt.datetime.now(dt.timezone.utc).strftime("%a, %d %b %Y %H:%M:%S GMT")
    titles = ["Test Co beats estimates as profit jumps on strong demand", "Brokerages raise target on Test Co citing robust order inflow"]
    return RSS.format(items="".join(ITEM.format(t=t, i=i, d=d) for i, t in enumerate(titles)))


class FakeHttp:
    def __init__(self, fail=()):
        self.fail, self.urls = set(fail), []

    def get(self, url, headers=None, timeout=None):
        self.urls.append(url)
        if "raise" in self.fail:
            raise ConnectionError("offline")
        for key in self.fail:
            if key in url:
                return FakeResp(403)
        if "yahoo" in url:
            return FakeResp(200, _yahoo())
        if "nseindia" in url:
            return FakeResp(200, NSE)
        if "news.google" in url:
            return FakeResp(200, text=_rss())
        return FakeResp(404)


def test_live_pipeline_runs_same_agents_on_live_sources(engine):
    out = engine.live_analyze("meera", ["TESTCO"], http=FakeHttp())
    t = out["tickers"][0]
    r = t["recommendation"]
    assert out["agents_usable"] == out["agents_total"] == 6
    assert all(s["ok"] for s in out["sources"].values()) and len(out["sources"]) == 3
    assert t["name"] == "Test Co Ltd" and t["price"] > 0 and r["grounded"]
    urls = {c["url"] for c in r["citations"].values() if c["url"]}
    assert any("nsearchives" in u for u in urls) and any("news.example" in u for u in urls) and any("yahoo" in u for u in urls)
    assert all(c["verified"] for c in r["citations"].values() if c["kind"] == "filing")
    assert any("Dividend" in e["kind"] for e in t["earnings_and_actions"])           # earnings/corporate actions surfaced
    assert r["agents"]["momentum"]["status"] == "ok"                                  # zero-volume bar dropped, nothing flagged


@pytest.mark.parametrize("fail,expect", [(["nseindia"], "filings"), (["news.google"], "sentiment"), (["yahoo"], "momentum"),
                                         (["raise"], "momentum")])
def test_live_source_failure_degrades_only_dependants(engine, fail, expect):
    out = engine.live_analyze("meera", ["TESTCO"], http=FakeHttp(fail))
    a = out["tickers"][0]["recommendation"]["agents"]
    assert a[expect]["status"] in ("abstained", "failed")
    if expect == "filings":
        assert a["momentum"]["status"] == "ok" and not out["tickers"][0]["recommendation"]["grounded"]
    if expect == "sentiment":
        assert a["filings"]["status"] == "ok"
    assert all(not o for o in [AgentOutput.validate(AgentOutput(**{k: v for k, v in x.items() if k in AgentOutput.__dataclass_fields__ and k not in ("evidence", "citations")}))
                               for x in a.values() if x["status"] not in ("ok", "degraded")])


def test_live_total_outage_does_not_crash(engine):
    out = engine.live_analyze("asha", ["AAA", "BBB"], http=FakeHttp(["raise"]))
    assert len(out["tickers"]) == 2 and out["agents_usable"] <= 4                      # only user-side agents may survive
    assert all(t["recommendation"]["action"] in ("WATCH", "HOLD", "AVOID") for t in out["tickers"])
    assert not any(s["ok"] for s in out["sources"].values())


def test_live_endpoint_validates_tickers():
    from fastapi.testclient import TestClient
    from app.server import app
    c = TestClient(app)
    assert c.post("/api/live", json={"user_id": "asha", "tickers": ["bad ticker!"]}).status_code == 400
    assert c.post("/api/live", json={"user_id": "ghost", "tickers": ["TCS"]}).status_code in (404, 200)

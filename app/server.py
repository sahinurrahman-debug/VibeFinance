"""FastAPI server: JSON API + the single-page dashboard."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field

from . import config
from .alerts import DEFAULT_RULES, RULE_HELP
from .backtest import public as bt_public
from .bootstrap import build_engine
from .live import DEFAULT_TICKERS, TICKER_RE
from .market import BY_SYMBOL, SYMBOLS, UNIVERSE

STATIC = Path(__file__).parent / "static"
app = FastAPI(title="VibeFinance", version="1.0")
ENGINE = build_engine()


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if hasattr(o, "isoformat"):
        return o.isoformat()
    raise TypeError(type(o))


def ok(obj) -> JSONResponse:
    return JSONResponse(json.loads(json.dumps(obj, default=_default)))


def _valid_degr(d: list[str]) -> list[str]:
    bad = [x for x in d if x not in config.DEGRADATIONS and not x.startswith("filings_missing:")]
    if bad:
        raise HTTPException(400, f"unknown degradation(s): {bad}")
    return d


class AnalyzeReq(BaseModel):
    user_id: str
    degradations: list[str] = Field(default_factory=list)


class CompareReq(BaseModel):
    symbol: str
    degradations: list[str] = Field(default_factory=list)


class TradeReq(BaseModel):
    symbol: str
    side: str
    qty: int = Field(gt=0)


class DecisionReq(BaseModel):
    user_id: str
    rec_id: int | None = None
    symbol: str
    action: str
    choice: str          # accepted | ignored | modified
    note: str = ""


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/meta")
def meta():
    return ok({"as_of": ENGINE.data.dates[-1], "universe": [s.__dict__ for s in UNIVERSE],
               "users": ENGINE.store.list_users(), "degradations": config.DEGRADATIONS,
               "backtest_meta": ENGINE.backtest["_meta"], "rag": ENGINE.rag.stats(), "live_defaults": DEFAULT_TICKERS,
               "alert_rule_help": RULE_HELP, "agent_timeout_s": config.AGENT_TIMEOUT_S,
               "disclosure": "All companies, prices, headlines, flows and filings are SYNTHETIC. Backtest figures validate pipeline "
                             "mechanics on simulated data, not real-market performance. Not investment advice."})


@app.post("/api/analyze")
def analyze(req: AnalyzeReq):
    _valid_degr(req.degradations)
    try:
        return ok(ENGINE.analyze(req.user_id, req.degradations))
    except KeyError:
        raise HTTPException(404, "unknown user")


@app.post("/api/compare")
def compare(req: CompareReq):
    if req.symbol not in BY_SYMBOL:
        raise HTTPException(404, "unknown symbol")
    return ok(ENGINE.compare(req.symbol, _valid_degr(req.degradations)))


@app.get("/api/users/{uid}")
def get_user(uid: str):
    u = ENGINE.store.get_user(uid)
    if not u:
        raise HTTPException(404, "unknown user")
    return ok({"user": u, "holdings": ENGINE.store.holdings(uid), "watchlist": ENGINE.store.watchlist(uid)})


@app.put("/api/users/{uid}")
def update_user(uid: str, fields: dict):
    if not ENGINE.store.get_user(uid):
        raise HTTPException(404, "unknown user")
    rt = fields.get("risk_tolerance")
    if rt is not None and not (1 <= int(rt) <= 10):
        raise HTTPException(400, "risk_tolerance must be 1-10")
    return ok(ENGINE.store.update_user(uid, fields))


@app.put("/api/users/{uid}/watchlist")
def set_watchlist(uid: str, symbols: list[str]):
    if any(s not in BY_SYMBOL for s in symbols):
        raise HTTPException(400, "unknown symbol")
    ENGINE.store.set_watchlist(uid, symbols)
    return ok(ENGINE.store.watchlist(uid))


@app.post("/api/users/{uid}/trades")
def add_trade(uid: str, t: TradeReq):
    if t.symbol not in BY_SYMBOL:
        raise HTTPException(404, "unknown symbol")
    try:
        return ok(ENGINE.store.apply_trade(uid, ENGINE.data.dates[-1].isoformat(), t.symbol, t.side.upper(), t.qty,
                                           ENGINE.feed.last_close(t.symbol), "manual"))
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/api/users/{uid}/history")
def history(uid: str):
    return ok({"sessions": ENGINE.store.sessions(uid), "decisions": ENGINE.store.decisions(uid),
               "trades": ENGINE.store.trades(uid)[-25:]})


@app.post("/api/decisions")
def decision(d: DecisionReq):
    did = ENGINE.store.save_decision(d.user_id, d.rec_id, d.symbol, d.action, d.choice, d.note)
    return ok({"id": did})


@app.get("/api/docs/{doc_id}")
def get_doc(doc_id: str):
    d = ENGINE.rag.doc(doc_id)
    if not d:
        raise HTTPException(404, "document not found")
    return ok(d.meta())


@app.get("/api/backtest")
def backtest():
    return ok({**bt_public(ENGINE.backtest), "calibration_model": ENGINE.calibrator.summary()})


@app.get("/api/strategy")
def strategy_bt():
    return ok(ENGINE.strategy())


@app.get("/api/users/{uid}/timeline")
def user_timeline(uid: str):
    if not ENGINE.store.get_user(uid):
        raise HTTPException(404, "unknown user")
    return ok(ENGINE.timeline(uid))


@app.put("/api/users/{uid}/alert_rules")
def set_rules(uid: str, rules: dict):
    if not ENGINE.store.get_user(uid):
        raise HTTPException(404, "unknown user")
    bad = [k for k in rules if k not in DEFAULT_RULES]
    if bad:
        raise HTTPException(400, f"unknown rules {bad}")
    ENGINE.store.set_alert_rules(uid, rules, DEFAULT_RULES)
    return ok(ENGINE.store.alert_rules(uid, DEFAULT_RULES))


@app.get("/api/rag/stats")
def rag_stats():
    return ok({**ENGINE.rag.stats(), "eval": ENGINE.rag_eval})


@app.get("/api/rag/search")
def rag_search(q: str, symbol: str | None = None, doc_type: str | None = None, date_from: str | None = None,
               mode: str = "hybrid", k: int = 8, recency: bool = False):
    import datetime as dt
    if mode not in ("hybrid", "bm25", "vector"):
        raise HTTPException(400, "mode must be hybrid|bm25|vector")
    sector = BY_SYMBOL[symbol].sector if symbol in BY_SYMBOL else None
    df = dt.date.fromisoformat(date_from) if date_from else None
    t0 = __import__("time").perf_counter()
    hits = ENGINE.rag.search(q, symbol=symbol or None, sector=sector, k=min(k, 25), doc_types=[doc_type] if doc_type else None,
                             date_from=df, mode=mode, recency=recency)
    ms = (__import__("time").perf_counter() - t0) * 1000
    return ok({"query": q, "mode": mode, "backend": ENGINE.rag.backend.name, "ms": round(ms, 1),
               "hits": [{"chunk_id": h.chunk.chunk_id, "doc_id": h.chunk.doc_id, "symbol": h.chunk.symbol, "title": h.chunk.title,
                         "doc_type": h.chunk.doc_type, "date": h.chunk.date, "period": h.chunk.period, "heading": h.chunk.heading,
                         "text": h.chunk.text, "score": round(h.score, 3), "bm25": round(h.bm25, 3), "vector": round(h.cosine, 3)}
                        for h in hits]})


class LiveReq(BaseModel):
    user_id: str
    tickers: list[str] = Field(default_factory=lambda: list(DEFAULT_TICKERS))
    degradations: list[str] = Field(default_factory=list)


@app.post("/api/live")
def live(req: LiveReq):
    tickers = [t.strip().upper() for t in req.tickers if t.strip()][:8]
    if not tickers or any(not TICKER_RE.match(t) for t in tickers):
        raise HTTPException(400, "tickers must be NSE symbols such as RELIANCE, TCS, M&M")
    _valid_degr(req.degradations)
    try:
        return ok(ENGINE.live_analyze(req.user_id, tickers, req.degradations))
    except KeyError:
        raise HTTPException(404, "unknown user")


@app.get("/api/benchmark")
def benchmark(repeats: int = 2):
    return ok({"io_scale": ENGINE.feed.io_scale, "rows": ENGINE.benchmark(repeats=max(1, min(repeats, 5)))})


@app.get("/api/benchmark/load")
def benchmark_load():
    return ok({"io_scale": ENGINE.feed.io_scale, "rows": ENGINE.load_benchmark(), "pool": ENGINE.pool_sweep()})


@app.get("/api/prices/{symbol}")
def prices(symbol: str, n: int = 120):
    if symbol not in BY_SYMBOL:
        raise HTTPException(404, "unknown symbol")
    d = ENGINE.data
    return ok({"symbol": symbol, "dates": d.dates[-n:], "close": d.close[symbol][-n:].round(2),
               "volume": d.volume[symbol][-n:]})


@app.get("/api/architecture")
def architecture():
    p = Path(__file__).parent.parent / "ARCHITECTURE.md"
    return PlainTextResponse(p.read_text(encoding="utf-8") if p.exists() else "ARCHITECTURE.md not found")

"""Wires the whole system together (used by the server, CLI and tests)."""
from __future__ import annotations

import time
from pathlib import Path

from . import config
from .backtest import run_backtest
from .engine import Engine
from .market import UNIVERSE, MarketData, MarketFeed
from .rag import RagIndex
from .rag_eval import evaluate
from .seed import seed
from .store import Store


def build_engine(db_path: Path | str | None = None, io_scale: float | None = None, reset: bool = False,
                 vector: str | None = None, qdrant_path: Path | str | None = None) -> Engine:
    path = Path(db_path or config.DB_PATH)
    if reset and path.exists():
        path.unlink()
    data = MarketData()
    store = Store(path)
    store.ensure_instruments(UNIVERSE)
    if store.is_empty():
        seed(store, data)
    feed = MarketFeed(data, config.IO_SCALE if io_scale is None else io_scale)
    vector = vector or config.VECTOR_BACKEND
    if vector == "auto":                       # persistent on-disk Qdrant for the real app, in-memory for scratch DBs
        vector = "qdrant" if db_path is None else "qdrant-memory"
    rag = RagIndex(backend=vector, qdrant_path=qdrant_path or config.QDRANT_PATH)
    t = time.perf_counter()
    bt = run_backtest(data)
    bt["_meta"]["compute_s"] = round(time.perf_counter() - t, 2)
    from .agents import FilingsAgent
    from .market import BY_SYMBOL, SYMBOLS
    t = time.perf_counter()
    for sym in SYMBOLS:                                 # warm the retrieval cache: the first live analysis must not pay for it
        for q in FilingsAgent.QUERIES:
            rag.search(q, symbol=sym, sector=BY_SYMBOL[sym].sector, k=3, recency=FilingsAgent.RECENCY_TAU)
    rag.warm_ms = (time.perf_counter() - t) * 1000
    eng = Engine(store, data, feed, rag, bt)
    eng.rag_eval = evaluate(rag)
    return eng

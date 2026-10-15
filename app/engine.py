"""Orchestrator + end-to-end pipeline.

`run_agents` executes every (symbol, agent) task concurrently on a shared thread pool with a deadline per
*running* task. A task that raises, violates the output contract, or misses its deadline is replaced by a
well-formed 'no signal' output, so one bad agent can never fail the pipeline. A per-run single-flight request
cache stops agents that need the same feed payload from each paying for it.
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, wait
from statistics import mean

from . import alerts as alerts_mod
from . import config, portfolio, strategy, timeline
from .agents import ALL_AGENTS, SIGNAL_AGENTS, USER_AGENTS, AgentContext
from .calibration import Calibrator
from .contracts import AgentOutput, failed_output
from .market import SYMBOLS, MarketData, MarketFeed, get_stock
from .rag import RagIndex
from .runcache import RunCache
from .store import Store
from .synthesis import SIGNALS, synthesize

POOL = ThreadPoolExecutor(max_workers=config.MAX_WORKERS, thread_name_prefix="agent")
ACTION_PRIORITY = {"SELL": 0, "TRIM": 1, "BUY": 2, "ACCUMULATE": 3, "AVOID": 6, "HOLD": 5, "WATCH": 4}


def _execute(agent, ctx: AgentContext, t0: float, started: dict | None = None, key=None) -> AgentOutput:
    start = time.perf_counter()
    if started is not None:
        started[key] = start
    try:
        out = agent.run(ctx)
        problems = out.validate()
        if problems:
            out = failed_output(agent.name, ctx.symbol, agent.kind, "failed",
                                "Output rejected by contract validation: " + "; ".join(problems[:2]),
                                ["contract violation"])
    except Exception as e:                      # noqa: BLE001 - isolation boundary
        out = failed_output(agent.name, ctx.symbol, agent.kind, "failed", f"Agent crashed: {type(e).__name__}: {e}")
    out.started_ms = (start - t0) * 1000
    out.latency_ms = (time.perf_counter() - start) * 1000
    return out


def run_agents(tasks: list[tuple[tuple, object, AgentContext]], parallel: bool = True,
               timeout_s: float | None = None, pool: ThreadPoolExecutor | None = None) -> tuple[dict, float]:
    """tasks: [(key, agent, ctx)] -> ({key: AgentOutput}, wall_ms).

    The deadline applies to a task's *running* time: time spent queued behind other analyses (under load) does
    not count against an agent, so concurrency never causes spurious timeouts.
    """
    timeout_s = timeout_s if timeout_s is not None else config.AGENT_TIMEOUT_S
    t0 = time.perf_counter()
    results: dict = {}
    if parallel:
        started: dict = {}
        futs = {(pool or POOL).submit(_execute, a, c, t0, started, k): (k, a, c) for k, a, c in tasks}
        pending = set(futs)
        while pending:
            done, pending = wait(pending, timeout=0.02)
            for f in done:
                results[futs[f][0]] = f.result()
            now = time.perf_counter()
            for f in list(pending):
                k, a, c = futs[f]
                if k in started and now - started[k] > timeout_s:
                    o = failed_output(a.name, c.symbol, a.kind, "timeout",
                                      f"{a.name} agent exceeded the {timeout_s:.1f}s deadline and was skipped",
                                      [f"{a.name} timed out"])
                    o.started_ms, o.latency_ms = (started[k] - t0) * 1000, timeout_s * 1000
                    results[k] = o
                    pending.discard(f)
    else:
        for k, a, c in tasks:
            results[k] = _execute(a, c, t0)
    return results, (time.perf_counter() - t0) * 1000


class Engine:
    def __init__(self, store: Store, data: MarketData, feed: MarketFeed, rag: RagIndex, backtest: dict):
        self.store, self.data, self.feed, self.rag, self.backtest = store, data, feed, rag, backtest
        self.calibrator = Calibrator(backtest)
        self.use_cache = True
        self._strategy: dict = {}
        self.live_feed = None

    # ------------------------------------------------------------------ helpers
    def _user_bundle(self, uid: str, degr: frozenset):
        user = self.store.get_user(uid)
        if not user:
            raise KeyError(uid)
        holdings = self.store.holdings(uid)
        prices = {s: self.feed.last_close(s) for s in SYMBOLS}
        note = "Valuation uses last cached close (price feed down)" if "price_feed_down" in degr else ""
        snap = portfolio.snapshot(user, holdings, prices, self.data, note)
        behavior = portfolio.behavior_profile(self.store.trades(uid), self.data)
        return user, holdings, prices, snap, behavior

    def _ctx(self, sym, degr, user=None, snap=None, behavior=None, decisions=None, cache=None, feed=None, rag=None) -> AgentContext:
        return AgentContext(sym, feed or self.feed, rag or self.rag, self.data, degr, user, snap, behavior, decisions or [], cache)

    @staticmethod
    def _split(results: dict, sym: str) -> dict[str, AgentOutput]:
        return {a: results[(sym, a)] for a in ALL_AGENTS if (sym, a) in results}

    # ------------------------------------------------------------------ main pipeline
    def analyze(self, uid: str, degradations=(), symbols=None, persist: bool = True) -> dict:
        t_all = time.perf_counter()
        degr = frozenset(degradations)
        symbols = list(symbols or SYMBOLS)
        trace = []

        t = time.perf_counter()
        user, holdings, prices, snap, behavior = self._user_bundle(uid, degr)
        decisions = self.store.decisions(uid, 30)
        watch = self.store.watchlist(uid)
        rules = self.store.alert_rules(uid, alerts_mod.DEFAULT_RULES)
        prev = self.store.snapshots(uid, 1)
        trace.append({"stage": "1. Ingest & profile", "ms": (time.perf_counter() - t) * 1000,
                      "detail": f"Loaded {user['name']}'s profile, {len(holdings)} holdings, {len(watch)} watchlist names, "
                                f"{behavior['n_trades']} trades; archetype from trade log: {behavior['archetype']}. "
                                f"Active degradations: {', '.join(sorted(degr)) or 'none'}"})

        tasks = []
        cache = RunCache() if self.use_cache else None
        calls0 = dict(self.feed.calls)
        for s in symbols:
            ctx = self._ctx(s, degr, user, snap, behavior, decisions, cache)
            for a, ag in {**SIGNAL_AGENTS, **USER_AGENTS}.items():
                tasks.append(((s, a), ag, ctx))
        results, wall_ms = run_agents(tasks, parallel=True)
        trace.append({"stage": "2. Parallel agents", "ms": wall_ms,
                      "detail": f"{len(tasks)} agent tasks ({len(symbols)} stocks x 6 agents) on a shared pool; "
                                f"sum of agent latencies {sum(o.latency_ms for o in results.values()):.0f} ms vs wall {wall_ms:.0f} ms"})

        t = time.perf_counter()
        recs = [synthesize(user, s, self._split(results, s), prices[s], snap, self.calibrator, degr) for s in symbols]
        recs.sort(key=lambda r: (ACTION_PRIORITY[r["action"]], -abs(r["composite"])))
        trace.append({"stage": "3. Synthesis & personalisation", "ms": (time.perf_counter() - t) * 1000,
                      "detail": f"Weighted composite, conflict detection, calibration, counterfactuals, risk gate and behaviour guard for {len(recs)} stocks"})

        t = time.perf_counter()
        trades = [{"symbol": r["symbol"], "side": r["sizing"]["side"], "qty": r["sizing"]["qty"]}
                  for r in recs if r["sizing"]]
        u2, h2 = portfolio.apply_hypothetical(user, holdings, trades, prices)
        after = portfolio.snapshot(u2, h2, prices, self.data)
        signals = self._signal_table(results, symbols)
        prev_ids = prev[-1]["payload"]["alert_ids"] if prev else None
        ev = alerts_mod.evaluate(user, rules, snap, recs, watch, signals, prev_ids)
        trace.append({"stage": "4. Portfolio risk & alerts", "ms": (time.perf_counter() - t) * 1000,
                      "detail": f"Concentration {snap['concentration_score']:.0f} -> {after['concentration_score']:.0f} if all suggested trades executed; "
                                f"top risk contributor {snap['risk'].get('top_risk_symbol', '-')} ({snap['risk'].get('top_risk_share', 0):.0%}); "
                                f"{len(ev['alerts'])} alerts ({sum(1 for a in ev['alerts'] if a['is_new'])} new, {len(ev['resolved_ids'])} resolved)"})

        metrics = self._metrics(results, wall_ms, recs, snap, after, t_all, degr)
        metrics["feed_calls"] = {k: v - calls0.get(k, 0) for k, v in self.feed.calls.items() if v - calls0.get(k, 0)}
        metrics["cache"] = {"enabled": self.use_cache, "hits": cache.hits if cache else 0, "misses": cache.misses if cache else 0}
        metrics["calibration"] = {k: self.calibrator.report.get(k) for k in ("ece_raw", "ece_calibrated", "brier_raw", "brier_calibrated", "brier_climatology")}

        t = time.perf_counter()
        sid = None
        if persist:
            sid = self.store.save_session(uid, sorted(degr), metrics, {
                "actions": {r["symbol"]: r["action"] for r in recs}, "concentration_before": snap["concentration_score"]})
            for r in recs:
                r["rec_id"] = self.store.save_recommendation(sid, uid, r)
            self.store.save_snapshot(sid, uid, timeline.make_snapshot(user, snap, behavior, recs, ev["ids"], decisions))
        trace.append({"stage": "5. Persist & log", "ms": (time.perf_counter() - t) * 1000,
                      "detail": f"Session {sid}: recommendations, portfolio snapshot and metrics stored in SQLite; metrics appended to logs/metrics.jsonl" if persist else "not persisted"})
        metrics["e2e_ms"] = round((time.perf_counter() - t_all) * 1000, 1)
        if persist:
            self._log(uid, sid, metrics)

        gantt = [{"symbol": k[0], "agent": k[1], "start": round(o.started_ms, 1), "ms": round(o.latency_ms, 1),
                  "status": o.status} for k, o in results.items()]
        return {"session_id": sid, "as_of": self.data.dates[-1].isoformat(), "user": user, "behavior": behavior,
                "portfolio": snap, "portfolio_after": after, "recommendations": recs, "alerts": ev["alerts"],
                "alerts_meta": {"resolved_ids": ev["resolved_ids"], "rules": rules, "first_session": not prev},
                "watchlist": watch, "signals": signals, "metrics": metrics,
                "trace": [{**x, "ms": round(x["ms"], 1)} for x in trace], "gantt": gantt, "degradations": sorted(degr)}

    def compare(self, symbol: str, degradations=(), users: list[dict] | None = None) -> dict:
        """Identical market inputs, several investors: proves personalisation.

        `users` is an explicit list (the signed-in user plus the public demo personas). It never
        defaults to "all users", so one customer's profile cannot leak into another's view.
        """
        degr = frozenset(degradations)
        users = users if users is not None else self.store.list_templates()
        bundles = {u["id"]: self._user_bundle(u["id"], degr) for u in users}
        tasks = []
        cache = RunCache() if self.use_cache else None
        base_ctx = self._ctx(symbol, degr, cache=cache)
        for a, ag in SIGNAL_AGENTS.items():
            tasks.append(((symbol, a), ag, base_ctx))
        for u in users:
            user, _, _, snap, beh = bundles[u["id"]]
            ctx = self._ctx(symbol, degr, user, snap, beh, self.store.decisions(u["id"], 30), cache)
            for a, ag in USER_AGENTS.items():
                tasks.append(((u["id"], a), ag, ctx))
        results, wall = run_agents(tasks)
        market = {a: results[(symbol, a)] for a in SIGNAL_AGENTS}
        out = []
        for u in users:
            user, _, prices, snap, _ = bundles[u["id"]]
            outs = {**market, "risk": results[(u["id"], "risk")], "behavior": results[(u["id"], "behavior")]}
            rec = synthesize(user, symbol, outs, prices[symbol], snap, self.calibrator, degr)
            out.append({"user": user, "recommendation": rec})
        return {"symbol": symbol, "name": get_stock(symbol).name, "wall_ms": round(wall, 1),
                "market_inputs": {a: o.to_dict() for a, o in market.items()}, "results": out}

    # ------------------------------------------------------------------ persistent sessions
    def timeline(self, uid: str) -> dict:
        snaps = self.store.snapshots(uid, 30)
        t = timeline.build(snaps, self.store.decisions(uid, 200), self.store.sessions(uid, 30))
        t["alert_rules"] = self.store.alert_rules(uid, alerts_mod.DEFAULT_RULES)
        t["rule_help"] = alerts_mod.RULE_HELP
        return t

    # ------------------------------------------------------------------ strategy back-test
    def strategy(self, users: list[dict] | None = None) -> dict:
        users = users if users is not None else self.store.list_templates()
        key = tuple((u["id"], u["risk_tolerance"], u["max_position_pct"]) for u in users)
        if key not in self._strategy:
            if len(self._strategy) >= 32:
                self._strategy.pop(next(iter(self._strategy)))
            self._strategy[key] = strategy.run_profiles(self.data, self.backtest["_grid"], users)
        return {"runs": self._strategy[key], "cost_per_turnover": strategy.COST, "rebalance_sessions": strategy.STEP,
                "caveat": "Simulated trending data inflates absolute returns; read the strategies relative to the benchmark and to each other."}

    # ------------------------------------------------------------------ live data
    def live_analyze(self, uid: str, tickers: list[str], degradations=(), http=None) -> dict:
        from .live import LiveFeed, register
        from .market import FeedUnavailable
        degr = frozenset(degradations)
        t_all = time.perf_counter()
        if http is not None or self.live_feed is None:
            self.live_feed = LiveFeed(http)
        lf = self.live_feed
        user, holdings, prices, snap, behavior = self._user_bundle(uid, frozenset())
        decisions = self.store.decisions(uid, 30)
        # 1. fetch disclosures for all tickers into one live RAG index
        docs, doc_by, fetch_errors = [], {}, {}

        def fetch_docs(tk):
            try:
                return tk, lf.documents(tk), None
            except FeedUnavailable as e:
                return tk, [], str(e)

        for tk, ds, err in POOL.map(fetch_docs, tickers):          # all tickers' disclosures in parallel
            docs += ds
            doc_by[tk] = ds
            if err:
                fetch_errors[tk] = err
        rag = RagIndex(docs, backend="numpy")

        def prime(tk):                                              # prices + names, also in parallel
            try:
                lf._chart(tk)
                register(tk, lf)
            except FeedUnavailable:
                register_fallback(tk)

        list(POOL.map(prime, tickers))
        cache = RunCache()
        tasks = []
        for tk in tickers:
            ctx = self._ctx(tk, degr, user, snap, behavior, decisions, cache, feed=lf, rag=rag)
            for a, ag in {**SIGNAL_AGENTS, **USER_AGENTS}.items():
                tasks.append(((tk, a), ag, ctx))
        results, wall = run_agents(tasks, timeout_s=max(config.AGENT_TIMEOUT_S, 12.0))
        out = []
        for tk in tickers:
            outs = self._split(results, tk)
            st = get_stock(tk)
            price = None
            try:
                price = lf.last_close(tk)
            except FeedUnavailable:
                pass
            rec = synthesize(user, tk, outs, price or 0.0, snap, self.calibrator, degr)
            ps = None
            try:
                ps = lf.prices(tk, degr).data
            except FeedUnavailable:
                pass
            out.append({"symbol": tk, "name": st.name, "price": price,
                        "chg_1d": float(ps.close[-1] / ps.close[-2] - 1) if ps is not None else None,
                        "chg_20d": float(ps.close[-1] / ps.close[-21] - 1) if ps is not None else None,
                        "spark": [round(float(x), 2) for x in ps.close[-60:]] if ps is not None else [],
                        "recommendation": rec, "earnings_and_actions": lf.earnings_and_actions(tk, doc_by.get(tk, [])),
                        "headlines": self._live_headlines(lf, tk, degr), "n_disclosures": len(doc_by.get(tk, []))})
        usable = sum(1 for o in results.values() if o.usable)
        return {"as_of": dt_today(), "user": {"id": user["id"], "name": user["name"], "risk_tolerance": user["risk_tolerance"]},
                "tickers": out, "sources": lf.status, "fetch_errors": fetch_errors, "wall_ms": round(wall, 1),
                "e2e_ms": round((time.perf_counter() - t_all) * 1000, 1), "agents_total": len(results), "agents_usable": usable,
                "limits": ["FII flows and option-chain PCR have no keyless live source: sentiment uses headlines only.",
                           "Filing tone is scored on NSE announcement subjects; attachments are linked, not parsed.",
                           "Prices are daily bars from Yahoo Finance and may be delayed."]}

    @staticmethod
    def _live_headlines(lf, tk, degr):
        try:
            name = lf.company_name(tk)
            return [h.to_dict() | {"url": h.url} for h in lf.news(tk, name)][:6] if "sentiment_feed_down" not in degr else []
        except Exception:                                # noqa: BLE001
            return []

    # ------------------------------------------------------------------ derived views
    def _signal_table(self, results: dict, symbols) -> list[dict]:
        rows = []
        for s in symbols:
            st = get_stock(s)
            row = {"symbol": s, "name": st.name, "sector": st.sector, "price": self.feed.last_close(s), "blurb": st.blurb}
            for a in SIGNALS:
                o = results[(s, a)]
                row[a] = {"label": o.label, "score": round(o.score, 3), "confidence": round(o.confidence, 3),
                          "status": o.status, "summary": o.summary}
            ch = self.data.close[s]
            row["chg_1d"], row["chg_20d"] = float(ch[-1] / ch[-2] - 1), float(ch[-1] / ch[-21] - 1)
            row["spark"] = [round(float(x), 2) for x in ch[-60:]]
            rows.append(row)
        return rows

    def _metrics(self, results, wall_ms, recs, before, after, t_all, degr) -> dict:
        outs = list(results.values())
        lat = [o.latency_ms for o in outs if o.status != "timeout"]
        per_agent = {}
        for a in ALL_AGENTS:
            ls = [o.latency_ms for k, o in results.items() if k[1] == a and o.status != "timeout"]
            per_agent[a] = {"mean_ms": round(mean(ls), 1) if ls else None, "max_ms": round(max(ls), 1) if ls else None,
                            "ok": sum(1 for k, o in results.items() if k[1] == a and o.usable),
                            "total": sum(1 for k in results if k[1] == a)}
        usable = [o for o in outs if o.usable]
        ev_total = ev_cited = 0
        for o in usable:
            ids = {c.source_id for c in o.citations}
            for e in o.evidence:
                ev_total += 1
                ev_cited += bool(e.source_id and e.source_id in ids)
        filing_cits = [c for o in usable for c in o.citations if c.kind == "filing"]
        ens = self.backtest["ensemble"]
        return {
            "e2e_ms": round((time.perf_counter() - t_all) * 1000, 1),
            "agent_wall_ms": round(wall_ms, 1),
            "agent_latency_sum_ms": round(sum(lat), 1),
            "parallel_speedup": round(sum(lat) / wall_ms, 2) if wall_ms else None,
            "per_agent": per_agent,
            "agents_total": len(outs), "agents_usable": len(usable),
            "agents_degraded": sum(1 for o in outs if o.status == "degraded"),
            "agents_failed": sum(1 for o in outs if o.status in ("failed", "timeout", "abstained")),
            "citation_coverage": round(ev_cited / ev_total, 4) if ev_total else 1.0,
            "filing_citations_verified": round(sum(1 for c in filing_cits if c.verified) / len(filing_cits), 4) if filing_cits else None,
            "grounded_recommendations": round(sum(1 for r in recs if r["grounded"]) / len(recs), 3),
            "concentration_score_before": before["concentration_score"],
            "concentration_score_after": after["concentration_score"],
            "portfolio_vol_ann": round(before["volatility_ann"], 4), "var95_1d": round(before["var95_1d"], 0),
            "portfolio_es95_hist": round(before["risk"].get("es95_hist", 0), 0),
            "top_risk_share": round(before["risk"].get("top_risk_share", 0), 3),
            "signal_accuracy_30d": {k: (round(self.backtest[k]["hit_rate"], 3) if self.backtest[k]["hit_rate"] is not None else None)
                                    for k in ("momentum", "volume", "sentiment", "ensemble")},
            "backtest_n": ens["n_signals"],
            "degradations": sorted(degr),
        }

    def _log(self, uid, sid, metrics):
        config.LOG_DIR.mkdir(parents=True, exist_ok=True)
        with open(config.LOG_DIR / "metrics.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": time.time(), "user": uid, "session": sid, **metrics}) + "\n")

    # ------------------------------------------------------------------ latency benchmarks
    def _bench_tasks(self, syms, cache):
        return [((s, a), ag, self._ctx(s, frozenset(), cache=cache)) for s in syms for a, ag in SIGNAL_AGENTS.items()]

    def benchmark(self, workloads=(1, 3, 6, 12), repeats: int = 3) -> list[dict]:
        """4 signal agents x N stocks: sequential vs parallel vs parallel + shared request cache."""
        rows = []
        for n in workloads:
            syms = SYMBOLS[:n]
            seq, par, opt, calls = [], [], [], {}
            for _ in range(repeats):
                _, w = run_agents(self._bench_tasks(syms, None), parallel=False)
                seq.append(w)
                c0 = sum(self.feed.calls.values())
                _, w = run_agents(self._bench_tasks(syms, None), parallel=True)
                par.append(w)
                c1 = sum(self.feed.calls.values())
                _, w = run_agents(self._bench_tasks(syms, RunCache()), parallel=True)
                opt.append(w)
                c2 = sum(self.feed.calls.values())
                calls = {"no_cache": c1 - c0, "cache": c2 - c1}
            rows.append({"symbols": n, "agent_tasks": n * len(SIGNAL_AGENTS), "sequential_ms": round(mean(seq), 1),
                         "parallel_ms": round(mean(par), 1), "parallel_cached_ms": round(mean(opt), 1),
                         "speedup": round(mean(seq) / mean(par), 1), "speedup_cached": round(mean(seq) / mean(opt), 1),
                         "feed_calls_no_cache": calls["no_cache"], "feed_calls_cached": calls["cache"]})
        return rows

    def load_benchmark(self, concurrency=(1, 2, 4, 8), per_level: int = 8, uid: str = "meera") -> list[dict]:
        """End-to-end analyses (12 stocks x 6 agents) issued by N simultaneous users."""
        rows = []
        for c in concurrency:
            lat = []
            t0 = time.perf_counter()
            with ThreadPoolExecutor(max_workers=c) as ex:
                for ms in ex.map(lambda _: self.analyze(uid, persist=False)["metrics"]["e2e_ms"], range(per_level)):
                    lat.append(ms)
            wall = time.perf_counter() - t0
            lat.sort()
            rows.append({"concurrent_users": c, "requests": per_level, "p50_ms": round(lat[len(lat) // 2], 1),
                         "p95_ms": round(lat[min(len(lat) - 1, int(len(lat) * 0.95))], 1), "max_ms": round(lat[-1], 1),
                         "throughput_per_s": round(per_level / wall, 2)})
        return rows

    def pool_sweep(self, sizes=(2, 4, 8, 16, 48), uid: str = "meera") -> list[dict]:
        """Effect of worker-pool size on one full 72-task analysis."""
        rows = []
        user, holdings, prices, snap, beh = self._user_bundle(uid, frozenset())
        for n in sizes:
            pool = ThreadPoolExecutor(max_workers=n)
            try:
                ws = []
                for _ in range(2):
                    tasks = [((s, a), ag, self._ctx(s, frozenset(), user, snap, beh, [], RunCache()))
                             for s in SYMBOLS for a, ag in ALL_AGENTS.items()]
                    ws.append(run_agents(tasks, pool=pool)[1])
                rows.append({"workers": n, "wall_ms": round(mean(ws), 1)})
            finally:
                pool.shutdown(wait=False)
        return rows


def dt_today() -> str:
    import datetime
    return datetime.date.today().isoformat()


def register_fallback(tk: str):
    from .market import LIVE_STOCKS, Stock
    LIVE_STOCKS.setdefault(tk, Stock(tk, tk, "Live market", 0.0, 0.3, 1.0, 0, 0.0, 0, 1.0, 0.0, "live"))

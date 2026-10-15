# VibeFinance - Multi-Agent Autonomous Financial Intelligence (PS 04)

Turns market data, regulatory filings and a user's behavioural history into **personalised, cited, confidence-rated**
investment intelligence. Six specialised agents run in parallel; a synthesis layer merges them per investor.
Full design: [ARCHITECTURE.md](ARCHITECTURE.md) (also opens from the dashboard's *Architecture* button).

> All companies, prices, headlines, flows and filings are **synthetic** (fictional NSE-style tickers, seeded generator).
> Back-test numbers validate pipeline mechanics, not real-market performance. Not investment advice.

## Run

```bash
pip install -r requirements.txt
python run.py            # http://127.0.0.1:8000   (add --reset to re-seed the SQLite DB)
python -m pytest -q      # 60 tests: contract, no-look-ahead, RAG + vector DB, personalisation, degraded modes, all 7 extensions
```
Python 3.11+, FastAPI, NumPy, qdrant-client (embedded vector DB), requests. Simulated mode needs no keys and no network and is fully reproducible (seed 42); the *Live data* tab uses the public internet.

## Judge's 5-minute walkthrough

1. **Dashboard** - 12 stocks x 4 independent signal dimensions (momentum, volume anomaly, sentiment & flows, filings via RAG),
   each with a classification chip and confidence bar; portfolio/watchlist state, alerts, concentration score, KPIs.
2. **Stock analysis** - click *VIKRAMINFRA*: price up 77% in 20 days on 3.5x normal volume, but filings show a show-cause notice and
   rising promoter pledge. See the full reasoning chain, the explanation of the agent disagreement, the filing quotes
   (click a source tag to open the document with the cited passage highlighted; "quote verified in source" badge),
   the contribution table and the parallel-execution timeline.
3. **Personalisation** - same stock, three investors, identical market inputs, different composites and actions.
   Drag the risk-tolerance slider and *Save* to watch the advice change (persisted in SQLite).
4. **Degraded data** - tick `filings_unavailable`, `sentiment_feed_down`, `price_feed_down`, `slow_agent`... The pipeline
   still completes; failed agents show as unavailable, confidence drops, ungrounded calls are capped, nothing is uncited.
5. **Evaluation & metrics** - per-session latency / speed-up / concentration / citation coverage, 30-day forward-return
   accuracy per dimension with CIs and a calibration table, and a sequential-vs-parallel latency benchmark.
6. **Persistent history** - sessions, accept/ignore decisions (fed back into the Behaviour agent), paper trades that update holdings.

## Extensions (all seven are implemented; details and measured numbers in ARCHITECTURE.md section 7)

| # | Extension | Where to see it |
|---|---|---|
| 1 | Multiple live sources: Yahoo Finance prices, NSE corporate announcements and earnings notices, Google News | *Live data* tab, `app/live.py` |
| 2 | Semantic search over a larger corpus (520 passages) in an embedded persistent Qdrant vector DB, with filters and retrieval metrics | *Filings search* tab, `vectorstore.py`, `rag.py` |
| 3 | Portfolio risk contributions, stress tests, VaR/ES, sector monitor, per-user watchlist alert rules with new/cleared tracking | *Portfolio risk & alerts* tab, `alerts.py` |
| 4 | Historical back-testing: per-dimension accuracy plus a personalised portfolio strategy vs benchmark | *Evaluation* tab, `backtest.py`, `strategy.py` |
| 5 | Isotonic confidence calibration with bootstrap bands, uncertainty drivers, leave-one-agent-out disagreement explanations | *Stock analysis* and *Evaluation* tabs, `calibration.py` |
| 6 | Persistent sessions: snapshots, diffs, recommendation flips, behavioural drift, decision follow-through | *Sessions & changes* tab, `timeline.py` |
| 7 | Parallel optimisation: single-flight cache, queue-aware deadlines, concurrent-user and pool-size benchmarks | *Evaluation* tab (Run benchmark), `runcache.py` |

## Layout

```
app/market.py      simulated NSE universe + feed layer with failure injection + data-quality gate
app/corpus.py      synthetic SEBI filings / earnings transcripts / sector notes
app/rag.py         hybrid BM25 + hashed-vector retrieval, quote verification
app/signals.py     pure signal functions (shared by live agents and back-test)
app/agents.py      Momentum, Volume, Sentiment, Filings(RAG), Risk, Behaviour
app/engine.py      parallel orchestrator (deadline, isolation, contract validation), pipeline, alerts, metrics, benchmark
app/synthesis.py   weighting, conflicts, calibration, action rules, sizing, reasoning chain
app/backtest.py    30-day forward-return evaluation + confidence calibration
app/store.py       SQLite persistence          app/portfolio.py  valuation, concentration, VaR, behaviour analytics
app/server.py      FastAPI API                 app/static/index.html  dashboard
```

## Rubric map

| Criterion | Where |
|---|---|
| Signal classification (4 dimensions, confidence, evidence) | `signals.py`, dashboard table, agent cards |
| Multi-agent + synthesis | `engine.run_agents` (parallel, timeout, contract), `synthesis.py` |
| Retrieval & attribution | `rag.py`, `FilingsAgent`, clickable verified citations, document viewer |
| Personalisation & risk | `/api/compare`, profile editor, risk gate, behaviour guard |
| Engineering & reliability | degradation toggles, isolation, SQLite, deterministic seed, 60 tests |
| Performance & evaluation | session metrics, back-test + calibration, latency benchmark |
| Explainability & relevance | reasoning chain, disagreement explanation, plain-language actions with sizing |

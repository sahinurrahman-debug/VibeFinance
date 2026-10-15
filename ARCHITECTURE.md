# VibeFinance - Agent Architecture & Decision Logic

Multi-agent financial intelligence for retail investors. Every recommendation is personalised, cited, confidence-rated and
reproducible. **All data (companies, prices, headlines, flows, filings) is synthetic and deterministic (seed 42).** Figures
below describe the shipped simulation, not real markets. Not investment advice.

## 1. Pipeline

```
 feeds (simulated NSE prices/volume, news + FII + options, filing corpus)
   |  MarketFeed: injectable outages, latency, staleness, corrupt bars
   v
 [1 Ingest]  data-quality gate (NaN fill, glitch-spike smoothing, staleness) + user profile / trade log load
   v
 [2 Agents, in parallel on one thread pool, 72 tasks = 12 stocks x 6 agents, hard deadline]
     Momentum   Volume   Sentiment   Filings(RAG)     |  Risk(user)   Behaviour(user)
        \         |         |            /                     \            /
         '--------+---------+-----------'                       '----------'
                          v                                          v
 [3 Synthesis & personalisation]  weights x confidence -> composite -> conflicts -> calibrated confidence
                                  -> action thresholds -> risk gate -> behaviour guard -> sizing -> reasoning chain
   v
 [4 Portfolio risk & alerts]  concentration, vol, VaR, before/after suggested trades, watchlist alerts
   v
 [5 Persist & log]  SQLite (sessions, recommendations, decisions, trades) + logs/metrics.jsonl
```

## 2. Agents and their output contract

All agents return one `AgentOutput` (`app/contracts.py`):
`agent, symbol, kind(signal|constraint), status(ok|degraded|abstained|failed|timeout), label, score[-1,1], confidence[0,1],
summary, evidence[{text, metric, value, source_id, stance}], citations[{source_id, kind, title, date, snippet, url, verified}],
data{}, warnings[], latency_ms`.

`validate()` is enforced by the orchestrator: a usable output must carry evidence, and **every evidence item must cite a
source that appears in its own citation list**. Violations, crashes and deadline misses become explicit "no signal" outputs
(`failed` / `timeout`) - no exception ever reaches the user and no uncited claim reaches synthesis.

| Agent | Role | Method | Sources it cites |
|---|---|---|---|
| **Momentum** (signal) | Price trend | 20/60-day return normalised by volatility, SMA50/200 structure, RSI(14) with overbought/oversold fade, MACD histogram -> weighted score; confidence from component agreement | `PX:<sym>:<date>` price series |
| **Volume** (signal) | Volume anomaly | 3-session volume z-score vs 20-session baseline x price move x OBV tilt -> Accumulation / Distribution / Churn / Dry-up / Normal | `PX:` |
| **Sentiment** (signal) | News, flows, options | Recency x source-reliability weighted headline polarity (finance lexicon), 5-day FII net flow, put-call ratio; confidence from headline count and dispersion | each headline id `N-...`, `FII:`, `OPT:` |
| **Filings** (signal, **RAG**) | Disclosures | Three queries -> hybrid retrieval (BM25 + dense LSA vectors in a Qdrant vector DB, thesaurus query expansion, recency weighting) restricted to the company (+ its sector notes) -> per-passage tone scored with the lexicon, weighted by relevance x recency -> the strongest sentence is quoted verbatim | passage ids `DOC#section`; each quote is **verified as a literal substring of the source** |
| **Risk** (constraint, per user) | Suitability | Annualised vol vs user limit, position headroom, sector headroom -> Fits / Stretch / Breaches | `PF:<user>`, `PX:` |
| **Behaviour** (constraint, per user) | Behavioural traps | Archetype *derived from the trade log vs the market context of each trade*: FOMO buys (>=12% 20-day run-up), panic sells (>=8% below 20-day high, later recovered), turnover; plus the user's own earlier accept/ignore decisions | `TRD-<id>`, `DEC-<id>`, `HIST:<user>` |

Signal agents are user-independent and their outputs are shared; constraint agents are computed per user. All six run
concurrently; an agent that exceeds `VF_AGENT_TIMEOUT` (default 1.2 s) is skipped.

## 3. Synthesis logic (`app/synthesis.py`)

1. **Profile parameters** from risk tolerance *t* (1-10): signal weights interpolate between a conservative mix
   (momentum 15 / volume 10 / sentiment 15 / **filings 60**) and an aggressive mix (38 / 22 / 15 / 25);
   buy threshold `0.45 - 0.035 t`, sell threshold `-(0.10 + 0.03 t)`, minimum confidence `0.62 - 0.03 t`.
2. **Composite** = sum(weight x agent confidence x score) / sum(weight x agent confidence) over usable agents.
3. **Conflict detection**: any pair of agents with opposite scores beyond +-0.25 and confidence >= 0.4. The conflict is
   *explained* (who disagrees, which side the profile weights more) and *penalised* (confidence x0.8) instead of averaged away.
4. **Confidence** = sum(weight x confidence) [missing agents count as 0] x (0.7 + 0.3 x agreement) x conflict penalty;
   capped at 0.60 when the filings agent could not ground the call. The UI also shows the **empirical back-test hit rate of
   past signals in the same confidence bucket** and the score interval (composite +- weighted dispersion).
5. **Action**: BUY / ACCUMULATE / HOLD / WATCH / TRIM / SELL / AVOID using the profile thresholds, then
   - *risk gate*: a positive signal that breaches volatility / position / sector limits becomes WATCH or HOLD with the reason;
   - *no price data*: cannot size -> WATCH;
   - *behaviour guard*: FOMO-prone user on a run-up -> BUY downgraded to ACCUMULATE, size halved, 3 tranches;
     panic-prone user on a dip -> TRIM/SELL softened to HOLD unless signals are strongly bearish;
   - *conservative profiles do not act while agents disagree*.
6. **Sizing**: headroom (position & sector) x signal intensity x confidence x behavioural / grounding multipliers, capped by cash.
7. **Reasoning chain**: ordered steps (ingest -> each agent -> weighting -> agreement -> confidence -> risk gate -> behaviour
   -> decision), each with clickable source tags.

## 4. Degraded-data handling (demonstrable from the UI toggles)

| Injected failure | Behaviour |
|---|---|
| `price_feed_stale` (3 sessions missing) | Analysis on last good bars, confidence x0.6, warning shown |
| `price_corrupt` (NaN + 45% glitch bar) | Gate forward-fills NaNs, smooths the reversed spike, status `degraded`, confidence reduced |
| `price_feed_down` | Momentum/Volume/Risk `failed`; valuation uses last cached close; no BUY can be sized; sentiment + filings still drive a watch-only view |
| `sentiment_feed_down` | Falls back to a 3-session-old cached snapshot, status `degraded`, confidence x0.5 |
| `filings_unavailable` / `filings_missing:SYM` | Filings agent *abstains* (no ungrounded claim); recommendation flagged ungrounded, confidence <= 60%, never BUY |
| `slow_agent` | Sentiment agent hangs; orchestrator deadline skips it (`timeout`); pipeline still completes |
| conflicting agents (natural: INFOSPHERE, VIKRAMINFRA) | Explained and confidence-penalised |
| agent exception / contract violation | Replaced by `failed` output; others unaffected (covered by tests) |

## 5. Evaluation & metrics

Logged per session (SQLite `sessions.metrics` + `logs/metrics.jsonl`): end-to-end latency, agent wall time vs summed
agent latency (parallel speed-up), per-agent latency, portfolio concentration score before/after suggested trades,
annualised portfolio volatility and 1-day 95% VaR, citation coverage (evidence with a valid citation), share of filing
quotes verified against source, share of grounded recommendations, back-test accuracy per dimension.

**Back-test** (`app/backtest.py`): every 5 sessions per stock, each dimension's signal is computed from data available at
that date only (the same functions the live agents call; unit-tested for look-ahead) and compared with the realised
30-calendar-day forward return; |score| >= 0.15 is scored. Reports hit rate with Wilson 95% CI, mean forward return for
bullish vs bearish calls, the coin-flip base rate, and a confidence-calibration table.
Shipped run: momentum 57%, volume 55%, sentiment 56%, ensemble 58% (coin-flip base rate 51.5%, n = 579 ensemble signals).
**Caveat**: the simulator deliberately embeds mild structure (slow drift regimes, post-volume-event drift, news that
partly anticipates returns), so these figures validate the *pipeline*, not real-market edge; overlapping windows make
samples serially correlated; the filings agent is not back-tested (single as-of corpus).

**Latency**: see section 7 (extension 7) for the sequential, parallel, cached, concurrent-user and pool-size measurements.

## 6. Personalisation proof

`/api/compare` runs the four market agents **once** for a stock and synthesises per user. For VIKRAMINFRA (parabolic rally,
adverse filings) the composite is -0.03 for the conservative investor (filings weigh 56%), +0.14 for the moderate one and
+0.35 for the aggressive one; actions are WATCH / WATCH / HOLD(over position limit) with different reasons. Editing a user's
risk tolerance / limits in the UI persists to SQLite and changes the next run.

## 7. Extensions (all seven implemented)

**1. Live multiple-source data** (`app/live.py`, *Live data* tab, `POST /api/live`). `LiveFeed` implements the same interface as the
simulated feed, so the same six agents, output contract, synthesis and degraded-data handling run on real NSE tickers:
Yahoo Finance daily bars (prices, volume, dividend events), NSE corporate-announcements API (disclosures, results and
investor-call notices, indexed into a live RAG index; each citation links to the NSE attachment), Google News RSS (headlines, used for
sentiment). Each source has its own timeout, TTL cache and status chip; one failing source degrades only its dependants
(tested with an HTTP test double: NSE 403 -> filings abstain, news down -> sentiment fails, total outage -> no crash).
The real data immediately exercised the quality gate (in-progress bars with zero volume). Limits: no keyless FII-flow or option-chain
source (sentiment = headlines only); filing tone is scored on announcement subjects, the PDFs are linked, not parsed.

**2. Semantic search over a larger corpus with a vector database** (`app/corpus_history.py`, `vectorstore.py`, `rag.py`, *Filings search* tab).
The corpus grows from 75 to **520 passages / 291 documents** (8 quarters of earnings calls and disclosures per company, two earlier annual
reports, quarterly sector notes; each company has a tone trajectory so the time structure is real). Dense vectors (LSA, 96-d, trained on the
corpus plus a small domain glossary, no downloads) live in **embedded, persistent Qdrant** with native payload filters (company, sector, document
type, date range); a restart reloads the vectors from disk (index build 8 s -> 2 s) and falls back to an in-memory index if the storage folder is locked.
Hybrid = BM25 + vector with min-max fusion and optional recency weighting. Retrieval quality on 14 hand-labelled queries over all 520 passages:
lexical queries recall@3 BM25 100% / vector 86% / hybrid 100%; paraphrase queries (almost no shared words) recall@3 **BM25 71% -> vector and hybrid 86%**.
Small set and an author-written glossary: indicative, not a benchmark. The filings agent now weights recent filings more and reports a
**disclosure trend** (e.g. VIKRAMINFRA "+0.70 -> -0.44, deteriorating").

**3. Portfolio-level risk, concentration monitoring, personalised watchlist alerts** (`portfolio.risk_analysis`, `alerts.py`, *Portfolio risk & alerts* tab).
Risk contributions (share of variance vs share of value: Rohan's VIKRAMINFRA is 34% of value but 63% of risk), correlation heat-map,
historical VaR 95/99, expected shortfall, drawdown, diversification ratio, stress tests (market -10/-20% via realised betas, largest sector -15%,
worst-month replay). Seven per-user alert rules are persisted in SQLite (confidence, volume z, daily move, adverse filings, risk share, concentration,
loss in a 10% market fall); every alert says which rule fired and why it matters to *this* investor, and is tagged NEW or cleared versus the last session.
The watchlist is a first-class panel (add/remove, action, composite, uncertainty).

**4. Historical back-testing** (`backtest.py`, `strategy.py`). Besides per-dimension 30-day accuracy there is a portfolio back-test per profile:
weekly rebalance, positions chosen with the investor's own weights, buy threshold, minimum confidence and position cap, 0.10% cost on turnover,
signals strictly from data up to the rebalance date, versus an equal-weight benchmark (Asha +27%, Sharpe 2.1, max drawdown -3%; Meera +35%;
Rohan +72%, max drawdown -14%; benchmark +9%, Sharpe 0.4). **Treat the absolute returns as a simulator artefact** (trending synthetic data);
the informative part is relative: risk-averse profiles take less exposure and less drawdown. Filings are excluded (no historical corpus).

**5. Calibration, uncertainty and disagreement explanations** (`calibration.py`). Isotonic (PAV) calibration with shrinkage toward the base rate maps raw
confidence to P(direction right over 30 days) with a 90% bootstrap band. Fitted on the first 60% of dates and scored on the last 40%:
ECE 0.093 -> 0.043, Brier 0.2487 -> 0.2410 against climatology 0.2418. So calibrated confidence is honest but carries **little information beyond
the base rate**, and the UI says so. Each recommendation also carries an uncertainty level with named drivers (agent disagreement,
missing or degraded agents, calibration-band width) and a **leave-one-agent-out counterfactual** ("without Filings the composite would be +0.63, a bullish
verdict, so that agent is decisive").

**6. Persistent user sessions** (`timeline.py`, `store.py`, *Sessions & changes* tab). Each analysis stores a snapshot (positions, cash, concentration,
behavioural metrics, every action recommended, alert ids). The timeline diffs consecutive snapshots: positions opened/closed/resized, cash and
concentration movement, recommendation flips (e.g. TECHNOVA ACCUMULATE -> WATCH), behavioural drift, alerts raised/cleared. It joins the user's
accept/ignore decisions to the session whose advice they answered. Decision history feeds back into the Behaviour agent; profile, rules, watchlist and
trades survive restarts.

**7. Parallel-execution optimisation** (`runcache.py`, `engine.run_agents`, `/api/benchmark`, `/api/benchmark/load`). Changes: a per-run
single-flight request cache (concurrent agents share one feed call: 48 -> 36 calls for 12 stocks), a retrieval-result cache plus start-up warm-up,
and **queue-aware deadlines** (a task times out only after it has been *running* too long, so many simultaneous analyses queueing on the pool
never cause spurious timeouts). Measured with 70-95 ms simulated feed latency: 4 signal agents x 12 stocks sequential 3.97 s -> parallel 0.25 s (15.6x)
-> parallel + cache 0.23 s (17x). Full 72-task analyses by N concurrent users: 1 user p50 293 ms, 2 users 464 ms, 4 users 834 ms, 8 users 1.7 s;
throughput saturates near 4.5 analyses/s (CPU-bound on the Python GIL, not I/O). Pool size for one analysis: 2 workers 3.0 s, 4 -> 1.5 s, 8 -> 0.8 s,
16 -> 0.45 s, 48 -> 0.30 s.

## 8. Honest limitations

- Synthetic companies, prices, filings and news; the lexicon scorer is simple (no negation handling beyond phrase lists).
- Dense retrieval uses LSA (trained on the corpus plus a glossary), not a neural sentence embedding; swap `LSAEmbedder.embed` for a
  sentence-transformer behind the same call. Embedded Qdrant local mode is single-process (a second process falls back to in-memory).
- The strategy back-test and calibration run on simulated data whose built-in structure makes them look better than any real market would.
- The behavioural archetype is rule-based on trade context; the confidence "calibration" is bucketed hit rates, not a fitted model.
- Orchestrator deadlines cannot kill a hung Python thread (it is abandoned, not terminated).
- No transaction costs, taxes, liquidity or slippage in sizing.

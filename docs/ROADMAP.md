# Roadmap

## Shipped
| Phase | Scope |
|---|---|
| **MVP** | Four-dimension signals, RAG with verified citations, six parallel agents, personalised synthesis, dashboard, session metrics, degraded-data handling, architecture notes |
| **Extensions** | Live data (Yahoo / NSE / Google News), Qdrant vector search over 520 passages, portfolio risk + per-user alerts, strategy back-test, isotonic calibration + counterfactuals, persistent sessions, parallel/queue-aware orchestration and load benchmarks |
| **Product shell** | Graphite Gold design system, responsive layout, accessible navigation, skeleton/empty/error states, dark and light themes |
| **Full-stack hardening** | Accounts (scrypt), HttpOnly session cookies, per-user data isolation, REST routes with correct verbs/status codes, uniform `{error}` responses, normalised schema with foreign keys, env management (`.env.example`, dev/prod profiles), 125 automated tests |

## Next (in priority order)
1. **Persistent storage in production**: Postgres (or a mounted disk) instead of ephemeral SQLite; move rate limits and sessions to Redis. *Unblocks horizontal scaling.*
2. **Account lifecycle**: email verification, password reset, "sign out everywhere", optional TOTP MFA.
3. **Real corpus**: ingest actual SEBI/NSE filing PDFs and earnings transcripts; replace LSA with a sentence-transformer embedding behind the same `embed()` call; evaluate on a larger labelled retrieval set.
4. **Real signal validation**: replace the simulator with licensed NSE history; walk-forward back-test with transaction costs, taxes and slippage; re-fit calibration monthly.
5. **Broker-read-only import** (holdings via CSV/account-aggregator) so portfolios start from the user's real positions.
6. **Notifications**: email/WhatsApp digests when a personal alert rule fires.
7. **Observability**: structured logs → hosted log search, error tracking, per-agent latency dashboards, uptime alerts.

## Deliberately out of scope
Order execution, regulated advice, derivatives strategies, multi-asset allocation.

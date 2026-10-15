# Product Requirements: VibeFinance

*Source brief: PS 04 - "Multi-Agent Autonomous Financial Intelligence for Retail Investors".*

## 1. Problem
India's retail investors can see prices, headlines, filings, FII flows and options data, but no tool *reasons across them for one person*. Screeners flag price anomalies,
news apps list headlines, trackers record holdings; none weighs the evidence against the investor's own risk limits and habits, shows why, or says how sure it is.
Result: people react to noise, chase rallies, panic-sell dips and ignore adverse disclosures.

## 2. Users
| Persona | Need |
|---|---|
| **Conservative saver** (Asha, 58) | Do not lose capital; wants disclosures weighted heavily; has history of selling into dips |
| **Moderate SIP investor** (Meera, 39) | Balanced growth; wants clear reasons and sizing |
| **Aggressive trader** (Rohan, 27) | Momentum and volume signals; needs a check against FOMO and over-concentration |

## 3. Goals and non-goals
**Goals**: (1) turn raw data + filings + behaviour into recommendations that are *personalised, cited and confidence-rated*; (2) show the whole reasoning chain; (3) degrade gracefully when
data is missing or agents disagree; (4) measure itself (accuracy, latency, risk concentration); (5) keep every user's data private.
**Non-goals**: executing real trades, giving regulated investment advice, real-time tick data, portfolio optimisation across asset classes.

## 4. Functional requirements (all met)
| # | Requirement | Where |
|---|---|---|
| F1 | Classify signals on 3+ independent dimensions with confidence and cited evidence | momentum, volume, sentiment, filings agents |
| F2 | RAG over filings with visible, verifiable attribution | Filings agent, document viewer, "quote verified" |
| F3 | 3+ specialised agents in parallel with a structured output contract and a synthesis layer | six agents, `AgentOutput`, `synthesis.py` |
| F4 | Individually stored profiles; same market inputs → different outputs per user | accounts + comparison view |
| F5 | Dashboard: signals, synthesised analysis with sources, portfolio/watchlist | Dashboard, Stock Analysis |
| F6 | ≥3 measurable session metrics | latency, speed-up, concentration, citation coverage, accuracy |
| F7 | End-to-end demonstration with the full reasoning chain | Stock Analysis |
| F8 | Graceful handling of degraded/conflicting data | six injectable failures, conflict explanations |
| F9 | Architecture documentation | `docs/ARCHITECTURE.md` |
| F10 | Extensions: live sources, vector DB, portfolio risk + alerts, back-testing, calibration, persistent sessions, parallel optimisation | see Architecture §7 |

## 5. Full-stack requirements
| Area | Requirement | Status |
|---|---|---|
| API design | Resource routes, correct verbs and status codes, documented | `docs/API_SPEC.md` |
| Database | Normalised, foreign keys, constraints, no raw passwords | schema v2 |
| Auth | Secure session cookies, protected routes, strict per-user data isolation | scrypt + HttpOnly cookies + ownership by construction |
| Errors | API errors are `{error}` only; UI shows friendly messages | global handlers + toasts |
| Environment | `.env.example`, no secrets in repo, dev vs production separated | `app/config.py`, `.env*.example`, `render.yaml` |

## 6. Success metrics
* Every recommendation has ≥1 citation and a confidence; citation coverage = 100%; filing quotes verified = 100%.
* A full 72-task analysis (12 stocks × 6 agents) in < 0.5 s with simulated feed latency.
* No cross-user data access in the automated test suite (125 tests).
* Calibrated confidence is *reported honestly* (it adds little beyond the base rate on this data).

## 7. Constraints and assumptions
* Data is **synthetic** (fictional companies, simulated prices, template-generated filings) and deterministic (seed 42); live mode uses public, unofficial endpoints that may block or change.
* The simulator embeds mild predictability, so back-test numbers validate the pipeline, not real-market alpha.
* Single-process deployment (embedded vector store, in-process rate limiter, SQLite).

## 8. Risks
| Risk | Mitigation |
|---|---|
| Users mistake demo output for advice | Persistent "Synthetic data" badge, disclosures in UI and docs |
| LLM-style hallucinated claims | Rule-based agents; every claim must cite a source or the output is rejected; quotes verified against source text |
| Account takeover / data leakage | scrypt hashing, hashed session tokens, rate limits, CSRF checks, 404-on-foreign-resource, tests |
| Live sources blocked from cloud IPs | Each source degrades independently; simulated mode unaffected |

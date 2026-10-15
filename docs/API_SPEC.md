# API Specification

Base path `/api`. JSON in, JSON out. Interactive Swagger UI is at `/docs` in development (disabled in production by default).

## Conventions

| Topic | Rule |
|---|---|
| Auth | Session cookie `vf_session` (`HttpOnly`, `SameSite=Lax`, `Secure` in production). Obtain it from `register`, `login` or `guest`. No tokens in URLs or bodies. |
| Identity | **No route takes a user id.** `me` always means the user behind the cookie, so one user cannot address another's data. |
| Success bodies | The resource itself (not wrapped). Lists are bare arrays. |
| Error bodies | **Always** `{"error": "<human-readable message>"}`. Never a stack trace, never `{"detail": ...}`. |
| Writes from browsers | Must be same-origin (or listed in `VF_ALLOWED_ORIGINS`); otherwise `403`. |
| Caching | `Cache-Control: no-store` on every `/api` response. Every response carries `X-Request-ID` for support/log correlation. |

### Status codes
`200` OK · `201` Created (+ `Location`) · `204` No Content · `400` malformed JSON · `401` not signed in / bad credentials ·
`403` cross-site write blocked · `404` not found (**also** returned for resources that belong to someone else) ·
`405` wrong method · `409` business-rule conflict (insufficient cash, overselling, duplicate email) ·
`422` validation failed (message names the field) · `429` rate limited (+ `Retry-After`) · `500` unexpected (generic message) · `503` demo at capacity.

## Routes

**Auth** = who may call it: *Public*, or *Required* (any signed-in user; always scoped to that user).

### Health and reference

| Method | Route | Auth | Success | Description |
|---|---|---|---|---|
| GET | `/api/health` | Public | 200 | Liveness + DB check (used by the host's health check) |
| GET | `/api/meta` | Public | 200 | Non-personal reference data: stock universe, scenarios, demo personas, disclosure |
| GET | `/api/architecture` | Public | 200 text | Architecture notes (markdown) |

### Authentication

| Method | Route | Auth | Success | Errors | Description |
|---|---|---|---|---|---|
| POST | `/api/auth/register` | Public | 201 `{user}` + cookie | 409 email taken, 422 invalid, 429 | Create an account `{email, password, name}`. Password: 10+ chars with a letter and a number |
| POST | `/api/auth/login` | Public | 200 `{user}` + cookie | 401 (same message for unknown email and wrong password), 429 after 5 failures | Sign in `{email, password}` |
| POST | `/api/auth/guest` | Public | 201 `{user}` + cookie | 429, 503 | Private expiring sandbox cloned from `{persona: asha\|meera\|rohan}` |
| POST | `/api/auth/logout` | Public (idempotent) | 204 | | Destroys the server-side session and clears the cookie |

### Me (the signed-in user)

| Method | Route | Auth | Success | Description |
|---|---|---|---|---|
| GET | `/api/me` | Required | 200 `{user, holdings, watchlist}` | Own profile and portfolio |
| PATCH | `/api/me` | Required | 200 `{user}` | Update profile/risk limits (`name, age, bio, risk_tolerance, max_position_pct, max_sector_pct, max_volatility, horizon_years`). Unknown fields → 422 |
| DELETE | `/api/me` | Required | 204 | Delete the account and all its data (cascade) |
| GET | `/api/me/watchlist` | Required | 200 `[symbol]` | |
| PUT | `/api/me/watchlist/{symbol}` | Required | 204 | Add (idempotent). Unknown symbol → 404 |
| DELETE | `/api/me/watchlist/{symbol}` | Required | 204 | Remove. Not on list → 404 |
| POST | `/api/me/trades` | Required | 201 `{trade_id, holdings, cash}` | Paper trade at the last close `{symbol, side: BUY\|SELL, qty}`. Insufficient cash / overselling → 409 (nothing is changed) |
| GET | `/api/me/trades?limit=` | Required | 200 | Trade log (1-500) |
| GET | `/api/me/history` | Required | 200 | Recent analysis runs, decisions and trades |
| GET | `/api/me/timeline` | Required | 200 | Session-to-session changes (positions, recommendation flips, behaviour drift) |
| GET | `/api/me/alert-rules` | Required | 200 `{rule: value}` | |
| PATCH | `/api/me/alert-rules` | Required | 200 | Update personal alert thresholds; unknown rule → 422 |
| GET | `/api/me/decisions?limit=` | Required | 200 | Accept/ignore history |
| POST | `/api/me/decisions` | Required | 201 `{id}` + Location | Record `{rec_id?, symbol, action, choice: accepted\|ignored\|modified, note?}`. `rec_id` must be yours, else 404 |
| DELETE | `/api/me/decisions/{id}` | Required | 204 | Own decision only, else 404 |

### Analyses (the multi-agent pipeline)

| Method | Route | Auth | Success | Description |
|---|---|---|---|---|
| POST | `/api/analyses` | Required | 201 + `Location: /api/analyses/{id}` | Run all six agents for you. Body `{degradations: []}` (e.g. `price_feed_down`, `slow_agent`). Rate limited: 30/min/user |
| GET | `/api/analyses?limit=` | Required | 200 | Your past runs |
| GET | `/api/analyses/{id}` | Required | 200 | A stored run with its recommendations. Not yours / not found → 404 |
| DELETE | `/api/analyses/{id}` | Required | 204 | Delete a stored run (cascades to its recommendations) |
| GET | `/api/comparisons/{symbol}?degradations=a,b` | Required | 200 | Identical market inputs, different advice: **you + the public demo personas** (never other customers). Each result has `is_you` |
| GET | `/api/strategy` | Required | 200 | Personalised strategy back-test for you + demo personas vs benchmark |
| GET | `/api/live?tickers=RELIANCE,TCS&degradations=` | Required | 200 | Same agents on live NSE data (Yahoo, NSE announcements, Google News). 6/min/user |

### Reference data (sign-in required: it is the compute we pay for)

| Method | Route | Description |
|---|---|---|
| GET | `/api/prices/{symbol}?n=` | Price/volume series (20-500 sessions) |
| GET | `/api/docs/{doc_id}` | A synthetic filing document |
| GET | `/api/backtest` | Per-dimension 30-day accuracy and calibration model |
| GET | `/api/rag/stats` | Vector index stats and retrieval quality |
| GET | `/api/rag/search?q=&symbol=&doc_type=&date_from=&mode=hybrid\|bm25\|vector&k=&recency=` | Semantic search over filings |
| GET | `/api/benchmark?repeats=` · `/api/benchmark/load` | Latency benchmarks (6/min/user) |

## Examples

```http
POST /api/auth/register
{"email":"priya@example.com","password":"tiger-lily-2026","name":"Priya"}
→ 201 Set-Cookie: vf_session=…; HttpOnly; SameSite=Lax; Path=/; Max-Age=604800
→ {"user":{"id":"…","role":"member","email":"priya@example.com","name":"Priya","risk_tolerance":5,…}}

POST /api/me/trades   {"symbol":"BHARATOIL","side":"BUY","qty":999999}
→ 409 {"error":"Not enough cash for this purchase."}

GET /api/analyses/42      (belongs to another user)
→ 404 {"error":"That analysis was not found."}

POST /api/auth/login   (6th wrong attempt)
→ 429 Retry-After: 887   {"error":"Too many failed attempts. Please try again in 15 minutes."}
```

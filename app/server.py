"""REST API + single-page dashboard.

Conventions
  * JSON in, JSON out. Success bodies are the resource itself; EVERY error is `{"error": "<message>"}`
    (never a stack trace, never FastAPI's default `{"detail": ...}`).
  * Status codes: 200 read/update, 201 created (+Location), 204 no content, 400 malformed, 401 not signed in,
    403 cross-site request blocked, 404 not found (also used for "belongs to someone else"), 409 conflict,
    422 validation failed, 429 rate limited, 503 capacity.
  * Auth: HttpOnly + SameSite=Lax session cookie (Secure in production). Every route that touches user data
    depends on `current_user` and only ever uses that user's id - there is no user id in any URL or body.
Full reference: docs/API_SPEC.md
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import secrets
import time
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import urlparse

import numpy as np
from fastapi import Depends, FastAPI, Query, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from starlette.exceptions import HTTPException as StarletteHTTPException

from . import config
from .alerts import DEFAULT_RULES, RULE_HELP
from .backtest import public as bt_public
from .bootstrap import build_engine
from .live import DEFAULT_TICKERS, TICKER_RE
from .market import BY_SYMBOL, UNIVERSE
from .security import (LIMITER, email_problem, hash_password, hash_token, new_id, new_token, normalise_email,
                       password_problem, verify_password)

config.validate()
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL, logging.INFO), format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("vibefinance")

STATIC = Path(__file__).parent / "static"
DOCS = Path(__file__).parent.parent / "docs"
app = FastAPI(title="VibeFinance API", version="2.0.0",
              docs_url="/docs" if config.ENABLE_DOCS else None, redoc_url=None,
              openapi_url="/openapi.json" if config.ENABLE_DOCS else None)
ENGINE = build_engine()
STORE = ENGINE.store
STORE.purge_expired(config.GUEST_HOURS * 3600)
app.mount("/static", StaticFiles(directory=STATIC), name="static")

UNSAFE = {"POST", "PUT", "PATCH", "DELETE"}
PERSONAS = ("asha", "meera", "rohan")
GENERIC_500 = "Something went wrong on our side. Please try again in a moment."


# ============================================================================ errors
class ApiError(Exception):
    def __init__(self, status: int, message: str, headers: dict | None = None):
        self.status, self.message, self.headers = status, message, headers or {}


def _json(body, status: int = 200, headers: dict | None = None) -> JSONResponse:
    def default(o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if hasattr(o, "isoformat"):
            return o.isoformat()
        raise TypeError(type(o))
    return JSONResponse(json.loads(json.dumps(body, default=default)), status_code=status, headers=headers)


def _error(status: int, message: str, headers: dict | None = None) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status, headers=headers)


_HTTP_MESSAGES = {404: "That resource was not found.", 405: "That method is not allowed on this route.",
                  401: "Please sign in to continue.", 403: "You do not have access to that.", 429: "Too many requests."}


@app.exception_handler(ApiError)
async def _api_error(request: Request, exc: ApiError):
    return _error(exc.status, exc.message, exc.headers)


@app.exception_handler(StarletteHTTPException)
async def _http_error(request: Request, exc: StarletteHTTPException):
    msg = _HTTP_MESSAGES.get(exc.status_code) or (exc.detail if isinstance(exc.detail, str) else "Request failed.")
    return _error(exc.status_code, msg, getattr(exc, "headers", None))


@app.exception_handler(RequestValidationError)
async def _validation_error(request: Request, exc: RequestValidationError):
    e = exc.errors()[0] if exc.errors() else {}
    if e.get("type") == "json_invalid":
        return _error(400, "The request body is not valid JSON.")
    field = ".".join(str(x) for x in e.get("loc", ()) if x not in ("body", "query", "path"))
    msg = str(e.get("msg", "Invalid request.")).removeprefix("Value error, ")
    return _error(422, f"{field}: {msg}" if field else msg)


@app.exception_handler(Exception)
async def _unhandled(request: Request, exc: Exception):
    log.exception("unhandled error rid=%s %s %s", getattr(request.state, "rid", "-"), request.method, request.url.path)
    return _error(500, GENERIC_500)


# ============================================================================ middleware
def client_ip(request: Request) -> str:
    if config.TRUST_PROXY:
        xff = request.headers.get("x-forwarded-for")
        if xff:
            return xff.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _origin_ok(request: Request, origin: str) -> bool:
    o = origin.rstrip("/")
    if o in config.ALLOWED_ORIGINS:
        return True
    host = (request.headers.get("x-forwarded-host") if config.TRUST_PROXY else None) or request.headers.get("host", "")
    return urlparse(o).netloc.lower() == host.lower()


CSP = ("default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
       "font-src https://fonts.gstatic.com; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; "
       "base-uri 'self'; form-action 'self'")


@app.middleware("http")
async def guard(request: Request, call_next):
    request.state.rid = secrets.token_hex(6)
    t0 = time.perf_counter()
    path = request.url.path
    # CSRF defence in depth (cookie is already SameSite=Lax): browsers always send Origin on cross-site writes
    if request.method in UNSAFE and path.startswith("/api/"):
        origin = request.headers.get("origin")
        if origin and not _origin_ok(request, origin):
            return _error(403, "Cross-site request blocked.")
    try:
        response = await call_next(request)
    except Exception:                                                     # noqa: BLE001 - last line of defence
        log.exception("unhandled error rid=%s %s %s", request.state.rid, request.method, path)
        response = _error(500, GENERIC_500)
    h = response.headers
    h["X-Request-ID"] = request.state.rid
    h["X-Content-Type-Options"] = "nosniff"
    h["X-Frame-Options"] = "DENY"
    h["Referrer-Policy"] = "strict-origin-when-cross-origin"
    h["Permissions-Policy"] = "geolocation=(), camera=(), microphone=()"
    if not path.startswith(("/docs", "/openapi")):
        h["Content-Security-Policy"] = CSP
    if path.startswith("/api/"):
        h["Cache-Control"] = "no-store"
    if config.HSTS:
        h["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    log.info("rid=%s %s %s -> %s %.0fms", request.state.rid, request.method, path, response.status_code,
             (time.perf_counter() - t0) * 1000)
    return response


# ============================================================================ auth plumbing
def current_user(request: Request) -> dict:
    token = request.cookies.get(config.COOKIE_NAME)
    user = STORE.user_for_session(hash_token(token)) if token else None
    if not user:
        raise ApiError(401, "Please sign in to continue.")
    request.state.user = user
    return user


User = Annotated[dict, Depends(current_user)]


def start_session(response: Response, request: Request, uid: str, ttl_s: float) -> None:
    old = request.cookies.get(config.COOKIE_NAME)
    if old:
        STORE.delete_auth_session(hash_token(old))                       # never reuse / fixate an existing session
    token = new_token()
    STORE.create_auth_session(uid, hash_token(token), ttl_s, request.headers.get("user-agent", ""))
    response.set_cookie(config.COOKIE_NAME, token, max_age=int(ttl_s), httponly=True, secure=config.COOKIE_SECURE,
                        samesite=config.COOKIE_SAMESITE, path="/")


def limit(kind: str, key: str) -> None:
    count, window = config.LIMITS[kind]
    ok, retry = LIMITER.consume(f"{kind}:{key}", count, window)
    if not ok:
        raise ApiError(429, f"Too many requests. Please try again in {retry} seconds.", {"Retry-After": str(retry)})


def symbol_or_404(symbol: str) -> str:
    s = symbol.upper()
    if s not in BY_SYMBOL:
        raise ApiError(404, f"Unknown stock symbol '{symbol[:20]}'.")
    return s


def degradations(csv_or_list) -> list[str]:
    items = csv_or_list if isinstance(csv_or_list, list) else [x for x in (csv_or_list or "").split(",") if x]
    bad = [x for x in items if x not in config.DEGRADATIONS and not x.startswith("filings_missing:")]
    if bad or len(items) > 8:
        raise ApiError(422, f"degradations: unknown scenario {bad[0]!r}." if bad else "degradations: too many scenarios.")
    return items


# ============================================================================ request models
class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class RegisterIn(_Strict):
    email: str = Field(max_length=254)
    password: str = Field(max_length=128)
    name: str = Field(min_length=1, max_length=60)

    @field_validator("email")
    @classmethod
    def _email(cls, v):
        p = email_problem(v)
        if p:
            raise ValueError(p)
        return normalise_email(v)

    @field_validator("password")
    @classmethod
    def _pw(cls, v):
        p = password_problem(v)
        if p:
            raise ValueError(p)
        return v


class LoginIn(_Strict):
    email: str = Field(max_length=254)
    password: str = Field(max_length=128)


class GuestIn(_Strict):
    persona: Literal["asha", "meera", "rohan"]


class ProfilePatch(_Strict):
    name: str | None = Field(None, min_length=1, max_length=60)
    age: int | None = Field(None, ge=16, le=110)
    bio: str | None = Field(None, max_length=280)
    risk_tolerance: int | None = Field(None, ge=1, le=10)
    max_position_pct: float | None = Field(None, ge=1, le=100)
    max_sector_pct: float | None = Field(None, ge=1, le=100)
    max_volatility: float | None = Field(None, ge=0.05, le=2.0)
    horizon_years: int | None = Field(None, ge=1, le=50)

    @model_validator(mode="after")
    def _non_empty(self):
        if not self.model_fields_set:
            raise ValueError("Provide at least one field to update.")
        return self


class TradeIn(_Strict):
    symbol: str
    side: Literal["BUY", "SELL"]
    qty: int = Field(gt=0, le=1_000_000)

    @field_validator("side", mode="before")
    @classmethod
    def _side(cls, v):
        return v.upper() if isinstance(v, str) else v


class DecisionIn(_Strict):
    rec_id: int | None = None
    symbol: str
    action: Literal["BUY", "ACCUMULATE", "HOLD", "WATCH", "TRIM", "SELL", "AVOID"]
    choice: Literal["accepted", "ignored", "modified"]
    note: str = Field("", max_length=280)


class AnalysisIn(_Strict):
    degradations: list[str] = Field(default_factory=list, max_length=8)


# ============================================================================ public routes
@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return FileResponse(STATIC / "logo.svg", media_type="image/svg+xml")


@app.get("/api/health")
def health():
    STORE._q("SELECT 1")
    return _json({"status": "ok", "env": config.ENV})


@app.get("/api/meta")
def meta():
    """Public, non-personal reference data (no user records)."""
    return _json({"as_of": ENGINE.data.dates[-1], "universe": [s.__dict__ for s in UNIVERSE],
                  "degradations": config.DEGRADATIONS, "backtest_meta": ENGINE.backtest["_meta"], "rag": ENGINE.rag.stats(),
                  "live_defaults": DEFAULT_TICKERS, "alert_rule_help": RULE_HELP, "agent_timeout_s": config.AGENT_TIMEOUT_S,
                  "personas": [{k: t[k] for k in ("id", "name", "age", "bio", "risk_tolerance", "max_position_pct", "horizon_years")}
                               for t in STORE.list_templates()],
                  "password_min_length": config.PASSWORD_MIN_LEN, "docs_url": "/docs" if config.ENABLE_DOCS else None,
                  "disclosure": "All companies, prices, headlines, flows and filings are SYNTHETIC. Backtest figures validate pipeline "
                                "mechanics on simulated data, not real-market performance. Not investment advice."})


@app.get("/api/architecture", response_class=PlainTextResponse)
def architecture():
    p = DOCS / "ARCHITECTURE.md"
    if not p.exists():
        raise ApiError(404, "Architecture notes are not available.")
    return PlainTextResponse(p.read_text(encoding="utf-8"))


# ============================================================================ auth
@app.post("/api/auth/register", status_code=201)
def register(body: RegisterIn, request: Request):
    limit("register", client_ip(request))
    if STORE.email_taken(body.email):
        raise ApiError(409, "An account with this email already exists. Try signing in instead.")
    uid = new_id()
    user = STORE.create_user({"id": uid, "role": "member", "email": body.email, "password_hash": hash_password(body.password),
                              "name": body.name})
    STORE.set_watchlist(uid, ["BHARATOIL", "NAVBANK", "TECHNOVA"])
    resp = _json({"user": user}, 201)
    start_session(resp, request, uid, config.SESSION_DAYS * 86400)
    return resp


@app.post("/api/auth/login")
def login(body: LoginIn, request: Request):
    email = normalise_email(body.email)
    key = f"login:{client_ip(request)}:{email}"
    count, window = config.LIMITS["login"]
    allowed, retry = LIMITER.check(key, count, window)
    if not allowed:
        raise ApiError(429, f"Too many failed attempts. Please try again in {retry // 60 + 1} minutes.", {"Retry-After": str(retry)})
    cred = STORE.get_credentials(email)
    if not verify_password(body.password, cred["password_hash"] if cred else None):
        LIMITER.hit(key, window)
        raise ApiError(401, "Incorrect email or password.")          # same message for unknown email and wrong password
    LIMITER.reset(key)
    resp = _json({"user": STORE.get_user(cred["id"])})
    start_session(resp, request, cred["id"], config.SESSION_DAYS * 86400)
    return resp


@app.post("/api/auth/guest", status_code=201)
def guest(body: GuestIn, request: Request):
    """Private, expiring sandbox cloned from a demo persona. Other visitors can never see it."""
    limit("guest", client_ip(request))
    STORE.purge_expired(config.GUEST_HOURS * 3600)
    if STORE.count_guests() >= config.MAX_GUESTS:
        raise ApiError(503, "The demo is at capacity right now. Please try again shortly or create an account.")
    user = STORE.clone_template(body.persona, new_id())
    resp = _json({"user": user}, 201)
    start_session(resp, request, user["id"], config.GUEST_HOURS * 3600)
    return resp


@app.post("/api/auth/logout", status_code=204)
def logout(request: Request):
    token = request.cookies.get(config.COOKIE_NAME)
    if token:
        STORE.delete_auth_session(hash_token(token))
    resp = Response(status_code=204)
    resp.delete_cookie(config.COOKIE_NAME, path="/", secure=config.COOKIE_SECURE, httponly=True, samesite=config.COOKIE_SAMESITE)
    return resp


# ============================================================================ the signed-in user ("me")
@app.get("/api/me")
def me(user: User):
    return _json({"user": user, "holdings": STORE.holdings(user["id"]), "watchlist": STORE.watchlist(user["id"])})


@app.patch("/api/me")
def update_me(body: ProfilePatch, user: User):
    fields = body.model_dump(exclude_unset=True)
    return _json({"user": STORE.update_user(user["id"], fields)})


@app.delete("/api/me", status_code=204)
def delete_me(user: User):
    STORE.delete_user(user["id"])
    resp = Response(status_code=204)
    resp.delete_cookie(config.COOKIE_NAME, path="/", secure=config.COOKIE_SECURE, httponly=True, samesite=config.COOKIE_SAMESITE)
    return resp


@app.get("/api/me/watchlist")
def get_watchlist(user: User):
    return _json(STORE.watchlist(user["id"]))


@app.put("/api/me/watchlist/{symbol}", status_code=204)
def watch_add(symbol: str, user: User):
    STORE.watch_add(user["id"], symbol_or_404(symbol))
    return Response(status_code=204)


@app.delete("/api/me/watchlist/{symbol}", status_code=204)
def watch_remove(symbol: str, user: User):
    if not STORE.watch_remove(user["id"], symbol_or_404(symbol)):
        raise ApiError(404, "That stock is not on your watchlist.")
    return Response(status_code=204)


@app.post("/api/me/trades", status_code=201)
def add_trade(body: TradeIn, user: User):
    symbol = symbol_or_404(body.symbol)
    try:
        result = STORE.apply_trade(user["id"], ENGINE.data.dates[-1].isoformat(), symbol, body.side, body.qty,
                                   ENGINE.feed.last_close(symbol), "manual")
    except ValueError as e:
        raise ApiError(409, str(e))
    return _json(result, 201, {"Location": "/api/me/trades"})


@app.get("/api/me/trades")
def list_trades(user: User, limit_: int = Query(50, alias="limit", ge=1, le=500)):
    return _json(STORE.trades(user["id"], limit_))


@app.get("/api/me/history")
def history(user: User):
    return _json({"sessions": STORE.sessions(user["id"]), "decisions": STORE.decisions(user["id"]),
                  "trades": STORE.trades(user["id"], 25)})


@app.get("/api/me/timeline")
def timeline(user: User):
    return _json(ENGINE.timeline(user["id"]))


@app.get("/api/me/alert-rules")
def get_rules(user: User):
    return _json(STORE.alert_rules(user["id"], DEFAULT_RULES))


@app.patch("/api/me/alert-rules")
def patch_rules(rules: dict[str, float], user: User):
    bad = [k for k in rules if k not in DEFAULT_RULES]
    if bad:
        raise ApiError(422, f"rules: unknown rule {bad[0]!r}.")
    if not rules:
        raise ApiError(422, "rules: provide at least one rule to update.")
    if any(v < 0 or v > 1_000_000 for v in rules.values()) or ("min_confidence" in rules and rules["min_confidence"] > 1):
        raise ApiError(422, "rules: values are out of range.")
    STORE.set_alert_rules(user["id"], rules, DEFAULT_RULES)
    return _json(STORE.alert_rules(user["id"], DEFAULT_RULES))


@app.get("/api/me/decisions")
def list_decisions(user: User, limit_: int = Query(50, alias="limit", ge=1, le=200)):
    return _json(STORE.decisions(user["id"], limit_))


@app.post("/api/me/decisions", status_code=201)
def add_decision(body: DecisionIn, user: User):
    symbol = symbol_or_404(body.symbol)
    try:
        did = STORE.save_decision(user["id"], body.rec_id, symbol, body.action, body.choice, body.note)
    except LookupError:
        raise ApiError(404, "That recommendation was not found.")
    return _json({"id": did}, 201, {"Location": f"/api/me/decisions/{did}"})


@app.delete("/api/me/decisions/{decision_id}", status_code=204)
def delete_decision(decision_id: int, user: User):
    if not STORE.delete_decision(user["id"], decision_id):
        raise ApiError(404, "That decision was not found.")
    return Response(status_code=204)


# ============================================================================ analyses (the multi-agent pipeline)
@app.post("/api/analyses", status_code=201)
def create_analysis(body: AnalysisIn, user: User):
    limit("analysis", user["id"])
    result = ENGINE.analyze(user["id"], degradations(body.degradations))
    return _json(result, 201, {"Location": f"/api/analyses/{result['session_id']}"})


@app.get("/api/analyses")
def list_analyses(user: User, limit_: int = Query(20, alias="limit", ge=1, le=100)):
    return _json(STORE.sessions(user["id"], limit_))


@app.get("/api/analyses/{run_id}")
def get_analysis(run_id: int, user: User):
    run = STORE.get_run(user["id"], run_id)
    if not run:
        raise ApiError(404, "That analysis was not found.")             # same answer whether it exists or is someone else's
    return _json(run)


@app.delete("/api/analyses/{run_id}", status_code=204)
def delete_analysis(run_id: int, user: User):
    if not STORE.delete_run(user["id"], run_id):
        raise ApiError(404, "That analysis was not found.")
    return Response(status_code=204)


@app.get("/api/comparisons/{symbol}")
def compare(symbol: str, user: User, degradations_: str = Query("", alias="degradations")):
    """You plus the public demo personas, with identical market inputs. Never other customers."""
    sym = symbol_or_404(symbol)
    who = [user] + [t for t in STORE.list_templates() if t["id"] != user.get("derived_from")]
    res = ENGINE.compare(sym, degradations(degradations_), who)
    for r in res["results"]:
        r["is_you"] = r["user"]["id"] == user["id"]
    return _json(res)


@app.get("/api/strategy")
def strategy_bt(user: User):
    who = [user] + [t for t in STORE.list_templates() if t["id"] != user.get("derived_from")]
    return _json({**ENGINE.strategy(who), "you": user["id"]})


@app.get("/api/live")
def live(user: User, tickers: str = Query(",".join(DEFAULT_TICKERS), max_length=200), degradations_: str = Query("", alias="degradations")):
    limit("heavy", user["id"])
    syms = [t.strip().upper() for t in tickers.split(",") if t.strip()][:8]
    if not syms or any(not TICKER_RE.match(t) for t in syms):
        raise ApiError(422, "tickers: use NSE symbols such as RELIANCE, TCS or M&M, separated by commas.")
    return _json(ENGINE.live_analyze(user["id"], syms, degradations(degradations_)))


# ============================================================================ reference data (sign-in required: it is the compute we pay for)
@app.get("/api/prices/{symbol}")
def prices(symbol: str, user: User, n: int = Query(120, ge=20, le=500)):
    sym = symbol_or_404(symbol)
    d = ENGINE.data
    return _json({"symbol": sym, "dates": d.dates[-n:], "close": d.close[sym][-n:].round(2), "volume": d.volume[sym][-n:]})


@app.get("/api/docs/{doc_id}")
def get_doc(doc_id: str, user: User):
    d = ENGINE.rag.doc(doc_id)
    if not d:
        raise ApiError(404, "That document was not found.")
    return _json(d.meta())


@app.get("/api/backtest")
def backtest(user: User):
    return _json({**bt_public(ENGINE.backtest), "calibration_model": ENGINE.calibrator.summary()})


@app.get("/api/rag/stats")
def rag_stats(user: User):
    return _json({**ENGINE.rag.stats(), "eval": ENGINE.rag_eval})


@app.get("/api/rag/search")
def rag_search(user: User, q: str = Query(..., min_length=2, max_length=200), symbol: str | None = None, doc_type: str | None = Query(None, max_length=40),
               date_from: str | None = None, mode: Literal["hybrid", "bm25", "vector"] = "hybrid",
               k: int = Query(8, ge=1, le=25), recency: bool = False):
    sym = symbol_or_404(symbol) if symbol else None
    try:
        df = dt.date.fromisoformat(date_from) if date_from else None
    except ValueError:
        raise ApiError(422, "date_from: use the format YYYY-MM-DD.")
    t0 = time.perf_counter()
    hits = ENGINE.rag.search(q, symbol=sym, sector=BY_SYMBOL[sym].sector if sym else None, k=k,
                             doc_types=[doc_type] if doc_type else None, date_from=df, mode=mode, recency=recency)
    return _json({"query": q, "mode": mode, "backend": ENGINE.rag.backend.name, "ms": round((time.perf_counter() - t0) * 1000, 1),
                  "hits": [{"chunk_id": h.chunk.chunk_id, "doc_id": h.chunk.doc_id, "symbol": h.chunk.symbol, "title": h.chunk.title,
                            "doc_type": h.chunk.doc_type, "date": h.chunk.date, "period": h.chunk.period, "heading": h.chunk.heading,
                            "text": h.chunk.text, "score": round(h.score, 3), "bm25": round(h.bm25, 3), "vector": round(h.cosine, 3)}
                           for h in hits]})


@app.get("/api/benchmark")
def benchmark(user: User, repeats: int = Query(2, ge=1, le=5)):
    limit("heavy", user["id"])
    return _json({"io_scale": ENGINE.feed.io_scale, "rows": ENGINE.benchmark(repeats=repeats)})


@app.get("/api/benchmark/load")
def benchmark_load(user: User):
    limit("heavy", user["id"])
    return _json({"io_scale": ENGINE.feed.io_scale, "rows": ENGINE.load_benchmark(), "pool": ENGINE.pool_sweep()})

"""API design, authentication, authorisation (data isolation), error handling, schema and env hygiene."""
import os
import re
import sqlite3
import uuid
from pathlib import Path

import pytest

from app import config, security
from tests.conftest import signup

ROOT = Path(__file__).resolve().parent.parent


def uniq(prefix="u"):
    return f"{prefix}-{uuid.uuid4().hex[:8]}@example.com"


def only_error_key(r):
    j = r.json()
    assert set(j) == {"error"} and isinstance(j["error"], str) and j["error"], j
    assert "Traceback" not in r.text and "File \"" not in r.text
    return j["error"]


# =============================================================================== authentication
def test_register_login_logout_cycle(client):
    email = uniq()
    r = client.post("/api/auth/register", json={"email": email.upper(), "password": "correct-horse-9", "name": "Dana"})
    assert r.status_code == 201
    user = r.json()["user"]
    assert user["email"] == email and user["role"] == "member" and "password" not in r.text and "password_hash" not in r.text
    assert client.get("/api/me").status_code == 200
    assert client.post("/api/auth/logout").status_code == 204
    assert client.get("/api/me").status_code == 401
    ok = client.post("/api/auth/login", json={"email": email, "password": "correct-horse-9"})
    assert ok.status_code == 200 and client.get("/api/me").json()["user"]["id"] == user["id"]


def test_login_failures_are_indistinguishable_and_generic(client):
    email = uniq()
    signup(client, email)
    client.post("/api/auth/logout")
    wrong = client.post("/api/auth/login", json={"email": email, "password": "wrong-password-1"})
    unknown = client.post("/api/auth/login", json={"email": uniq("nobody"), "password": "wrong-password-1"})
    assert wrong.status_code == unknown.status_code == 401
    assert only_error_key(wrong) == only_error_key(unknown)                  # no account enumeration


def test_duplicate_email_conflict_is_case_insensitive(client, api_app):
    from fastapi.testclient import TestClient
    email = uniq()
    signup(client, email)
    with TestClient(api_app, raise_server_exceptions=False) as other:
        r = other.post("/api/auth/register", json={"email": email.upper(), "password": "correct-horse-9", "name": "X"})
    assert r.status_code == 409 and only_error_key(r)


@pytest.mark.parametrize("body,fragment", [
    ({"email": "not-an-email", "password": "correct-horse-9", "name": "A"}, "email"),
    ({"email": "a@b.co", "password": "short1", "name": "A"}, "password"),
    ({"email": "a@b.co", "password": "alllettersnonumbers", "name": "A"}, "password"),
    ({"email": "a@b.co", "password": "correct-horse-9", "name": ""}, "name"),
    ({"email": "a@b.co", "password": "correct-horse-9", "name": "A", "role": "admin"}, "role"),   # mass-assignment attempt
])
def test_registration_validation_returns_422_with_error_string(client, body, fragment):
    r = client.post("/api/auth/register", json=body)
    assert r.status_code == 422 and fragment in only_error_key(r)


def test_malformed_json_is_400(client):
    r = client.post("/api/auth/login", content="{nope", headers={"content-type": "application/json"})
    assert r.status_code == 400 and only_error_key(r)


def test_login_is_rate_limited_per_ip_and_account(client):
    email = uniq()
    signup(client, email)
    client.post("/api/auth/logout")
    codes = [client.post("/api/auth/login", json={"email": email, "password": "wrong-password-1"}).status_code for _ in range(6)]
    assert codes[:5] == [401] * 5 and codes[5] == 429
    r = client.post("/api/auth/login", json={"email": email, "password": "correct-horse-9"})
    assert r.status_code == 429 and "Retry-After" in r.headers and only_error_key(r)   # even the right password is blocked


# =============================================================================== secrets at rest
def test_passwords_and_session_tokens_are_never_stored_raw(client):
    from app.server import STORE
    email, pw = uniq(), "very-secret-pw-77"
    client.post("/api/auth/register", json={"email": email, "password": pw, "name": "Eve"})
    token = client.cookies.get(config.COOKIE_NAME)
    row = STORE._q("SELECT password_hash FROM users WHERE email=?", (email,))[0]["password_hash"]
    assert row.startswith("scrypt$") and pw not in row
    stored = [r["token_hash"] for r in STORE._q("SELECT token_hash FROM auth_sessions")]
    assert token and token not in stored and security.hash_token(token) in stored
    dump = "\n".join(str(r) for r in STORE._q("SELECT * FROM users WHERE email=?", (email,)))
    assert pw not in dump


def test_password_hashing_primitives():
    h = security.hash_password("tiger-lily-2026")
    assert security.verify_password("tiger-lily-2026", h) and not security.verify_password("tiger-lily-2027", h)
    assert h != security.hash_password("tiger-lily-2026")                    # per-user random salt
    assert security.verify_password("x", None) is False                      # unknown account path
    assert security.verify_password("x", "garbage") is False                 # malformed hash never raises


def test_session_cookie_flags(client, monkeypatch):
    r = client.post("/api/auth/register", json={"email": uniq(), "password": "correct-horse-9", "name": "Cy"})
    c = r.headers["set-cookie"].lower()
    assert "httponly" in c and "samesite=lax" in c and "path=/" in c and "max-age=" in c
    monkeypatch.setattr(config, "COOKIE_SECURE", True)                         # what production turns on
    r2 = client.post("/api/auth/register", json={"email": uniq(), "password": "correct-horse-9", "name": "Di"})
    assert "secure" in r2.headers["set-cookie"].lower()


def test_login_rotates_session_and_logout_invalidates_server_side(client):
    from app.server import STORE
    signup(client, uniq())
    token = client.cookies.get(config.COOKIE_NAME)
    client.post("/api/auth/logout")
    assert STORE.user_for_session(security.hash_token(token)) is None       # token is dead on the server, not just cleared in the browser


# =============================================================================== protected routes
PROTECTED = [
    ("GET", "/api/me"), ("PATCH", "/api/me"), ("DELETE", "/api/me"), ("GET", "/api/me/watchlist"), ("PUT", "/api/me/watchlist/NAVBANK"),
    ("DELETE", "/api/me/watchlist/NAVBANK"), ("POST", "/api/me/trades"), ("GET", "/api/me/trades"), ("GET", "/api/me/history"),
    ("GET", "/api/me/timeline"), ("GET", "/api/me/alert-rules"), ("PATCH", "/api/me/alert-rules"), ("GET", "/api/me/decisions"),
    ("POST", "/api/me/decisions"), ("POST", "/api/analyses"), ("GET", "/api/analyses"), ("GET", "/api/analyses/1"),
    ("DELETE", "/api/analyses/1"), ("GET", "/api/comparisons/NAVBANK"), ("GET", "/api/strategy"), ("GET", "/api/live"),
    ("GET", "/api/prices/NAVBANK"), ("GET", "/api/backtest"), ("GET", "/api/rag/stats"), ("GET", "/api/rag/search?q=pledge"),
    ("GET", "/api/docs/NAVBANK-ER-Q2FY27"), ("GET", "/api/benchmark"), ("GET", "/api/benchmark/load"),
]


@pytest.mark.parametrize("method,path", PROTECTED)
def test_every_protected_route_requires_sign_in(client, method, path):
    r = client.request(method, path, json={} if method in ("POST", "PUT", "PATCH") else None)
    assert r.status_code == 401, (method, path, r.status_code)
    assert only_error_key(r)


def test_public_routes_expose_no_personal_data(client):
    assert client.get("/api/health").status_code == 200
    meta = client.get("/api/meta").json()
    assert "users" not in meta and "email" not in str(meta)
    assert {p["id"] for p in meta["personas"]} == {"asha", "meera", "rohan"}
    assert client.get("/").status_code == 200


# =============================================================================== authorisation: nobody sees anybody else's data
def test_users_cannot_read_or_delete_each_others_analyses(alice, bob):
    run = alice.post("/api/analyses", json={}).json()["session_id"]
    assert alice.get(f"/api/analyses/{run}").status_code == 200
    r = bob.get(f"/api/analyses/{run}")
    assert r.status_code == 404 and only_error_key(r)                          # 404, not 403: existence is not revealed
    assert bob.delete(f"/api/analyses/{run}").status_code == 404
    assert alice.get(f"/api/analyses/{run}").status_code == 200                # untouched
    assert run not in [x["id"] for x in bob.get("/api/analyses").json()]


def test_users_cannot_act_on_each_others_recommendations(alice, bob):
    rec = alice.post("/api/analyses", json={}).json()["recommendations"][0]
    body = {"rec_id": rec["rec_id"], "symbol": rec["symbol"], "action": rec["action"], "choice": "accepted"}
    assert bob.post("/api/me/decisions", json=body).status_code == 404
    assert alice.post("/api/me/decisions", json=body).status_code == 201
    assert bob.get("/api/me/decisions").json() == []


def test_portfolios_watchlists_and_trades_are_private(alice, bob):
    alice.post("/api/me/trades", json={"symbol": "KISANFOODS", "side": "BUY", "qty": 10})
    alice.put("/api/me/watchlist/SAMRIDDHI")
    assert [h["symbol"] for h in alice.get("/api/me").json()["holdings"]] == ["KISANFOODS"]
    assert bob.get("/api/me").json()["holdings"] == []
    assert "SAMRIDDHI" in alice.get("/api/me/watchlist").json() and "SAMRIDDHI" not in bob.get("/api/me/watchlist").json()
    assert bob.get("/api/me/trades").json() == []
    assert bob.get("/api/me").json()["user"]["cash"] == 1_000_000
    assert all(t["symbol"] != "KISANFOODS" for t in bob.get("/api/me/history").json()["trades"])


def test_no_endpoint_takes_a_user_id_so_ids_cannot_be_tampered_with(alice):
    for path in ("/api/users/asha", "/api/users/asha/trades", "/api/users/asha/history"):
        assert alice.get(path).status_code == 404
    # a body that tries to name another user or role is rejected, not silently honoured
    assert alice.patch("/api/me", json={"id": "asha", "role": "template"}).status_code == 422
    assert alice.post("/api/analyses", json={"user_id": "asha"}).status_code == 422


def test_comparison_and_strategy_never_leak_other_customers(alice, bob):
    me_a = alice.get("/api/me").json()["user"]
    assert bob.patch("/api/me", json={"name": "Bob Secretname"}).status_code == 200
    for payload in (alice.get("/api/comparisons/NAVBANK").text, alice.get("/api/strategy").text):
        assert "Secretname" not in payload and "bob-" not in payload
    people = {r["user"]["name"] for r in alice.get("/api/comparisons/NAVBANK").json()["results"]}
    assert me_a["name"] in people and {"Asha Iyer", "Meera Nair", "Rohan Mehta"} <= people
    assert sum(1 for r in alice.get("/api/comparisons/NAVBANK").json()["results"] if r["is_you"]) == 1


def test_guest_sandboxes_are_isolated_from_each_other_and_from_the_template(api_app):
    from fastapi.testclient import TestClient
    from app.server import STORE
    with TestClient(api_app, raise_server_exceptions=False) as g1, TestClient(api_app, raise_server_exceptions=False) as g2:
        assert g1.post("/api/auth/guest", json={"persona": "meera"}).status_code == 201
        assert g2.post("/api/auth/guest", json={"persona": "meera"}).status_code == 201
        cash_before = g2.get("/api/me").json()["user"]["cash"]
        tmpl_cash = STORE.get_user("meera")["cash"]
        assert g1.post("/api/me/trades", json={"symbol": "KISANFOODS", "side": "BUY", "qty": 5}).status_code == 201
        assert g2.get("/api/me").json()["user"]["cash"] == cash_before        # other visitor unaffected
        assert STORE.get_user("meera")["cash"] == tmpl_cash                   # template unaffected
        assert g1.get("/api/me").json()["user"]["id"] != g2.get("/api/me").json()["user"]["id"]
        assert g1.get("/api/me").json()["user"]["role"] == "guest"


def test_templates_cannot_sign_in(client):
    r = client.post("/api/auth/login", json={"email": "asha@example.com", "password": "correct-horse-9"})
    assert r.status_code == 401


def test_delete_account_removes_everything_and_ends_the_session(client):
    from app.server import STORE
    signup(client, uniq())
    uid = client.get("/api/me").json()["user"]["id"]
    client.post("/api/me/trades", json={"symbol": "KISANFOODS", "side": "BUY", "qty": 3})
    client.post("/api/analyses", json={})
    assert client.delete("/api/me").status_code == 204
    assert client.get("/api/me").status_code == 401
    for table in ("users", "holdings", "trades", "watchlist", "analysis_runs", "recommendations", "snapshots", "decisions", "alert_rules", "auth_sessions"):
        col = "id" if table == "users" else "user_id"
        assert STORE._q(f"SELECT 1 FROM {table} WHERE {col}=?", (uid,)) == [], table


# =============================================================================== API design: verbs and status codes
def test_resource_verbs_and_status_codes(alice):
    r = alice.post("/api/analyses", json={"degradations": ["sentiment_feed_down"]})
    assert r.status_code == 201 and r.headers["location"] == f"/api/analyses/{r.json()['session_id']}"
    run = r.json()["session_id"]
    assert alice.get("/api/analyses").status_code == 200
    assert alice.delete(f"/api/analyses/{run}").status_code == 204
    assert alice.get(f"/api/analyses/{run}").status_code == 404
    assert alice.patch("/api/me", json={"risk_tolerance": 8}).json()["user"]["risk_tolerance"] == 8
    assert alice.put("/api/me/watchlist/TECHNOVA").status_code == 204
    assert alice.put("/api/me/watchlist/TECHNOVA").status_code == 204        # idempotent
    assert alice.delete("/api/me/watchlist/TECHNOVA").status_code == 204
    assert alice.delete("/api/me/watchlist/TECHNOVA").status_code == 404
    t = alice.post("/api/me/trades", json={"symbol": "navbank", "side": "buy", "qty": 2})
    assert t.status_code == 201 and t.json()["holdings"][0]["symbol"] == "NAVBANK"
    assert alice.get("/api/me/trades").status_code == 200
    assert alice.patch("/api/me/alert-rules", json={"daily_move_pct": 9}).json()["daily_move_pct"] == 9


def test_business_rule_conflicts_are_409_and_validation_is_422(alice):
    assert alice.post("/api/me/trades", json={"symbol": "BHARATOIL", "side": "BUY", "qty": 999_999}).status_code == 409
    assert alice.post("/api/me/trades", json={"symbol": "BHARATOIL", "side": "SELL", "qty": 1}).status_code == 409
    assert alice.post("/api/me/trades", json={"symbol": "BHARATOIL", "side": "BUY", "qty": 0}).status_code == 422
    assert alice.post("/api/me/trades", json={"symbol": "BHARATOIL", "side": "HODL", "qty": 1}).status_code == 422
    assert alice.post("/api/me/trades", json={"symbol": "FAKE", "side": "BUY", "qty": 1}).status_code == 404
    assert alice.patch("/api/me", json={"risk_tolerance": 11}).status_code == 422
    assert alice.patch("/api/me", json={}).status_code == 422
    assert alice.patch("/api/me/alert-rules", json={"nonsense": 1}).status_code == 422
    assert alice.post("/api/analyses", json={"degradations": ["meteor_strike"]}).status_code == 422
    assert alice.get("/api/comparisons/NOPE").status_code == 404
    assert alice.get("/api/live?tickers=bad ticker!").status_code == 422
    assert alice.get("/api/rag/search?q=x").status_code == 422
    assert alice.get("/api/me/trades?limit=0").status_code == 422


def test_trade_is_atomic_when_it_fails(alice):
    before = alice.get("/api/me").json()
    assert alice.post("/api/me/trades", json={"symbol": "BHARATOIL", "side": "BUY", "qty": 999_999}).status_code == 409
    after = alice.get("/api/me").json()
    assert before["user"]["cash"] == after["user"]["cash"] and before["holdings"] == after["holdings"]
    assert alice.get("/api/me/trades").json() == []


def test_analysis_is_rate_limited(alice, monkeypatch):
    monkeypatch.setitem(config.LIMITS, "analysis", (2, 60))
    codes = [alice.post("/api/analyses", json={}).status_code for _ in range(3)]
    assert codes == [201, 201, 429]
    assert only_error_key(alice.post("/api/analyses", json={}))


# =============================================================================== error handling
def test_unknown_route_and_wrong_method_use_the_error_shape(client):
    assert only_error_key(client.get("/api/does-not-exist")) and client.get("/api/does-not-exist").status_code == 404
    r = client.put("/api/me/trades", json={})
    assert r.status_code in (401, 405) and only_error_key(r)
    assert client.delete("/api/health").status_code == 405 and only_error_key(client.delete("/api/health"))


def test_unexpected_exception_returns_generic_500_without_internals(alice, monkeypatch):
    import app.server as srv

    def boom(*a, **k):
        raise RuntimeError("SECRET-INTERNAL-DETAIL /home/user/app.py line 42")

    monkeypatch.setattr(srv.ENGINE, "analyze", boom)
    r = alice.post("/api/analyses", json={})
    assert r.status_code == 500
    msg = only_error_key(r)
    assert "SECRET-INTERNAL" not in r.text and "RuntimeError" not in r.text and ".py" not in r.text
    assert "try again" in msg.lower() and r.headers.get("x-request-id")


def test_cross_site_writes_are_blocked_but_same_origin_works(alice):
    evil = alice.post("/api/analyses", json={}, headers={"Origin": "https://evil.example"})
    assert evil.status_code == 403 and only_error_key(evil)
    assert alice.post("/api/analyses", json={}, headers={"Origin": "http://testserver"}).status_code == 201
    assert alice.post("/api/analyses", json={}).status_code == 201            # non-browser clients send no Origin


def test_security_headers(client, monkeypatch):
    r = client.get("/api/health")
    h = r.headers
    assert h["x-content-type-options"] == "nosniff" and h["x-frame-options"] == "DENY" and "default-src 'self'" in h["content-security-policy"]
    assert h["cache-control"] == "no-store" and h["referrer-policy"]
    assert "strict-transport-security" not in h                                # development/test only
    monkeypatch.setattr(config, "HSTS", True)
    assert "max-age=" in client.get("/api/health").headers["strict-transport-security"]


# =============================================================================== database schema
def test_schema_is_normalised_with_enforced_foreign_keys(engine):
    st = engine.store
    tables = {r["name"] for r in st._q("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"instruments", "users", "auth_sessions", "holdings", "trades", "watchlist", "alert_rules", "analysis_runs",
            "recommendations", "snapshots", "decisions"} <= tables
    fks = {t: {r["table"] for r in st._q(f"PRAGMA foreign_key_list({t})")} for t in tables if not t.startswith("sqlite")}
    assert "users" in fks["holdings"] and "instruments" in fks["holdings"] and "users" in fks["trades"]
    assert {"users", "analysis_runs"} <= fks["recommendations"] and "recommendations" in fks["decisions"] and "users" in fks["auth_sessions"]
    assert st._q("PRAGMA foreign_keys")[0]["foreign_keys"] == 1
    cols = {r["name"] for r in st._q("PRAGMA table_info(users)")}
    assert "password_hash" in cols and "password" not in cols                    # hashes only


def test_database_rejects_orphans_and_invalid_values(engine):
    st = engine.store
    with pytest.raises(sqlite3.IntegrityError):
        st._x("INSERT INTO holdings VALUES('no-such-user','NAVBANK',1,1.0)")
    with pytest.raises(sqlite3.IntegrityError):
        st._x("INSERT INTO trades(user_id,ts,symbol,side,qty,price) VALUES('asha','2026-01-01','FAKESYM','BUY',1,1.0)")
    with pytest.raises(sqlite3.IntegrityError):
        st._x("INSERT INTO trades(user_id,ts,symbol,side,qty,price) VALUES('asha','2026-01-01','NAVBANK','SHORT',1,1.0)")
    with pytest.raises(sqlite3.IntegrityError):
        st.create_user({"id": "bad1", "name": "x", "risk_tolerance": 11, "role": "template"})
    with pytest.raises(sqlite3.IntegrityError):                                    # a member must have credentials
        st.create_user({"id": "bad2", "name": "x", "role": "member"})


def test_user_records_never_carry_the_password_hash(engine):
    st = engine.store
    from app.security import hash_password
    u = st.create_user({"id": uuid.uuid4().hex, "role": "member", "email": uniq(), "password_hash": hash_password("tiger-lily-2026"), "name": "Z"})
    assert "password_hash" not in u and "password_hash" not in engine.analyze(u["id"], persist=False)["user"]


# =============================================================================== environment management
def test_production_refuses_to_start_without_a_strong_secret(monkeypatch):
    monkeypatch.setattr(config, "IS_PROD", True)
    monkeypatch.setattr(config, "SECRET_KEY_ENV", "")
    with pytest.raises(RuntimeError, match="VF_SECRET_KEY"):
        config.validate()
    monkeypatch.setattr(config, "SECRET_KEY_ENV", "short")
    with pytest.raises(RuntimeError):
        config.validate()
    monkeypatch.setattr(config, "SECRET_KEY_ENV", "x" * 40)
    monkeypatch.setattr(config, "COOKIE_SECURE", False)
    with pytest.raises(RuntimeError, match="COOKIE_SECURE"):
        config.validate()
    monkeypatch.setattr(config, "COOKIE_SECURE", True)
    config.validate()


def test_dotenv_loader_does_not_override_real_environment(tmp_path, monkeypatch):
    f = tmp_path / ".env"
    f.write_text("# comment\nVF_TEST_A=from-file\nVF_TEST_B='quoted value'\nVF_TEST_C=\"dq\"\nnot a line\n")
    monkeypatch.setenv("VF_TEST_A", "from-real-env")
    monkeypatch.delenv("VF_TEST_B", raising=False)
    monkeypatch.delenv("VF_TEST_C", raising=False)
    config._load_dotenv(f)
    assert os.environ["VF_TEST_A"] == "from-real-env" and os.environ["VF_TEST_B"] == "quoted value" and os.environ["VF_TEST_C"] == "dq"
    for k in ("VF_TEST_B", "VF_TEST_C"):
        os.environ.pop(k, None)


def test_env_templates_exist_cover_every_setting_and_hold_no_secrets():
    example = (ROOT / ".env.example").read_text()
    prod = (ROOT / ".env.production.example").read_text()
    used = set(re.findall(r'"(VF_[A-Z_]+)"', (ROOT / "app" / "config.py").read_text()))
    missing = sorted(v for v in used if v not in example and v not in prod and v not in ("VF_DATA_DIR", "VF_LOG_DIR", "VF_COOKIE_NAME", "VF_SCRYPT_N", "VF_MAX_GUESTS", "VF_TRUST_PROXY", "VF_WORKERS"))
    assert not missing, missing
    assert re.search(r"^VF_ENV=production", prod, re.M) and re.search(r"^VF_ENV=development", example, re.M)
    for text in (example, prod):
        assert not re.search(r"VF_SECRET_KEY=[0-9a-fA-F]{32,}", text)            # placeholders only
    ignore = (ROOT / ".gitignore").read_text()
    assert re.search(r"^\.env$", ignore, re.M) and "!.env.example" in ignore


def test_no_secrets_committed_in_source():
    pat = re.compile(r"(sk-[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----|(?i:password|secret|token)\s*=\s*['\"][^'\"\s]{12,}['\"])")
    offenders = []
    for path in list((ROOT / "app").rglob("*.py")) + list((ROOT / "app").rglob("*.html")) + [ROOT / "run.py", ROOT / "render.yaml"]:
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if pat.search(line) and "example" not in line.lower():
                offenders.append(f"{path.name}:{i}")
    assert not offenders, offenders
    assert not (ROOT / ".env").exists() or ".env" in (ROOT / ".gitignore").read_text()


def test_development_and_production_profiles_differ():
    import importlib
    import app.config as cfg
    assert cfg.IS_TEST and not cfg.IS_PROD
    prod_defaults = (ROOT / ".env.production.example").read_text()
    assert "VF_COOKIE_SECURE=true" in prod_defaults and "VF_ENABLE_DOCS=false" in prod_defaults
    assert "VF_COOKIE_SECURE=false" in (ROOT / ".env.example").read_text()
    assert importlib.import_module("app.config").LIMITS["login"][0] == 5          # brute-force limit is NOT relaxed in tests

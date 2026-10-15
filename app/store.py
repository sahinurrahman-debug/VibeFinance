"""SQLite persistence (schema v2).

Normalised relational model with enforced foreign keys and CHECK constraints:

  instruments  (symbol PK)                          reference data for every tradable symbol
  users        (id PK, role, email UNIQUE, password_hash, risk limits, cash)
                 role = member   a real account: email + scrypt password hash
                        guest    an anonymous sandbox cloned from a template (expires)
                        template a seeded demo persona; cannot sign in; used as comparison baseline
  auth_sessions(user_id FK, token_hash UNIQUE, expires_at)   only an HMAC of the cookie token is stored
  holdings / trades / watchlist / alert_rules  -> user_id FK (ON DELETE CASCADE), symbol FK instruments
  analysis_runs(user_id FK)  -> recommendations(run_id FK, user_id FK) -> decisions(rec_id FK)
                             -> snapshots(run_id FK, user_id FK)

Every read/write helper that touches user data takes the user id and filters on it, so the API layer
can only ever reach the signed-in user's rows. Deleting a user cascades to everything they own.
Raw passwords are never stored or returned; `get_user` strips `password_hash`.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

SCHEMA_VERSION = 2

SCHEMA = """
CREATE TABLE IF NOT EXISTS instruments(
  symbol TEXT PRIMARY KEY, name TEXT NOT NULL, sector TEXT NOT NULL);

CREATE TABLE IF NOT EXISTS users(
  id TEXT PRIMARY KEY,
  role TEXT NOT NULL CHECK(role IN ('member','guest','template')),
  email TEXT UNIQUE COLLATE NOCASE,
  password_hash TEXT,
  name TEXT NOT NULL,
  age INTEGER CHECK(age IS NULL OR age BETWEEN 16 AND 110),
  bio TEXT NOT NULL DEFAULT '',
  risk_tolerance INTEGER NOT NULL CHECK(risk_tolerance BETWEEN 1 AND 10),
  max_position_pct REAL NOT NULL CHECK(max_position_pct BETWEEN 1 AND 100),
  max_sector_pct REAL NOT NULL CHECK(max_sector_pct BETWEEN 1 AND 100),
  max_volatility REAL NOT NULL CHECK(max_volatility BETWEEN 0.05 AND 2.0),
  horizon_years INTEGER NOT NULL DEFAULT 5 CHECK(horizon_years BETWEEN 1 AND 50),
  cash REAL NOT NULL CHECK(cash >= 0),
  derived_from TEXT REFERENCES users(id) ON DELETE SET NULL,
  created_at REAL NOT NULL,
  CHECK((role = 'member' AND email IS NOT NULL AND password_hash IS NOT NULL)
     OR (role != 'member' AND password_hash IS NULL))
);

CREATE TABLE IF NOT EXISTS auth_sessions(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  token_hash TEXT NOT NULL UNIQUE,
  created_at REAL NOT NULL, expires_at REAL NOT NULL, user_agent TEXT NOT NULL DEFAULT '');
CREATE INDEX IF NOT EXISTS idx_auth_sessions_user ON auth_sessions(user_id);

CREATE TABLE IF NOT EXISTS holdings(
  user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  symbol TEXT NOT NULL REFERENCES instruments(symbol),
  qty INTEGER NOT NULL CHECK(qty >= 0), avg_cost REAL NOT NULL CHECK(avg_cost >= 0),
  PRIMARY KEY(user_id, symbol));

CREATE TABLE IF NOT EXISTS trades(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  ts TEXT NOT NULL, symbol TEXT NOT NULL REFERENCES instruments(symbol),
  side TEXT NOT NULL CHECK(side IN ('BUY','SELL')), qty INTEGER NOT NULL CHECK(qty > 0),
  price REAL NOT NULL CHECK(price > 0), note TEXT NOT NULL DEFAULT '');
CREATE INDEX IF NOT EXISTS idx_trades_user ON trades(user_id, ts);

CREATE TABLE IF NOT EXISTS watchlist(
  user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  symbol TEXT NOT NULL REFERENCES instruments(symbol),
  PRIMARY KEY(user_id, symbol));

CREATE TABLE IF NOT EXISTS alert_rules(
  user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  key TEXT NOT NULL, value REAL NOT NULL, PRIMARY KEY(user_id, key));

CREATE TABLE IF NOT EXISTS analysis_runs(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  ts REAL NOT NULL, degradations TEXT NOT NULL, metrics TEXT NOT NULL, summary TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS idx_runs_user ON analysis_runs(user_id, id);

CREATE TABLE IF NOT EXISTS recommendations(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id INTEGER NOT NULL REFERENCES analysis_runs(id) ON DELETE CASCADE,
  user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  symbol TEXT NOT NULL REFERENCES instruments(symbol),
  action TEXT NOT NULL, confidence REAL NOT NULL, composite REAL NOT NULL, payload TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS idx_recs_run ON recommendations(run_id);

CREATE TABLE IF NOT EXISTS snapshots(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id INTEGER NOT NULL REFERENCES analysis_runs(id) ON DELETE CASCADE,
  user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  ts REAL NOT NULL, payload TEXT NOT NULL);

CREATE TABLE IF NOT EXISTS decisions(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  rec_id INTEGER REFERENCES recommendations(id) ON DELETE SET NULL,
  symbol TEXT NOT NULL REFERENCES instruments(symbol),
  action TEXT NOT NULL, choice TEXT NOT NULL CHECK(choice IN ('accepted','ignored','modified')),
  ts REAL NOT NULL, note TEXT NOT NULL DEFAULT '');
CREATE INDEX IF NOT EXISTS idx_decisions_user ON decisions(user_id, id);
"""

USER_FIELDS = ("name", "age", "bio", "risk_tolerance", "max_position_pct", "max_sector_pct",
               "max_volatility", "horizon_years", "cash")
_PUBLIC = "id, role, email, name, age, bio, risk_tolerance, max_position_pct, max_sector_pct, max_volatility, horizon_years, cash, derived_from, created_at"

MEMBER_DEFAULTS = dict(age=None, bio="", risk_tolerance=5, max_position_pct=20.0, max_sector_pct=40.0,
                       max_volatility=0.45, horizon_years=10, cash=1_000_000.0)


class Store:
    def __init__(self, path: Path | str):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        with self._lock:
            self._db.execute("PRAGMA foreign_keys=ON")
            try:
                self._db.execute("PRAGMA journal_mode=WAL")
            except sqlite3.DatabaseError:
                pass
            self._migrate()
            self._db.executescript(SCHEMA)
            self._db.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
            self._db.commit()

    def _migrate(self):
        """v0/v1 databases (no auth, no foreign keys) held only seeded demo data: rebuild them."""
        ver = self._db.execute("PRAGMA user_version").fetchone()[0]
        has_tables = self._db.execute("SELECT 1 FROM sqlite_master WHERE type='table' LIMIT 1").fetchone()
        if has_tables and ver < SCHEMA_VERSION:
            self._db.commit()
            self._db.execute("PRAGMA foreign_keys=OFF")
            for (t,) in self._db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'").fetchall():
                self._db.execute(f'DROP TABLE IF EXISTS "{t}"')
            self._db.commit()
            self._db.execute("PRAGMA foreign_keys=ON")

    def _q(self, sql, args=()):
        with self._lock:
            return [dict(r) for r in self._db.execute(sql, args).fetchall()]

    def _x(self, sql, args=()):
        with self._lock:
            cur = self._db.execute(sql, args)
            self._db.commit()
            return cur.lastrowid

    # ------------------------------------------------------------------ reference data
    def ensure_instruments(self, universe):
        with self._lock, self._db:
            for s in universe:
                self._db.execute("INSERT OR REPLACE INTO instruments VALUES(?,?,?)", (s.symbol, s.name, s.sector))

    # ------------------------------------------------------------------ users
    def is_empty(self) -> bool:
        return not self._q("SELECT 1 FROM users WHERE role='template' LIMIT 1")

    def create_user(self, u: dict) -> dict:
        row = {**MEMBER_DEFAULTS, **u}
        row.setdefault("role", "template")
        row.setdefault("email", None)
        row.setdefault("password_hash", None)
        row.setdefault("derived_from", None)
        row.setdefault("created_at", time.time())
        self._x("INSERT INTO users(id,role,email,password_hash,name,age,bio,risk_tolerance,max_position_pct,max_sector_pct,"
                "max_volatility,horizon_years,cash,derived_from,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (row["id"], row["role"], row["email"], row["password_hash"], row["name"], row["age"], row["bio"],
                 row["risk_tolerance"], row["max_position_pct"], row["max_sector_pct"], row["max_volatility"],
                 row["horizon_years"], row["cash"], row["derived_from"], row["created_at"]))
        return self.get_user(row["id"])

    def get_user(self, uid: str) -> dict | None:
        r = self._q(f"SELECT {_PUBLIC} FROM users WHERE id=?", (uid,))
        return r[0] if r else None

    def get_credentials(self, email: str) -> dict | None:
        """Login lookup only: returns the password hash. Never exposed through the API."""
        r = self._q("SELECT id, password_hash FROM users WHERE email=? AND role='member'", (email,))
        return r[0] if r else None

    def email_taken(self, email: str) -> bool:
        return bool(self._q("SELECT 1 FROM users WHERE email=?", (email,)))

    def list_templates(self) -> list[dict]:
        return self._q(f"SELECT {_PUBLIC} FROM users WHERE role='template' ORDER BY id")

    def update_user(self, uid: str, fields: dict) -> dict | None:
        fields = {k: v for k, v in fields.items() if k in USER_FIELDS}
        if fields:
            sets = ",".join(f"{k}=?" for k in fields)
            self._x(f"UPDATE users SET {sets} WHERE id=?", (*fields.values(), uid))
        return self.get_user(uid)

    def delete_user(self, uid: str) -> bool:
        with self._lock, self._db:
            return self._db.execute("DELETE FROM users WHERE id=? AND role != 'template'", (uid,)).rowcount > 0

    def count_guests(self) -> int:
        return self._q("SELECT COUNT(*) n FROM users WHERE role='guest'")[0]["n"]

    def clone_template(self, template_id: str, new_id: str) -> dict:
        """Create a private guest sandbox that starts as a copy of a seeded persona."""
        t = self.get_user(template_id)
        if not t or t["role"] != "template":
            raise KeyError(template_id)
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO users(id,role,name,age,bio,risk_tolerance,max_position_pct,max_sector_pct,max_volatility,"
                "horizon_years,cash,derived_from,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (new_id, "guest", t["name"], t["age"], t["bio"], t["risk_tolerance"], t["max_position_pct"], t["max_sector_pct"],
                 t["max_volatility"], t["horizon_years"], t["cash"], template_id, time.time()))
            self._db.execute("INSERT INTO holdings SELECT ?,symbol,qty,avg_cost FROM holdings WHERE user_id=?", (new_id, template_id))
            self._db.execute("INSERT INTO trades(user_id,ts,symbol,side,qty,price,note) SELECT ?,ts,symbol,side,qty,price,note "
                             "FROM trades WHERE user_id=? ORDER BY id", (new_id, template_id))
            self._db.execute("INSERT INTO watchlist SELECT ?,symbol FROM watchlist WHERE user_id=?", (new_id, template_id))
            self._db.execute("INSERT INTO alert_rules SELECT ?,key,value FROM alert_rules WHERE user_id=?", (new_id, template_id))
        return self.get_user(new_id)

    # ------------------------------------------------------------------ auth sessions
    def create_auth_session(self, uid: str, token_hash: str, ttl_s: float, user_agent: str = "") -> None:
        now = time.time()
        self._x("INSERT INTO auth_sessions(user_id,token_hash,created_at,expires_at,user_agent) VALUES(?,?,?,?,?)",
                (uid, token_hash, now, now + ttl_s, user_agent[:200]))

    def user_for_session(self, token_hash: str) -> dict | None:
        r = self._q(f"SELECT {', '.join('u.' + c.strip() for c in _PUBLIC.split(','))} FROM auth_sessions s JOIN users u ON u.id = s.user_id "
                    "WHERE s.token_hash=? AND s.expires_at > ?", (token_hash, time.time()))
        return r[0] if r else None

    def delete_auth_session(self, token_hash: str) -> None:
        self._x("DELETE FROM auth_sessions WHERE token_hash=?", (token_hash,))

    def purge_expired(self, guest_ttl_s: float) -> dict:
        """Remove expired sessions and expired guest sandboxes (cascades to all their data)."""
        now = time.time()
        with self._lock, self._db:
            s = self._db.execute("DELETE FROM auth_sessions WHERE expires_at <= ?", (now,)).rowcount
            g = self._db.execute("DELETE FROM users WHERE role='guest' AND created_at < ? AND id NOT IN "
                                 "(SELECT user_id FROM auth_sessions)", (now - guest_ttl_s,)).rowcount
        return {"sessions": s, "guests": g}

    # ------------------------------------------------------------------ holdings / trades
    def holdings(self, uid: str) -> list[dict]:
        return self._q("SELECT symbol, qty, avg_cost FROM holdings WHERE user_id=? AND qty>0 ORDER BY symbol", (uid,))

    def set_holding(self, uid: str, symbol: str, qty: int, avg_cost: float):
        self._x("INSERT OR REPLACE INTO holdings VALUES(?,?,?,?)", (uid, symbol, qty, avg_cost))

    def add_trade_log(self, uid: str, ts: str, symbol: str, side: str, qty: int, price: float, note: str = "") -> int:
        return self._x("INSERT INTO trades(user_id,ts,symbol,side,qty,price,note) VALUES(?,?,?,?,?,?,?)",
                       (uid, ts, symbol, side, qty, price, note))

    def trades(self, uid: str, limit: int | None = None) -> list[dict]:
        rows = self._q("SELECT * FROM trades WHERE user_id=? ORDER BY ts, id", (uid,))
        return rows[-limit:] if limit else rows

    def apply_trade(self, uid: str, ts: str, symbol: str, side: str, qty: int, price: float, note: str = "") -> dict:
        """Record a trade and update holdings + cash in ONE transaction (all or nothing)."""
        if side not in ("BUY", "SELL"):
            raise ValueError("side must be BUY or SELL")
        with self._lock:
            try:
                with self._db:
                    cash = self._db.execute("SELECT cash FROM users WHERE id=?", (uid,)).fetchone()
                    if cash is None:
                        raise KeyError(uid)
                    cash = cash["cash"]
                    h = self._db.execute("SELECT qty, avg_cost FROM holdings WHERE user_id=? AND symbol=?", (uid, symbol)).fetchone()
                    cur_q, cur_c = (h["qty"], h["avg_cost"]) if h else (0, 0.0)
                    if side == "BUY":
                        cost = qty * price
                        if cost > cash + 1e-6:
                            raise ValueError("Not enough cash for this purchase.")
                        new_q = cur_q + qty
                        self._db.execute("INSERT OR REPLACE INTO holdings VALUES(?,?,?,?)",
                                         (uid, symbol, new_q, (cur_q * cur_c + cost) / new_q))
                        self._db.execute("UPDATE users SET cash=cash-? WHERE id=?", (cost, uid))
                    else:
                        if qty > cur_q:
                            raise ValueError("You cannot sell more shares than you hold.")
                        self._db.execute("INSERT OR REPLACE INTO holdings VALUES(?,?,?,?)", (uid, symbol, cur_q - qty, cur_c))
                        self._db.execute("UPDATE users SET cash=cash+? WHERE id=?", (qty * price, uid))
                    tid = self._db.execute("INSERT INTO trades(user_id,ts,symbol,side,qty,price,note) VALUES(?,?,?,?,?,?,?)",
                                           (uid, ts, symbol, side, qty, price, note)).lastrowid
            except sqlite3.IntegrityError as e:
                raise ValueError("That trade violates a data rule.") from e
        return {"trade_id": tid, "holdings": self.holdings(uid), "cash": self.get_user(uid)["cash"]}

    # ------------------------------------------------------------------ watchlist
    def watchlist(self, uid: str) -> list[str]:
        return [r["symbol"] for r in self._q("SELECT symbol FROM watchlist WHERE user_id=? ORDER BY symbol", (uid,))]

    def set_watchlist(self, uid: str, symbols: list[str]):
        with self._lock, self._db:
            self._db.execute("DELETE FROM watchlist WHERE user_id=?", (uid,))
            for s in dict.fromkeys(symbols):
                self._db.execute("INSERT INTO watchlist VALUES(?,?)", (uid, s))

    def watch_add(self, uid: str, symbol: str) -> bool:
        with self._lock, self._db:
            return self._db.execute("INSERT OR IGNORE INTO watchlist VALUES(?,?)", (uid, symbol)).rowcount > 0

    def watch_remove(self, uid: str, symbol: str) -> bool:
        with self._lock, self._db:
            return self._db.execute("DELETE FROM watchlist WHERE user_id=? AND symbol=?", (uid, symbol)).rowcount > 0

    # ------------------------------------------------------------------ analysis runs / recommendations / decisions
    def save_session(self, uid: str, degradations: list[str], metrics: dict, summary: dict) -> int:
        return self._x("INSERT INTO analysis_runs(user_id, ts, degradations, metrics, summary) VALUES(?,?,?,?,?)",
                       (uid, time.time(), json.dumps(degradations), json.dumps(metrics), json.dumps(summary)))

    def save_recommendation(self, session_id: int, uid: str, rec: dict) -> int:
        return self._x("INSERT INTO recommendations(run_id,user_id,symbol,action,confidence,composite,payload)"
                       " VALUES(?,?,?,?,?,?,?)",
                       (session_id, uid, rec["symbol"], rec["action"], rec["confidence"], rec["composite"],
                        json.dumps(rec, default=str)))

    def get_recommendation(self, uid: str, rec_id: int) -> dict | None:
        r = self._q("SELECT * FROM recommendations WHERE id=? AND user_id=?", (rec_id, uid))
        return r[0] if r else None

    def save_decision(self, uid: str, rec_id: int | None, symbol: str, action: str, choice: str, note: str = "") -> int:
        if rec_id is not None and not self.get_recommendation(uid, rec_id):
            raise LookupError("recommendation not found")        # also covers "belongs to someone else"
        return self._x("INSERT INTO decisions(user_id,rec_id,symbol,action,choice,ts,note) VALUES(?,?,?,?,?,?,?)",
                       (uid, rec_id, symbol, action, choice, time.time(), note))

    def delete_decision(self, uid: str, decision_id: int) -> bool:
        with self._lock, self._db:
            return self._db.execute("DELETE FROM decisions WHERE id=? AND user_id=?", (decision_id, uid)).rowcount > 0

    def decisions(self, uid: str, limit: int = 50) -> list[dict]:
        return self._q("SELECT d.*, r.run_id AS session_id FROM decisions d LEFT JOIN recommendations r ON r.id = d.rec_id "
                       "WHERE d.user_id=? ORDER BY d.id DESC LIMIT ?", (uid, limit))

    def sessions(self, uid: str, limit: int = 20) -> list[dict]:
        rows = self._q("SELECT * FROM analysis_runs WHERE user_id=? ORDER BY id DESC LIMIT ?", (uid, limit))
        for r in rows:
            r["degradations"] = json.loads(r["degradations"])
            r["metrics"] = json.loads(r["metrics"])
            r["summary"] = json.loads(r["summary"])
        return rows

    def get_run(self, uid: str, run_id: int) -> dict | None:
        rows = self._q("SELECT * FROM analysis_runs WHERE id=? AND user_id=?", (run_id, uid))
        if not rows:
            return None
        r = rows[0]
        r["degradations"], r["metrics"], r["summary"] = json.loads(r["degradations"]), json.loads(r["metrics"]), json.loads(r["summary"])
        r["recommendations"] = [json.loads(x["payload"]) | {"rec_id": x["id"]} for x in
                                self._q("SELECT id, payload FROM recommendations WHERE run_id=? AND user_id=? ORDER BY id", (run_id, uid))]
        return r

    def delete_run(self, uid: str, run_id: int) -> bool:
        with self._lock, self._db:
            return self._db.execute("DELETE FROM analysis_runs WHERE id=? AND user_id=?", (run_id, uid)).rowcount > 0

    def recent_recs(self, uid: str, symbol: str | None = None, limit: int = 20) -> list[dict]:
        if symbol:
            return self._q("SELECT id, run_id AS session_id, symbol, action, confidence, composite FROM recommendations "
                           "WHERE user_id=? AND symbol=? ORDER BY id DESC LIMIT ?", (uid, symbol, limit))
        return self._q("SELECT id, run_id AS session_id, symbol, action, confidence, composite FROM recommendations "
                       "WHERE user_id=? ORDER BY id DESC LIMIT ?", (uid, limit))

    # ------------------------------------------------------------------ snapshots / alert rules
    def save_snapshot(self, session_id: int, uid: str, payload: dict) -> int:
        return self._x("INSERT INTO snapshots(run_id,user_id,ts,payload) VALUES(?,?,?,?)",
                       (session_id, uid, time.time(), json.dumps(payload, default=str)))

    def snapshots(self, uid: str, limit: int = 30) -> list[dict]:
        rows = self._q("SELECT *, run_id AS session_id FROM snapshots WHERE user_id=? ORDER BY id DESC LIMIT ?", (uid, limit))
        for r in rows:
            r["payload"] = json.loads(r["payload"])
        return rows[::-1]                                  # oldest first

    def alert_rules(self, uid: str, defaults: dict) -> dict:
        rules = dict(defaults)
        for r in self._q("SELECT key, value FROM alert_rules WHERE user_id=?", (uid,)):
            if r["key"] in defaults:
                rules[r["key"]] = r["value"]
        return rules

    def set_alert_rules(self, uid: str, rules: dict, defaults: dict):
        with self._lock, self._db:
            for k, v in rules.items():
                if k in defaults:
                    self._db.execute("INSERT OR REPLACE INTO alert_rules VALUES(?,?,?)", (uid, k, float(v)))

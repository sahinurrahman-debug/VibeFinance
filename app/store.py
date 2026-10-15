"""SQLite persistence: user profiles, holdings, trade log, watchlists, sessions,
recommendations and the user's recorded decisions (the 'persistent session' extension)."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS users(
  id TEXT PRIMARY KEY, name TEXT, age INTEGER, bio TEXT,
  risk_tolerance INTEGER, max_position_pct REAL, max_sector_pct REAL, max_volatility REAL,
  horizon_years INTEGER, cash REAL);
CREATE TABLE IF NOT EXISTS holdings(user_id TEXT, symbol TEXT, qty INTEGER, avg_cost REAL,
  PRIMARY KEY(user_id, symbol));
CREATE TABLE IF NOT EXISTS trades(id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT, ts TEXT,
  symbol TEXT, side TEXT, qty INTEGER, price REAL, note TEXT DEFAULT '');
CREATE TABLE IF NOT EXISTS watchlist(user_id TEXT, symbol TEXT, PRIMARY KEY(user_id, symbol));
CREATE TABLE IF NOT EXISTS sessions(id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT, ts REAL,
  degradations TEXT, metrics TEXT, summary TEXT);
CREATE TABLE IF NOT EXISTS recommendations(id INTEGER PRIMARY KEY AUTOINCREMENT, session_id INTEGER,
  user_id TEXT, symbol TEXT, action TEXT, confidence REAL, composite REAL, payload TEXT);
CREATE TABLE IF NOT EXISTS snapshots(id INTEGER PRIMARY KEY AUTOINCREMENT, session_id INTEGER, user_id TEXT, ts REAL,
  payload TEXT);
CREATE TABLE IF NOT EXISTS alert_rules(user_id TEXT, key TEXT, value REAL, PRIMARY KEY(user_id, key));
CREATE TABLE IF NOT EXISTS decisions(id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT, rec_id INTEGER,
  symbol TEXT, action TEXT, choice TEXT, ts REAL, note TEXT DEFAULT '');
"""

USER_FIELDS = ("name", "age", "bio", "risk_tolerance", "max_position_pct", "max_sector_pct",
               "max_volatility", "horizon_years", "cash")


class Store:
    def __init__(self, path: Path | str):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        with self._lock:
            self._db.executescript(SCHEMA)
            self._db.commit()

    def _q(self, sql, args=()):
        with self._lock:
            return [dict(r) for r in self._db.execute(sql, args).fetchall()]

    def _x(self, sql, args=()):
        with self._lock:
            cur = self._db.execute(sql, args)
            self._db.commit()
            return cur.lastrowid

    # ------------------------------------------------------------------ users
    def is_empty(self) -> bool:
        return not self._q("SELECT 1 FROM users LIMIT 1")

    def create_user(self, u: dict):
        self._x("INSERT OR REPLACE INTO users VALUES(?,?,?,?,?,?,?,?,?,?)",
                (u["id"], u["name"], u["age"], u["bio"], u["risk_tolerance"], u["max_position_pct"],
                 u["max_sector_pct"], u["max_volatility"], u["horizon_years"], u["cash"]))

    def list_users(self) -> list[dict]:
        return self._q("SELECT * FROM users ORDER BY id")

    def get_user(self, uid: str) -> dict | None:
        r = self._q("SELECT * FROM users WHERE id=?", (uid,))
        return r[0] if r else None

    def update_user(self, uid: str, fields: dict) -> dict | None:
        fields = {k: v for k, v in fields.items() if k in USER_FIELDS}
        if fields:
            sets = ",".join(f"{k}=?" for k in fields)
            self._x(f"UPDATE users SET {sets} WHERE id=?", (*fields.values(), uid))
        return self.get_user(uid)

    # ------------------------------------------------------------------ holdings / trades
    def holdings(self, uid: str) -> list[dict]:
        return self._q("SELECT symbol, qty, avg_cost FROM holdings WHERE user_id=? AND qty>0 ORDER BY symbol", (uid,))

    def set_holding(self, uid: str, symbol: str, qty: int, avg_cost: float):
        self._x("INSERT OR REPLACE INTO holdings VALUES(?,?,?,?)", (uid, symbol, qty, avg_cost))

    def add_trade_log(self, uid: str, ts: str, symbol: str, side: str, qty: int, price: float, note: str = "") -> int:
        return self._x("INSERT INTO trades(user_id,ts,symbol,side,qty,price,note) VALUES(?,?,?,?,?,?,?)",
                       (uid, ts, symbol, side, qty, price, note))

    def trades(self, uid: str) -> list[dict]:
        return self._q("SELECT * FROM trades WHERE user_id=? ORDER BY ts, id", (uid,))

    def apply_trade(self, uid: str, ts: str, symbol: str, side: str, qty: int, price: float, note: str = "") -> dict:
        """Record a trade and update holdings + cash atomically (persistent portfolio state)."""
        with self._lock:
            user = self.get_user(uid)
            cur = {h["symbol"]: h for h in self.holdings(uid)}.get(symbol, {"qty": 0, "avg_cost": 0.0})
            if side == "BUY":
                cost = qty * price
                if cost > user["cash"] + 1e-6:
                    raise ValueError("insufficient cash")
                new_qty = cur["qty"] + qty
                avg = (cur["qty"] * cur["avg_cost"] + cost) / new_qty
                self.set_holding(uid, symbol, new_qty, avg)
                self.update_user(uid, {"cash": user["cash"] - cost})
            elif side == "SELL":
                if qty > cur["qty"]:
                    raise ValueError("cannot sell more than held")
                self.set_holding(uid, symbol, cur["qty"] - qty, cur["avg_cost"])
                self.update_user(uid, {"cash": user["cash"] + qty * price})
            else:
                raise ValueError("side must be BUY or SELL")
            tid = self.add_trade_log(uid, ts, symbol, side, qty, price, note)
        return {"trade_id": tid, "holdings": self.holdings(uid), "cash": self.get_user(uid)["cash"]}

    def watchlist(self, uid: str) -> list[str]:
        return [r["symbol"] for r in self._q("SELECT symbol FROM watchlist WHERE user_id=? ORDER BY symbol", (uid,))]

    def set_watchlist(self, uid: str, symbols: list[str]):
        with self._lock:
            self._x("DELETE FROM watchlist WHERE user_id=?", (uid,))
            for s in symbols:
                self._x("INSERT OR IGNORE INTO watchlist VALUES(?,?)", (uid, s))

    # ------------------------------------------------------------------ sessions / recs / decisions
    def save_session(self, uid: str, degradations: list[str], metrics: dict, summary: dict) -> int:
        return self._x("INSERT INTO sessions(user_id, ts, degradations, metrics, summary) VALUES(?,?,?,?,?)",
                       (uid, time.time(), json.dumps(degradations), json.dumps(metrics), json.dumps(summary)))

    def save_recommendation(self, session_id: int, uid: str, rec: dict) -> int:
        return self._x("INSERT INTO recommendations(session_id,user_id,symbol,action,confidence,composite,payload)"
                       " VALUES(?,?,?,?,?,?,?)",
                       (session_id, uid, rec["symbol"], rec["action"], rec["confidence"], rec["composite"],
                        json.dumps(rec, default=str)))

    def get_recommendation(self, rec_id: int) -> dict | None:
        r = self._q("SELECT * FROM recommendations WHERE id=?", (rec_id,))
        return r[0] if r else None

    def save_decision(self, uid: str, rec_id: int | None, symbol: str, action: str, choice: str, note: str = "") -> int:
        return self._x("INSERT INTO decisions(user_id,rec_id,symbol,action,choice,ts,note) VALUES(?,?,?,?,?,?,?)",
                       (uid, rec_id, symbol, action, choice, time.time(), note))

    def decisions(self, uid: str, limit: int = 50) -> list[dict]:
        return self._q("SELECT d.*, r.session_id FROM decisions d LEFT JOIN recommendations r ON r.id = d.rec_id "
                       "WHERE d.user_id=? ORDER BY d.id DESC LIMIT ?", (uid, limit))

    def sessions(self, uid: str, limit: int = 20) -> list[dict]:
        rows = self._q("SELECT * FROM sessions WHERE user_id=? ORDER BY id DESC LIMIT ?", (uid, limit))
        for r in rows:
            r["degradations"] = json.loads(r["degradations"])
            r["metrics"] = json.loads(r["metrics"])
            r["summary"] = json.loads(r["summary"])
        return rows

    def recent_recs(self, uid: str, symbol: str | None = None, limit: int = 20) -> list[dict]:
        if symbol:
            rows = self._q("SELECT id, session_id, symbol, action, confidence, composite FROM recommendations "
                           "WHERE user_id=? AND symbol=? ORDER BY id DESC LIMIT ?", (uid, symbol, limit))
        else:
            rows = self._q("SELECT id, session_id, symbol, action, confidence, composite FROM recommendations "
                           "WHERE user_id=? ORDER BY id DESC LIMIT ?", (uid, limit))
        return rows

    # ------------------------------------------------------------------ snapshots / alert rules
    def save_snapshot(self, session_id: int, uid: str, payload: dict) -> int:
        return self._x("INSERT INTO snapshots(session_id,user_id,ts,payload) VALUES(?,?,?,?)",
                       (session_id, uid, time.time(), json.dumps(payload, default=str)))

    def snapshots(self, uid: str, limit: int = 30) -> list[dict]:
        rows = self._q("SELECT * FROM snapshots WHERE user_id=? ORDER BY id DESC LIMIT ?", (uid, limit))
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
        for k, v in rules.items():
            if k in defaults:
                self._x("INSERT OR REPLACE INTO alert_rules VALUES(?,?,?)", (uid, k, float(v)))

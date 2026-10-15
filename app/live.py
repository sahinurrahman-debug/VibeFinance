"""Live data adapters: real NSE prices, NSE corporate disclosures/earnings notices and market news.

Sources (all public, no API key):
  * prices & volume   - Yahoo Finance chart API  (`<TICKER>.NS`, 1y daily bars, dividend events)
  * disclosures       - NSE corporate-announcements API (results, board outcomes, investor-call notices, ...)
  * news / sentiment  - Google News RSS search

`LiveFeed` implements the same interface as the simulated `MarketFeed`, so the *same six agents*, the same
output contract, the same synthesis and the same degraded-data handling run on live data. Every source is
fetched independently with a timeout, a short TTL cache and its own status record; one source failing degrades
only the agents that depend on it. If the machine is offline the live tab reports that and nothing else breaks.

Limits (stated in the UI): FII flows and option-chain PCR have no keyless live source here, so the sentiment
agent works from headlines only; filing "tone" is scored on announcement subjects (the PDFs are linked, not parsed).
"""
from __future__ import annotations

import datetime as dt
import re
import threading
import time
import zlib
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from urllib.parse import quote

import numpy as np
import requests

from .corpus import Document
from .market import FeedResult, FeedUnavailable, Headline, LIVE_STOCKS, PriceSeries, Stock

UA = {"User-Agent": "Mozilla/5.0 (VibeFinance research demo)"}
TIMEOUT = 8
TTL = 300
TICKER_RE = re.compile(r"^[A-Z0-9&\-]{1,20}$")
DEFAULT_TICKERS = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "SBIN", "ITC"]
RELIABLE = {"reuters": 0.9, "livemint": 0.8, "mint": 0.8, "the economic times": 0.8, "economic times": 0.8, "moneycontrol": 0.75,
            "business standard": 0.8, "the hindu businessline": 0.8, "ndtv profit": 0.75, "cnbc-tv18": 0.75, "bloomberg": 0.9}
EARNINGS_WORDS = ("result", "earning", "investor meet", "conference call", "analysts", "dividend", "board meeting")


class LiveFeed:
    """Same interface as MarketFeed, backed by real public endpoints."""

    def __init__(self, http=None, ttl: int = TTL):
        self.http = http or requests.Session()
        self.ttl = ttl
        self._cache: dict = {}
        self._lock = threading.Lock()
        self.calls: dict[str, int] = {}
        self.status: dict[str, dict] = {}
        self.as_of = dt.date.today()
        self.io_scale = 1.0
        self._meta: dict[str, dict] = {}
        self._events: dict[str, dict] = {}

    # ------------------------------------------------------------------ plumbing
    def _io(self, source: str):                       # parity with MarketFeed (no simulated latency here)
        with self._lock:
            self.calls[source] = self.calls.get(source, 0) + 1

    def _record(self, source: str, ok: bool, detail: str, ms: float, url: str = ""):
        with self._lock:
            self.status[source] = {"ok": ok, "detail": detail, "ms": round(ms), "url": url,
                                   "at": dt.datetime.now().strftime("%H:%M:%S")}

    def _get(self, source: str, key: str, url: str, parse, headers=None):
        """Fetch + parse with TTL cache; raises FeedUnavailable (never anything else)."""
        now = time.time()
        with self._lock:
            hit = self._cache.get(key)
            if hit and now - hit[0] < self.ttl:
                return hit[1]
        self._io(source)
        t0 = time.perf_counter()
        try:
            r = self.http.get(url, headers={**UA, **(headers or {})}, timeout=TIMEOUT)
            if r.status_code != 200:
                raise FeedUnavailable(f"HTTP {r.status_code}")
            val = parse(r)
        except FeedUnavailable as e:
            self._record(source, False, str(e), (time.perf_counter() - t0) * 1000, url)
            raise
        except Exception as e:                        # noqa: BLE001 - network, JSON, XML...
            self._record(source, False, f"{type(e).__name__}: {str(e)[:80]}", (time.perf_counter() - t0) * 1000, url)
            raise FeedUnavailable(f"{source}: {type(e).__name__}") from e
        self._record(source, True, "ok", (time.perf_counter() - t0) * 1000, url)
        with self._lock:
            self._cache[key] = (now, val)
        return val

    # ------------------------------------------------------------------ prices (Yahoo Finance)
    def _chart(self, symbol: str):
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{quote(symbol)}.NS?range=1y&interval=1d&events=div%7Csplit"

        def parse(r):
            res = r.json()["chart"]["result"][0]
            off = res["meta"].get("gmtoffset", 19800)
            q = res["indicators"]["quote"][0]
            rows = [(dt.datetime.fromtimestamp(t + off, dt.timezone.utc).date(), c, v) for t, c, v in zip(res["timestamp"], q["close"], q["volume"])
                    if c is not None and v is not None]
            while rows and rows[-1][2] <= 0:           # in-progress / holiday bar with no volume yet
                rows.pop()
            if len(rows) < 70:
                raise FeedUnavailable("fewer than 70 daily bars")
            return {"meta": res["meta"], "rows": rows, "events": res.get("events", {})}

        d = self._get("Yahoo Finance (prices)", f"px:{symbol}", url, parse)
        self._meta[symbol] = d["meta"]
        self._events[symbol] = d["events"]
        return d, url

    def prices(self, symbol: str, degr=()) -> FeedResult:
        if "price_feed_down" in degr:
            raise FeedUnavailable("price feed disabled (degradation toggle)")
        d, url = self._chart(symbol)
        rows = d["rows"]
        self._meta[symbol] = d["meta"]
        self._events[symbol] = d["events"]
        warns, stale = [], 0
        if "price_feed_stale" in degr:
            rows = rows[:-3]
            stale = 3
            warns.append("Price feed lagging: last 3 sessions removed (degradation toggle)")
        dates = [r[0] for r in rows]
        self.as_of = dates[-1]
        return FeedResult(PriceSeries(dates, np.array([r[1] for r in rows], float), np.array([r[2] for r in rows], float)),
                          "Yahoo Finance NSE daily bars", dates[-1], stale, False, warns, url)

    def last_close(self, symbol: str) -> float:
        d, _ = self._chart(symbol)
        return float(d["rows"][-1][1])

    def company_name(self, symbol: str) -> str:
        m = self._meta.get(symbol, {})
        return m.get("longName") or m.get("shortName") or symbol

    # ------------------------------------------------------------------ news (Google News RSS)
    def news(self, symbol: str, name: str) -> list[Headline]:
        q = quote(f'"{name}" OR {symbol} NSE stock when:14d')
        url = f"https://news.google.com/rss/search?q={q}&hl=en-IN&gl=IN&ceid=IN:en"

        def parse(r):
            root = ET.fromstring(r.content)
            out = []
            for i, it in enumerate(root.iter("item")):
                title = (it.findtext("title") or "").strip()
                outlet = (it.findtext("source") or "").strip() or (title.rsplit(" - ", 1)[-1] if " - " in title else "news")
                if " - " in title:
                    title = title.rsplit(" - ", 1)[0]
                try:
                    d = parsedate_to_datetime(it.findtext("pubDate")).date()
                except Exception:                      # noqa: BLE001
                    continue
                out.append(Headline(f"GN-{symbol}-{d:%Y%m%d}-{i}", symbol, d, outlet, RELIABLE.get(outlet.lower(), 0.55), title,
                                    (it.findtext("link") or "").strip()))
            return out[:25]

        return self._get("Google News (headlines)", f"news:{symbol}", url, parse)

    def sentiment(self, symbol: str, degr=()) -> FeedResult:
        if "sentiment_feed_down" in degr:
            raise FeedUnavailable("news feed disabled (degradation toggle)")
        name = self.company_name(symbol) if symbol in self._meta else symbol
        hl = self.news(symbol, name)
        today = dt.date.today()
        recent = [h for h in hl if 0 <= (today - h.date).days <= 7]
        bundle = {"headlines": recent, "fii_5d": None, "pcr": None, "as_of": today}
        return FeedResult(bundle, "Google News RSS (no live FII/option-chain source)", today, 0, False,
                          [] if recent else ["No headlines in the last 7 days"], "")

    def cached_sentiment(self, symbol: str) -> FeedResult:
        raise FeedUnavailable("no cached live sentiment snapshot")

    # ------------------------------------------------------------------ disclosures (NSE)
    def announcements(self, symbol: str) -> list[dict]:
        url = f"https://www.nseindia.com/api/corporate-announcements?index=equities&symbol={quote(symbol)}"

        def parse(r):
            data = r.json()
            if not isinstance(data, list):
                raise FeedUnavailable("unexpected NSE payload")
            return data

        return self._get("NSE corporate announcements", f"nse:{symbol}", url, parse,
                         {"Referer": "https://www.nseindia.com/", "Accept": "application/json"})

    def documents(self, symbol: str, limit: int = 40) -> list[Document]:
        out = []
        for a in self.announcements(symbol)[:limit]:
            try:
                d = dt.datetime.strptime(a["an_dt"][:11], "%d-%b-%Y").date()
            except Exception:                           # noqa: BLE001
                continue
            subj = (a.get("attchmntText") or a.get("desc") or "").strip().replace("\n", " ")
            if len(subj) < 15:
                continue
            desc = (a.get("desc") or "Announcement").strip()
            did = f"NSE-{symbol}-{d:%Y%m%d}-{zlib.crc32(subj.encode()) % 100_000}"
            out.append(Document(did, symbol, ["Live market"], f"{desc} ({d.isoformat()}) - {symbol}", desc, d,
                                [(desc, subj)], "NSE corporate announcement (live)", "", False, a.get("attchmntFile") or ""))
        return out

    def earnings_and_actions(self, symbol: str, docs: list[Document]) -> list[dict]:
        rows = [{"date": d.date.isoformat(), "kind": d.doc_type, "text": d.sections[0][1], "url": d.url}
                for d in docs if any(w in (d.doc_type + " " + d.sections[0][1]).lower() for w in EARNINGS_WORDS)]
        for ev in (self._events.get(symbol, {}).get("dividends") or {}).values():
            rows.append({"date": dt.datetime.fromtimestamp(ev["date"], dt.timezone.utc).date().isoformat(), "kind": "Dividend (Yahoo)",
                         "text": f"Dividend of ₹{ev['amount']} per share", "url": ""})
        return sorted(rows, key=lambda r: r["date"], reverse=True)[:8]


def register(symbol: str, feed: LiveFeed):
    """Make a live ticker known to the rest of the system (sector lookup, naming)."""
    if symbol not in LIVE_STOCKS:
        try:
            price = feed.last_close(symbol)
        except FeedUnavailable:
            price = 0.0
        LIVE_STOCKS[symbol] = Stock(symbol, feed.company_name(symbol), "Live market", price, 0.3, 1.0, 0, 0.0, 0, 1.0, 0.0, "live")
    return LIVE_STOCKS[symbol]

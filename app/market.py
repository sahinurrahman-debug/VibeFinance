"""Simulated NSE-style market data + a feed layer with failure injection.

DISCLOSURE: every company, price, headline and flow here is synthetic. The generator is
deterministic (seeded) and embeds *mild* statistical structure - momentum persistence,
post-volume-event drift, and news that partly anticipates returns - so that the signal
dimensions can be evaluated end-to-end. Backtest accuracy therefore validates the
pipeline's mechanics, NOT real-market alpha. Swap `MarketFeed` for a live adapter
(NSE bhavcopy / vendor API) and nothing downstream changes.
"""
from __future__ import annotations

import datetime as dt
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from . import config

AS_OF = dt.date(2026, 10, 15)
N_DAYS = 520


@dataclass(frozen=True)
class Stock:
    symbol: str
    name: str
    sector: str
    price_now: float
    ann_vol: float
    beta: float
    avg_volume: int
    drift: float          # scenario drift/day overlay applied to the last `window` sessions
    window: int
    vol_mult: float       # volume multiplier over the last 3 sessions
    sent_bias: float      # news sentiment level over the last 14 sessions
    blurb: str = ""


UNIVERSE: list[Stock] = [
    Stock("BHARATOIL", "Bharat Petrochem & Retail", "Energy", 2874.0, 0.24, 1.00, 6_200_000, 0.0050, 30, 2.6, 0.55, "Bull consensus: price, volume, news and filings agree"),
    Stock("TECHNOVA", "Technova Services", "IT", 3912.0, 0.22, 0.85, 2_100_000, 0.0028, 30, 1.2, 0.20, "Quiet uptrend, steady disclosures"),
    Stock("NAVBANK", "Navbharat Bank", "Banking", 1688.0, 0.21, 1.05, 9_800_000, 0.0036, 25, 1.4, 0.25, "Recovery with mixed margin commentary"),
    Stock("INFOSPHERE", "Infosphere Digital", "IT", 1544.0, 0.26, 0.90, 7_400_000, -0.0045, 25, 1.7, 0.45, "CONFLICT: price falling while filings & news are upbeat"),
    Stock("SAMRIDDHI", "Samriddhi Finance", "Financials", 7120.0, 0.28, 1.15, 1_300_000, 0.0014, 30, 1.0, 0.05, "Range-bound lender"),
    Stock("KISANFOODS", "Kisan Foods & Hotels", "FMCG", 468.0, 0.17, 0.60, 11_500_000, 0.0001, 30, 0.8, 0.00, "Defensive, sideways"),
    Stock("RASHTRIYA", "Rashtriya Bank of India", "Banking", 812.0, 0.30, 1.20, 24_000_000, 0.0055, 20, 3.2, 0.40, "PSU-bank rally on heavy volume"),
    Stock("SANCHARNET", "Sanchar Telecom", "Telecom", 1721.0, 0.23, 0.75, 4_600_000, 0.0038, 30, 1.4, 0.30, "Steady compounder, ARPU story"),
    Stock("VIKRAMINFRA", "Vikram Infra & Ports", "Infrastructure", 2406.0, 0.50, 1.55, 3_800_000, 0.0070, 20, 3.8, 0.10, "CONFLICT: parabolic rally vs adverse regulatory filings"),
    Stock("GATIMOTORS", "Gati Motors", "Auto", 946.0, 0.34, 1.30, 15_000_000, -0.0060, 25, 2.5, -0.50, "Bear consensus: distribution volume, weak guidance"),
    Stock("AROGYAPHARMA", "Arogya Pharma", "Pharma", 1633.0, 0.25, 0.70, 2_900_000, 0.0016, 30, 1.1, 0.05, "Mixed: domestic strength vs regulatory warning letter"),
    Stock("HINDCONSUMER", "Hind Consumer Products", "FMCG", 2402.0, 0.18, 0.55, 1_900_000, -0.0026, 30, 0.7, -0.25, "Slow drift lower, volume drying up"),
]
SYMBOLS = [s.symbol for s in UNIVERSE]
BY_SYMBOL = {s.symbol: s for s in UNIVERSE}
LIVE_STOCKS: dict[str, Stock] = {}          # tickers registered by the live-data adapters


def get_stock(symbol: str) -> Stock:
    return BY_SYMBOL.get(symbol) or LIVE_STOCKS[symbol]

SOURCES = [  # (name, reliability weight)
    ("Exchange Filing", 1.0), ("Business Daily", 0.8), ("Market Wire", 0.8),
    ("Brokerage Note", 0.7), ("Analyst Blog", 0.5), ("Social Media Chatter", 0.3),
]
HL_POS = [
    "{n} beats estimates as quarterly profit jumps on strong demand",
    "Brokerages raise target on {n} citing robust order inflow",
    "{n} wins large multi-year contract; order book at record high",
    "Funds increase their stake in {n} as margin expansion continues",
    "{n} shares rally on healthy volume growth and improved outlook",
    "{n} board approves dividend and buyback",
    "Analysts upgrade {n}; see visibility on earnings recovery",
]
HL_NEG = [
    "{n} shares slide as brokerages cut estimates",
    "{n} faces margin pressure from higher input costs, analysts flag concerns",
    "Regulator issues show-cause notice to {n} over related-party dealings",
    "{n} downgraded by analysts on weak demand and slowdown fears",
    "Promoter pledge rises at {n}, investors raise concerns",
    "{n} missed estimates; guidance cut on delayed projects",
    "{n} stock tumbles amid sell-off in the sector",
]
HL_NEU = [
    "{n} to announce quarterly results next week",
    "{n} management to meet investors at industry conference",
    "{n} names new head for an internal division",
    "Trading volumes in {n} remain in a narrow range",
]


@dataclass
class Headline:
    id: str
    symbol: str
    date: dt.date
    source: str
    reliability: float
    text: str
    url: str = ""

    def to_dict(self):
        return {"id": self.id, "symbol": self.symbol, "date": self.date.isoformat(),
                "source": self.source, "reliability": self.reliability, "text": self.text}


@dataclass
class PriceSeries:
    dates: list[dt.date]
    close: np.ndarray
    volume: np.ndarray


@dataclass
class FeedResult:
    data: object
    source: str
    as_of: dt.date
    stale_days: int = 0
    from_cache: bool = False
    warnings: list[str] = field(default_factory=list)
    url: str = ""


class FeedUnavailable(Exception):
    pass


def _business_days(end: dt.date, n: int) -> list[dt.date]:
    out, d = [], end
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d -= dt.timedelta(days=1)
    return out[::-1]


class MarketData:
    """Deterministic synthetic universe (the 'ground truth' behind the feeds)."""

    def __init__(self, seed: int = config.SEED):
        self.seed = seed
        self.dates = _business_days(AS_OF, N_DAYS)
        self._idx = {d: i for i, d in enumerate(self.dates)}
        self.close: dict[str, np.ndarray] = {}
        self.volume: dict[str, np.ndarray] = {}
        self.rets: dict[str, np.ndarray] = {}
        self.sent_idx: dict[str, np.ndarray] = {}
        self.pcr: dict[str, np.ndarray] = {}
        self.headlines: dict[str, list[Headline]] = {s: [] for s in SYMBOLS}
        self._generate()

    # ------------------------------------------------------------------ generation
    def _generate(self):
        n = N_DAYS
        rm = np.random.default_rng(self.seed)
        mret = rm.normal(0.0003, 0.0078, n)
        mret[-12:] += 0.0008                       # gently positive tape into the as-of date
        self.fii = 800 * (mret / 0.0078) + rm.normal(0, 900, n)
        self.fii[-5:] += 150
        for i, st in enumerate(UNIVERSE):
            rng = np.random.default_rng(self.seed + 101 * (i + 1))
            sigma = st.ann_vol / np.sqrt(252)
            shock = np.zeros(n)
            cont = np.zeros(n)
            volev = np.ones(n)
            for d in rng.choice(np.arange(40, n - 45), size=9, replace=False):
                direction = rng.choice([-1, 1])
                shock[d] += direction * 3.0 * sigma
                cdir = direction if rng.random() < 0.75 else -direction   # 75% continuation
                cont[d + 1: d + 13] += cdir * 0.0025
                volev[d] = 3.0
                volev[d + 1] = 1.6
            overlay = np.zeros(n)
            overlay[-st.window:] = st.drift * (0.5 + 0.5 * np.linspace(0, 1, st.window))
            ret = np.zeros(n)
            mu = 0.0                               # slowly-varying drift regime (trends persist ~1-2 months)
            for t in range(1, n):
                mu = 0.97 * mu + rng.normal(0, 0.0006)
                ret[t] = st.beta * mret[t] + rng.normal(0, sigma * 0.85) + 0.05 * ret[t - 1] \
                    + mu + cont[t] + shock[t] + overlay[t]
            if st.vol_mult >= 2:                   # make the last 3 sessions unambiguous
                sgn = np.sign(st.drift)
                for k in (1, 2, 3):
                    ret[-k] = sgn * max(abs(ret[-k]), 0.8 * sigma)
            close = np.exp(np.cumsum(ret))
            close = close / close[-1] * st.price_now
            vol = st.avg_volume * rng.lognormal(0, 0.28, n) * (1 + 0.5 * np.abs(ret) / sigma) * volev
            vm = np.ones(n)
            vm[-10:-3] = 1 + (st.vol_mult - 1) * 0.25
            vm[-3:] = st.vol_mult
            vol = (vol * vm).astype(np.int64)
            # sentiment index: partly reactive to the past, partly anticipating the future (by design)
            c5 = np.convolve(ret, np.ones(5), mode="full")[:n]
            f10 = np.array([ret[t + 1: t + 11].sum() for t in range(n)])
            s = np.tanh(c5 / 0.04) * 0.45 + 0.15 * np.tanh(f10 / 0.06) + rng.normal(0, 0.15, n)
            s[-14:] = st.sent_bias + rng.normal(0, 0.12, 14)
            s = np.clip(s, -1, 1)
            self.close[st.symbol] = close
            self.volume[st.symbol] = vol
            self.rets[st.symbol] = ret
            self.sent_idx[st.symbol] = s
            self.pcr[st.symbol] = np.clip(1.0 - 0.5 * s + rng.normal(0, 0.1, n), 0.5, 1.8)
            self._make_headlines(st, s, rng)
        self.index = np.mean([np.diff(np.log(c)) for c in self.close.values()], axis=0)

    def _make_headlines(self, st: Stock, s: np.ndarray, rng):
        for t, d in enumerate(self.dates):
            p_news = 0.55 if t >= N_DAYS - 14 else 0.28
            if rng.random() > p_news:
                continue
            for k in range(1 + int(rng.random() < 0.35)):
                pol = float(np.clip(s[t] + rng.normal(0, 0.35), -1.5, 1.5))
                pool = HL_POS if pol > 0.2 else HL_NEG if pol < -0.2 else HL_NEU
                src, rel = SOURCES[int(rng.integers(0, len(SOURCES)))]
                text = pool[int(rng.integers(0, len(pool)))].format(n=st.name)
                self.headlines[st.symbol].append(
                    Headline(f"N-{st.symbol}-{d:%Y%m%d}-{k}", st.symbol, d, src, rel, text))

    # ------------------------------------------------------------------ accessors
    def idx(self, d: dt.date) -> int:
        return self._idx[d]

    def headlines_between(self, symbol: str, start: dt.date, end: dt.date) -> list[Headline]:
        return [h for h in self.headlines[symbol] if start <= h.date <= end]


# ---------------------------------------------------------------------- feed layer
def validate_prices(ps: PriceSeries) -> tuple[PriceSeries, list[str], float]:
    """Data-quality gate. Returns (clean series, warnings, confidence multiplier)."""
    close, vol = ps.close.astype(float).copy(), ps.volume.astype(float).copy()
    warnings, mult = [], 1.0
    nan_n = int(np.isnan(close).sum())
    if nan_n:
        for i in range(len(close)):
            if np.isnan(close[i]):
                close[i] = close[i - 1] if i else np.nanmean(close)
        warnings.append(f"{nan_n} missing close(s) forward-filled")
        mult *= max(0.5, 1 - 0.12 * nan_n)
    spikes = 0
    for i in range(1, len(close) - 1):
        r1, r2 = close[i] / close[i - 1] - 1, close[i + 1] / close[i] - 1
        if abs(r1) > 0.25 and r1 * r2 < 0 and abs(r2) > 0.15:
            close[i] = float(np.sqrt(close[i - 1] * close[i + 1]))
            spikes += 1
    if spikes:
        warnings.append(f"{spikes} glitch spike(s) (>25% reversed next bar) smoothed")
        mult *= max(0.5, 1 - 0.15 * spikes)
    bad_v = vol <= 0
    if bad_v.any():
        vol[bad_v] = np.median(vol[~bad_v])
        warnings.append(f"{int(bad_v.sum())} non-positive volume bar(s) replaced by median")
        mult *= 0.9
    return PriceSeries(ps.dates, close, vol), warnings, mult


class MarketFeed:
    """What the agents see. Adds latency and injectable outages on top of `MarketData`."""

    def __init__(self, data: MarketData, io_scale: float = config.IO_SCALE):
        self.data = data
        self.io_scale = io_scale
        # last-good snapshots (what a real system would keep in Redis / on disk)
        self._sent_cache_end = len(data.dates) - 1 - 3
        self.calls: dict[str, int] = {}           # feed round-trips by source (for the cache metrics)
        self._calls_lock = threading.Lock()

    def _io(self, source: str):
        with self._calls_lock:
            self.calls[source] = self.calls.get(source, 0) + 1
        ms = config.FEED_LATENCY_MS.get(source, 50) * self.io_scale
        if ms > 0:
            time.sleep(ms / 1000.0)

    def prices(self, symbol: str, degr=()) -> FeedResult:
        self._io("prices")
        if "price_feed_down" in degr:
            raise FeedUnavailable("NSE price feed timed out (HTTP 503)")
        d = self.data
        end = len(d.dates)
        warnings, stale = [], 0
        if "price_feed_stale" in degr:
            end -= 3
            stale = 3
            warnings.append("Price feed lagging: last 3 sessions missing; analysis is as of "
                            f"{d.dates[end - 1].isoformat()}")
        close, vol = d.close[symbol][:end].copy(), d.volume[symbol][:end].copy()
        if "price_corrupt" in degr:
            close[-8] = np.nan
            close[-4] = close[-5] * 1.45
            warnings.append("Feed returned NaN / glitch bars (see data-quality gate)")
        return FeedResult(PriceSeries(d.dates[:end], close, vol), "NSE price feed (simulated)",
                          d.dates[end - 1], stale, False, warnings)

    def sentiment(self, symbol: str, degr=()) -> FeedResult:
        self._io("sentiment")
        if "sentiment_feed_down" in degr:
            raise FeedUnavailable("News/sentiment API unreachable")
        return self._sentiment_at(symbol, len(self.data.dates) - 1, "News + FII + options feed (simulated)")

    def cached_sentiment(self, symbol: str) -> FeedResult:
        r = self._sentiment_at(symbol, self._sent_cache_end, "Cached sentiment snapshot")
        r.from_cache = True
        r.stale_days = len(self.data.dates) - 1 - self._sent_cache_end
        r.warnings.append(f"Using cached sentiment snapshot from {r.as_of.isoformat()} "
                          f"({r.stale_days} sessions old)")
        return r

    def _sentiment_at(self, symbol: str, t: int, label: str) -> FeedResult:
        d = self.data
        end = d.dates[t]
        bundle = {
            "headlines": d.headlines_between(symbol, end - dt.timedelta(days=7), end),
            "fii_5d": float(d.fii[t - 4: t + 1].sum()),
            "pcr": float(d.pcr[symbol][t]),
            "as_of": end,
        }
        return FeedResult(bundle, label, end)

    @property
    def as_of(self) -> dt.date:
        return self.data.dates[-1]

    def last_close(self, symbol: str) -> float:
        """System-of-record last close, used for portfolio valuation even in an outage."""
        return float(self.data.close[symbol][-1])

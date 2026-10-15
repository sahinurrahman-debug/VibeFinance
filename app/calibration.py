"""Confidence calibration and uncertainty estimation.

* Isotonic regression (pool-adjacent-violators) maps the ensemble's raw confidence to the empirical
  probability that the signal's direction is right over the next 30 days.
* A bootstrap (resampling the back-test observations and re-fitting) gives a 90% band around that
  probability, so the UI can show "63% (52-71%)" rather than a point estimate.
* The calibrator is first fitted on the earlier 60% of dates and scored on the later 40%
  (out-of-sample ECE / Brier), then re-fitted on all data for live use.
* Leave-one-agent-out counterfactuals explain disagreements: which agent is decisive?
"""
from __future__ import annotations

import numpy as np

from . import config

GRID = np.linspace(0.15, 0.95, 41)


SHRINK = 40.0      # pseudo-observations pulling sparse blocks toward the base rate (stops edge overfitting)


def pav(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Increasing isotonic fit with shrinkage toward the base rate. Returns (block x-centres, block means)."""
    order = np.argsort(x, kind="stable")
    xs, ys = x[order], y[order].astype(float)
    blocks = [[ys[i], 1.0, xs[i], xs[i]] for i in range(len(ys))]       # mean, weight, xmin, xmax
    out: list[list[float]] = []
    for b in blocks:
        out.append(b)
        while len(out) > 1 and out[-2][0] > out[-1][0]:
            m2, w2, a2, b2 = out.pop()
            m1, w1, a1, b1 = out.pop()
            out.append([(m1 * w1 + m2 * w2) / (w1 + w2), w1 + w2, a1, b2])
    base = float(ys.mean())
    means = np.array([(b[0] * b[1] + SHRINK * base) / (b[1] + SHRINK) for b in out])
    return np.array([(b[2] + b[3]) / 2 for b in out]), np.maximum.accumulate(means)


def _predict(fit: tuple[np.ndarray, np.ndarray], c) -> np.ndarray:
    return np.interp(c, fit[0], fit[1])


def _ece(p: np.ndarray, y: np.ndarray, bins: int = 5) -> float:
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    e, n = 0.0, len(p)
    for i in range(bins):
        m = (p >= edges[i]) & ((p <= edges[i + 1]) if i == bins - 1 else (p < edges[i + 1]))
        if m.any():
            e += m.sum() / n * abs(p[m].mean() - y[m].mean())
    return float(e)


class Calibrator:
    def __init__(self, bt: dict, boot: int = 200, seed: int = 11):
        recs = [r for r in bt["_records"] if abs(r["score"]) >= config.SIGNAL_THRESHOLD]
        conf = np.array([r["conf"] for r in recs])
        hit = np.array([float(np.sign(r["score"]) == np.sign(r["fwd"])) for r in recs])
        ts = np.array([r["t"] for r in recs])
        self.n = len(recs)
        self.base_hit = float(hit.mean()) if self.n else 0.5
        cut = np.quantile(ts, 0.6)
        tr, te = ts <= cut, ts > cut
        self.report = {"n_train": int(tr.sum()), "n_test": int(te.sum())}
        if tr.sum() > 30 and te.sum() > 30:
            fit_tr = pav(conf[tr], hit[tr])
            p_cal = _predict(fit_tr, conf[te])
            p_raw = np.clip(conf[te], 0, 1)
            self.report.update({
                "ece_raw": round(_ece(p_raw, hit[te]), 4), "ece_calibrated": round(_ece(p_cal, hit[te]), 4),
                "brier_raw": round(float(np.mean((p_raw - hit[te]) ** 2)), 4),
                "brier_calibrated": round(float(np.mean((p_cal - hit[te]) ** 2)), 4),
                "brier_climatology": round(float(np.mean((hit[tr].mean() - hit[te]) ** 2)), 4),
                "test_hit_rate": round(float(hit[te].mean()), 4)})
            edges = np.quantile(p_cal, np.linspace(0, 1, 5))
            rel = []
            for i in range(4):
                m = (p_cal >= edges[i]) & ((p_cal <= edges[i + 1]) if i == 3 else (p_cal < edges[i + 1]))
                if m.any():
                    rel.append({"n": int(m.sum()), "mean_raw_conf": round(float(p_raw[m].mean()), 3),
                                "mean_calibrated": round(float(p_cal[m].mean()), 3), "actual": round(float(hit[te][m].mean()), 3)})
            self.report["reliability"] = rel
        self.fit = pav(conf, hit) if self.n else (np.array([0.5]), np.array([0.5]))
        rng = np.random.default_rng(seed)
        curves = []
        for _ in range(boot if self.n else 0):
            idx = rng.integers(0, self.n, self.n)
            curves.append(_predict(pav(conf[idx], hit[idx]), GRID))
        if curves:
            c = np.vstack(curves)
            self.lo, self.hi = np.percentile(c, 5, axis=0), np.percentile(c, 95, axis=0)
        else:
            self.lo = self.hi = np.full_like(GRID, 0.5)
        self.buckets = bt["ensemble"]["calibration"]

    def predict(self, conf: float) -> dict:
        c = float(np.clip(conf, GRID[0], GRID[-1]))
        p = float(_predict(self.fit, c))
        return {"p": round(p, 3), "lo": round(float(np.interp(c, GRID, self.lo)), 3),
                "hi": round(float(np.interp(c, GRID, self.hi)), 3), "n": self.n, "raw_confidence": round(conf, 3)}

    def lookup(self, conf: float) -> dict:
        """Bucketed empirical hit rate (kept for the reliability table)."""
        for b in self.buckets:
            if b["lo"] <= conf < b["hi"]:
                return {"bucket": b["range"], "n": b["n"], "hit_rate": b["hit_rate"], "smoothed": b["smoothed"]}
        return {"bucket": None, "n": 0, "hit_rate": None, "smoothed": None}

    def summary(self) -> dict:
        return {"n_observations": self.n, "base_hit_rate": round(self.base_hit, 4), **self.report,
                "curve": [{"conf": round(float(c), 3), "p": round(float(_predict(self.fit, c)), 3),
                           "lo": round(float(l), 3), "hi": round(float(h), 3)}
                          for c, l, h in list(zip(GRID, self.lo, self.hi))[::4]]}


def zone(composite: float, buy_thr: float, sell_thr: float) -> str:
    return "bullish" if composite >= buy_thr else "bearish" if composite <= sell_thr else "neutral"


def counterfactuals(usable: dict, weights: dict, buy_thr: float, sell_thr: float, composite: float) -> list[dict]:
    """Leave-one-agent-out: how would the verdict change without each agent? Marks decisive agents."""
    base_zone = zone(composite, buy_thr, sell_thr)
    out = []
    for a in usable:
        rest = {k: v for k, v in usable.items() if k != a}
        eff = {k: weights[k] * v.confidence for k, v in rest.items()}
        tot = sum(eff.values())
        if tot <= 0:
            continue
        c = sum(eff[k] * rest[k].score for k in rest) / tot
        z = zone(c, buy_thr, sell_thr)
        out.append({"without": a, "composite": round(c, 3), "delta": round(c - composite, 3), "zone": z,
                    "decisive": z != base_zone})
    return sorted(out, key=lambda x: -abs(x["delta"]))

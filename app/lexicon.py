"""Small finance-tone lexicon (Loughran-McDonald style, hand-curated for this demo).

Used both for headline polarity and for scoring retrieved filing passages. Matching is
phrase-first (longest wins, matched spans are blanked) so that "net debt fell" style
phrases are not double counted by their single-word components.
"""
from __future__ import annotations

import re

POS = [
    "record", "beat estimates", "beats estimates", "beats", "beat", "raised guidance", "raises guidance",
    "raised fy27", "upgrade", "upgraded", "order book", "order inflow", "deleveraging", "deleveraged",
    "margin expansion", "margins expanded", "margin improvement", "strong demand", "market share gains",
    "gained market share", "gain market share", "robust", "healthy", "improved", "dividend", "buyback",
    "wins", "won", "approval", "approved", "accretive", "recovery", "resilient", "visibility",
    "no shares pledged", "no pledge", "outperform", "surge", "rally", "rallies", "growth", "grew",
    "raise target", "raise their target", "increase their stake", "raise stake", "net debt fell",
    "gnpa fell", "asset quality improved", "reduced debt", "debt reduction", "profit jumps", "expands",
    "no material litigation", "increased their stake", "supportive", "highest-ever", "positioned to announce",
    "wins", "no pending regulatory action", "no regulatory orders", "not subject to any regulatory action",
]
NEG = [
    "pledge", "pledged", "probe", "investigation", "show-cause", "penalty", "penalties", "fine",
    "litigation", "downgrade", "downgraded", "warning letter", "margin pressure", "impairment",
    "write-off", "delay", "delayed", "default", "weak demand", "slowdown", "headwinds", "cuts guidance",
    "cut guidance", "cut estimates", "cut their estimates", "cut target", "missed estimates", "miss",
    "missed", "contingent liabilities", "qualified opinion", "resignation", "resigned", "recall",
    "shortfall", "elevated", "stress", "npas rose", "slippages rose", "going concern", "related-party",
    "related party", "slide", "slump", "plunge", "tumble", "scrutiny", "volatile", "concern", "concerns",
    "weaker", "weakness", "declined", "decline", "shed", "sell-off", "profit falls", "lower guidance",
]

_SIGN = {t: 1 for t in POS}
_SIGN.update({t: -1 for t in NEG})
# One alternation, longest phrase first: at any position the longest matching phrase wins, and
# matches never overlap, so "net debt fell" is not also counted as "debt".
_RX = re.compile(r"(?<![a-z])(?:" + "|".join(re.escape(t) for t in sorted(_SIGN, key=len, reverse=True)) + r")(?![a-z])")


def tone(text: str) -> tuple[float, list[str], list[str]]:
    """Return (score in [-1, 1], positive terms, negative terms)."""
    pos, neg = [], []
    for m in _RX.finditer(text.lower()):
        (pos if _SIGN[m.group(0)] > 0 else neg).append(m.group(0))
    p, n = len(pos), len(neg)
    return (p - n) / (p + n + 1.0), pos, neg


def sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

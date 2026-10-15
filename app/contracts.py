"""Structured output contract shared by every agent.

The synthesis layer only ever consumes `AgentOutput`. Anything that violates the
contract is rejected by `AgentOutput.validate()` and treated as a failed agent,
so a buggy or hallucinating agent can never leak uncited claims downstream.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Any

STATUSES = ("ok", "degraded", "abstained", "failed", "timeout")
KINDS = ("signal", "constraint")
STANCES = ("bullish", "bearish", "neutral")


@dataclass
class Evidence:
    text: str
    metric: str = ""
    value: Any = None
    source_id: str = ""
    stance: str = "neutral"


@dataclass
class Citation:
    source_id: str
    kind: str                # filing | news | market_data | flow | portfolio | history
    title: str
    date: str = ""
    snippet: str = ""
    score: float | None = None
    url: str = ""
    verified: bool | None = None


@dataclass
class AgentOutput:
    agent: str
    symbol: str
    kind: str
    status: str
    label: str
    score: float             # signal: -1 bearish .. +1 bullish.  constraint: -1 unsuitable .. +1 fits
    confidence: float        # 0..1
    summary: str
    evidence: list[Evidence] = field(default_factory=list)
    citations: list[Citation] = field(default_factory=list)
    data: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    started_ms: float = 0.0

    @property
    def usable(self) -> bool:
        return self.status in ("ok", "degraded")

    def validate(self) -> list[str]:
        problems = []
        if self.status not in STATUSES:
            problems.append(f"bad status {self.status!r}")
        if self.kind not in KINDS:
            problems.append(f"bad kind {self.kind!r}")
        for name, v, lo, hi in (("score", self.score, -1, 1), ("confidence", self.confidence, 0, 1)):
            if not isinstance(v, (int, float)) or math.isnan(v) or not lo - 1e-9 <= v <= hi + 1e-9:
                problems.append(f"{name} out of range: {v!r}")
        if self.usable:
            if not self.evidence:
                problems.append("usable output without evidence")
            known = {c.source_id for c in self.citations}
            for e in self.evidence:
                if not e.source_id:
                    problems.append(f"uncited evidence: {e.text[:40]!r}")
                elif e.source_id not in known:
                    problems.append(f"evidence cites unknown source {e.source_id}")
                if e.stance not in STANCES:
                    problems.append(f"bad stance {e.stance!r}")
        return problems

    def to_dict(self) -> dict:
        d = asdict(self)
        d["score"] = round(float(self.score), 3)
        d["confidence"] = round(float(self.confidence), 3)
        d["latency_ms"] = round(self.latency_ms, 1)
        d["started_ms"] = round(self.started_ms, 1)
        return d


def failed_output(agent: str, symbol: str, kind: str, status: str, summary: str,
                  warnings: list[str] | None = None) -> AgentOutput:
    """A well-formed 'no opinion' output used for outages, timeouts and abstentions."""
    return AgentOutput(agent=agent, symbol=symbol, kind=kind, status=status, label="No signal",
                       score=0.0, confidence=0.0, summary=summary, warnings=warnings or [])

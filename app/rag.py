"""Retrieval over the filing corpus: hybrid BM25 + dense (LSA) vector search served by a vector database.

Why hybrid: BM25 gives exact-term precision ("show-cause", "pledge"); the dense vector (cosine, from
`vectorstore.LSAEmbedder`, stored and filtered in Qdrant) adds semantic recall for paraphrases, and a small
finance thesaurus expands queries for the lexical side. Scores are min-max fused. `search()` supports
payload filters (company, sector, document type, date range), recency weighting and three modes
(bm25 / vector / hybrid) so retrieval quality can be measured per mode.
"""
from __future__ import annotations

import datetime as dt
import math
import re
import threading
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .corpus import Document, build_corpus

STOP = set("a an and are as at be by for from has have in is it its of on or that the this to was were will with "
           "which their than into over under per also not no any may".split())
THESAURUS = {
    "risk": ["litigation", "penalty", "probe", "scrutiny", "warning", "show-cause", "pledge", "contingent"],
    "debt": ["leverage", "borrowings", "net debt", "refinancing", "pledge", "obligations"],
    "guidance": ["outlook", "estimates", "expects", "target"],
    "growth": ["revenue", "volumes", "demand", "order book", "arpu"],
    "regulatory": ["sebi", "regulator", "notice", "disclosure", "compliance"],
    "margin": ["ebitda", "margins", "pressure", "costs", "inflation"],
}
AS_OF = dt.date(2026, 10, 15)


def tokens(text: str) -> list[str]:
    out = []
    for w in re.findall(r"[a-z0-9]+", text.lower()):
        if w in STOP or len(w) < 2:
            continue
        for suf in ("ing", "ed", "es", "s"):
            if len(w) > len(suf) + 3 and w.endswith(suf):
                w = w[: -len(suf)]
                break
        out.append(w)
    return out


def expand(query: str) -> list[str]:
    toks = tokens(query)
    extra = []
    for t in list(toks):
        for k, vs in THESAURUS.items():
            if t.startswith(k[:5]):
                for v in vs:
                    extra.extend(tokens(v))
    return toks + extra


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    symbol: str | None
    sectors: list[str]
    heading: str
    text: str
    date: str
    title: str
    doc_type: str
    period: str = ""
    url: str = ""


@dataclass
class Hit:
    chunk: Chunk
    score: float
    bm25: float
    cosine: float


class RagIndex:
    def __init__(self, docs: list[Document] | None = None, k1: float = 1.4, b: float = 0.75, backend: str = "numpy",
                 qdrant_path: str | Path | None = None, dim: int = 96):
        from .vectorstore import LSAEmbedder, NumpyBackend, QdrantBackend, fingerprint
        t0 = time.perf_counter()
        self.docs = {d.doc_id: d for d in (docs if docs is not None else build_corpus())}
        self.chunks: list[Chunk] = []
        for d in self.docs.values():
            for i, (h, t) in enumerate(d.sections):
                self.chunks.append(Chunk(f"{d.doc_id}#{i}", d.doc_id, d.symbol, d.sectors, h, t, d.date.isoformat(),
                                         d.title, d.doc_type, d.period, getattr(d, "url", "")))
        self.k1, self.b = k1, b
        self._tok = [tokens(c.heading + " " + c.text) for c in self.chunks]
        self._tf = [Counter(t) for t in self._tok]
        n = len(self.chunks)
        df = Counter(w for t in self._tok for w in set(t))
        self._idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}
        self._avgdl = sum(len(t) for t in self._tok) / max(n, 1)
        texts = [c.heading + " " + c.text for c in self.chunks]
        from .glossary import GLOSSARY
        self.embedder = LSAEmbedder.fit(texts + GLOSSARY, dim)       # glossary only shapes the latent space
        vecs = self.embedder.embed_many(texts)
        payloads = [{"chunk_id": c.chunk_id, "doc_id": c.doc_id, "symbol": c.symbol or "", "sectors": c.sectors,
                     "doc_type": c.doc_type, "date_ord": dt.date.fromisoformat(c.date).toordinal(), "heading": c.heading}
                    for c in self.chunks]
        self._qcache: dict = {}
        self._qlock = threading.Lock()
        self.cache_hits = self.cache_misses = 0
        self.backend_note = ""
        self.backend = NumpyBackend()
        self.reused = False
        if backend in ("qdrant", "qdrant-memory"):
            if backend == "qdrant-memory":
                qdrant_path = None
            try:
                qb = QdrantBackend(str(qdrant_path) if qdrant_path else None, self.embedder.dim)
                fp_file = Path(qdrant_path) / "fingerprint.txt" if qdrant_path else None
                fp = fingerprint(texts, self.embedder.dim)
                if fp_file and fp_file.exists() and fp_file.read_text() == fp and qb.count() == n:
                    self.reused = True                      # vectors already on disk from a previous run
                    emb_file = Path(qdrant_path) / "embedder.npz"
                    if emb_file.exists():
                        self.embedder = LSAEmbedder.load(emb_file)
                else:
                    qb.upsert(vecs, payloads)
                    if fp_file:
                        fp_file.write_text(fp)
                        self.embedder.save(Path(qdrant_path) / "embedder.npz")
                self.backend = qb
            except Exception as e:                           # noqa: BLE001 - e.g. folder locked by another process
                self.backend_note = f"Qdrant unavailable ({type(e).__name__}); using in-memory numpy index"
                self.backend = NumpyBackend()
        if isinstance(self.backend, NumpyBackend):
            self.backend.upsert(vecs, payloads)
        self.index_ms = (time.perf_counter() - t0) * 1000

    def stats(self) -> dict:
        types = Counter(c.doc_type for c in self.chunks)
        return {"documents": len(self.docs), "passages": len(self.chunks), "embedding_dim": self.embedder.dim,
                "vocab": len(self.embedder.vocab), "backend": self.backend.name, "persisted": self.backend.persisted,
                "reused_from_disk": self.reused, "index_ms": round(self.index_ms, 1), "by_type": dict(types),
                "companies": len({c.symbol for c in self.chunks if c.symbol}), "note": self.backend_note,
                "date_range": [min(c.date for c in self.chunks), max(c.date for c in self.chunks)]}

    def _bm25(self, qtoks: list[str], i: int) -> float:
        tf, dl, s = self._tf[i], len(self._tok[i]), 0.0
        for w in set(qtoks):
            f = tf.get(w, 0)
            if f:
                s += self._idf.get(w, 0) * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self._avgdl))
        return s

    def _candidates(self, flt: dict) -> list[int]:
        from .vectorstore import payload_match
        return [i for i, p in enumerate(self.backend.payloads if hasattr(self.backend, "payloads") else self._payloads()) if payload_match(p, flt)]

    def _payloads(self):
        if not hasattr(self, "_pl"):
            self._pl = [{"symbol": c.symbol or "", "sectors": c.sectors, "doc_type": c.doc_type,
                         "date_ord": dt.date.fromisoformat(c.date).toordinal(), "doc_id": c.doc_id} for c in self.chunks]
        return self._pl

    # ---- public API
    def search(self, query: str, symbol: str | None = None, sector: str | None = None, k: int = 4,
               blocked_docs: set[str] | None = None, doc_types: list[str] | None = None, date_from: dt.date | None = None,
               mode: str = "hybrid", recency: bool | float = False) -> list[Hit]:
        """Cached: the index is immutable, so identical queries return identical hits (results are never mutated)."""
        key = (query, symbol, sector, k, tuple(sorted(blocked_docs or ())), tuple(doc_types or ()), date_from, mode, recency)
        with self._qlock:
            if key in self._qcache:
                self.cache_hits += 1
                return self._qcache[key]
        hits = self._search(query, symbol, sector, k, blocked_docs, doc_types, date_from, mode, recency)
        with self._qlock:
            self._qcache[key] = hits
            self.cache_misses += 1
        return hits

    def _search(self, query, symbol, sector, k, blocked_docs, doc_types, date_from, mode, recency) -> list[Hit]:
        flt = {"symbol": symbol, "sector": sector, "doc_types": doc_types, "blocked_docs": list(blocked_docs or []),
               "date_from": date_from.toordinal() if date_from else None}
        cand = self._candidates(flt)
        if not cand:
            return []
        pos = {i: j for j, i in enumerate(cand)}
        bm = np.zeros(len(cand))
        cos = np.zeros(len(cand))
        if mode in ("bm25", "hybrid"):
            qt = expand(query)
            bm = np.array([self._bm25(qt, i) for i in cand])
        if mode in ("vector", "hybrid"):
            for i, sc in self.backend.query(self.embedder.embed(query, expand_query=True), flt, len(cand)):
                if i in pos:
                    cos[pos[i]] = max(sc, 0.0)
        bmn = bm / bm.max() if bm.max() > 0 else bm
        cosn = cos / cos.max() if cos.max() > 0 else cos
        score = {"bm25": bmn, "vector": cosn, "hybrid": 0.55 * bmn + 0.45 * cosn}[mode]
        if recency:
            age = np.array([(AS_OF - dt.date.fromisoformat(self.chunks[i].date)).days for i in cand])
            tau = 270.0 if recency is True else float(recency)
            score = score * (0.3 + 0.7 * np.exp(-age / tau))
        order = np.argsort(-score)[:k]
        return [Hit(self.chunks[cand[j]], float(score[j]), float(bmn[j]), float(cosn[j])) for j in order if score[j] > 0]

    def doc(self, doc_id: str) -> Document | None:
        return self.docs.get(doc_id)

    def verify_quote(self, doc_id: str, quote: str) -> bool:
        """Attribution check: the quoted text must literally occur in the cited source document."""
        d = self.docs.get(doc_id)
        return bool(d and quote.strip() and quote.strip() in d.full_text)

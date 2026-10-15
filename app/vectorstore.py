"""Dense semantic index: LSA embeddings + a vector database backend.

* `LSAEmbedder`  - TF-IDF -> truncated SVD (latent semantic analysis). Words that co-occur across filings
  ("show-cause", "notice", "penalty", "pledge") land close together, so a query can match a passage that
  shares *no* literal words with it. It is trained on the corpus itself, needs no network or model download,
  and is deterministic. A sentence-transformer can replace it behind the same `embed()` call.
* `QdrantBackend` - embedded, persistent Qdrant (local mode) with native payload filtering
  (company / sector / document type / date range). Falls back to `NumpyBackend` if Qdrant is unavailable
  (e.g. the storage folder is locked by another process), and the UI reports which backend served the query.
"""
from __future__ import annotations

import atexit
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from .rag import expand, tokens


class LSAEmbedder:
    def __init__(self, vocab: dict[str, int], idf: np.ndarray, vt: np.ndarray):
        self.vocab, self.idf, self.vt = vocab, idf, vt
        self.dim = vt.shape[0]

    @classmethod
    def fit(cls, texts: list[str], dim: int = 96) -> "LSAEmbedder":
        toks = [tokens(t) for t in texts]
        df = Counter(w for t in toks for w in set(t))
        words = sorted(w for w, c in df.items() if c >= 2)
        vocab = {w: i for i, w in enumerate(words)}
        n = len(texts)
        idf = np.array([math.log(1 + n / df[w]) for w in words])
        x = np.zeros((n, len(words)))
        for i, t in enumerate(toks):
            for w, c in Counter(t).items():
                if w in vocab:
                    x[i, vocab[w]] = (1 + math.log(c)) * idf[vocab[w]]
        x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9)
        _, _, vt = np.linalg.svd(x, full_matrices=False)
        return cls(vocab, idf, vt[: min(dim, vt.shape[0])])

    def embed(self, text: str, expand_query: bool = False) -> np.ndarray:
        x = np.zeros(len(self.vocab))
        for w, c in Counter(expand(text) if expand_query else tokens(text)).items():
            j = self.vocab.get(w)
            if j is not None:
                x[j] = (1 + math.log(c)) * self.idf[j]
        v = self.vt @ x
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def embed_many(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim))
        return np.vstack([self.embed(t) for t in texts])

    def save(self, path: Path):
        np.savez(path, idf=self.idf, vt=self.vt, vocab=np.array(json.dumps(self.vocab)))

    @classmethod
    def load(cls, path: Path) -> "LSAEmbedder":
        z = np.load(path, allow_pickle=False)
        return cls(json.loads(str(z["vocab"])), z["idf"], z["vt"])


def fingerprint(texts: list[str], dim: int) -> str:
    h = hashlib.sha1(str(dim).encode())
    for t in texts:
        h.update(t.encode())
    return h.hexdigest()


def payload_match(p: dict, flt: dict) -> bool:
    """Same semantics as the Qdrant filter, for the in-memory backend."""
    sym = flt.get("symbol")
    if sym is not None and not (p["symbol"] == sym or (p["symbol"] == "" and flt.get("sector") in p["sectors"])):
        return False
    if flt.get("doc_types") and p["doc_type"] not in flt["doc_types"]:
        return False
    if flt.get("date_from") is not None and p["date_ord"] < flt["date_from"]:
        return False
    return not (flt.get("blocked_docs") and p["doc_id"] in flt["blocked_docs"])


class NumpyBackend:
    name = "numpy (in-memory)"
    persisted = False

    def __init__(self):
        self.vecs: np.ndarray | None = None
        self.payloads: list[dict] = []

    def upsert(self, vecs: np.ndarray, payloads: list[dict]):
        self.vecs, self.payloads = vecs, payloads

    def count(self) -> int:
        return len(self.payloads)

    def query(self, vec: np.ndarray, flt: dict, limit: int) -> list[tuple[int, float]]:
        idx = [i for i, p in enumerate(self.payloads) if payload_match(p, flt)]
        if not idx:
            return []
        sc = self.vecs[idx] @ vec
        order = np.argsort(-sc)[:limit]
        return [(idx[j], float(sc[j])) for j in order]


class QdrantBackend:
    COLLECTION = "filings"
    persisted = True

    def __init__(self, path: str | None, dim: int):
        from qdrant_client import QdrantClient, models
        self.models = models
        if path:
            Path(path).mkdir(parents=True, exist_ok=True)
        self.client = QdrantClient(path=path) if path else QdrantClient(":memory:")
        atexit.register(self.close)
        self.name = f"Qdrant {'local, on disk' if path else 'in-memory'}"
        self.persisted = bool(path)
        self.dim = dim

    def close(self):
        try:
            self.client.close()
        except Exception:                      # noqa: BLE001
            pass

    def count(self) -> int:
        if not self.client.collection_exists(self.COLLECTION):
            return 0
        return self.client.count(self.COLLECTION).count

    def upsert(self, vecs: np.ndarray, payloads: list[dict]):
        m = self.models
        if self.client.collection_exists(self.COLLECTION):
            self.client.delete_collection(self.COLLECTION)
        self.client.create_collection(self.COLLECTION, vectors_config=m.VectorParams(size=self.dim, distance=m.Distance.COSINE))
        self.client.upload_collection(self.COLLECTION, vectors=vecs, payload=payloads, ids=list(range(len(payloads))),
                                      batch_size=256)

    def query(self, vec: np.ndarray, flt: dict, limit: int) -> list[tuple[int, float]]:
        m = self.models
        must, sym, sec = [], flt.get("symbol"), flt.get("sector")
        if sym is not None:
            own = m.FieldCondition(key="symbol", match=m.MatchValue(value=sym))
            should = [own]
            if sec is not None:
                should.append(m.Filter(must=[m.FieldCondition(key="symbol", match=m.MatchValue(value="")),
                                             m.FieldCondition(key="sectors", match=m.MatchValue(value=sec))]))
            must.append(m.Filter(should=should))
        if flt.get("doc_types"):
            must.append(m.FieldCondition(key="doc_type", match=m.MatchAny(any=list(flt["doc_types"]))))
        if flt.get("date_from") is not None:
            must.append(m.FieldCondition(key="date_ord", range=m.Range(gte=flt["date_from"])))
        if flt.get("blocked_docs"):
            must.append(m.Filter(must_not=[m.FieldCondition(key="doc_id", match=m.MatchAny(any=list(flt["blocked_docs"])))]))
        res = self.client.query_points(self.COLLECTION, query=vec.tolist(), query_filter=m.Filter(must=must) if must else None,
                                       limit=limit, with_payload=False)
        return [(int(p.id), float(p.score)) for p in res.points]

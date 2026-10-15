"""Retrieval quality: recall@3 and MRR for BM25-only, vector-only and hybrid search.

Queries are searched over the WHOLE corpus (no company filter), so a hit must beat ~500 other passages,
many of them near-duplicates from other quarters. Two query sets: 'lexical' queries reuse words from the
target passage; 'paraphrase' queries deliberately share (almost) none, which is where dense retrieval is
meant to help. Labels were written by hand from the source passages.
"""
from __future__ import annotations

LEXICAL = [
    ("promoter pledge increased show-cause notice", {"VIKRAMINFRA-LODR-R30#0", "VIKRAMINFRA-LODR-R30#1"}),
    ("warning letter US drug regulator Unit-4 plant", {"AROGYAPHARMA-ER-Q2FY27#1", "AROGYAPHARMA-LODR-R30#0"}),
    ("voluntary recall faulty airbag component", {"GATIMOTORS-LODR-R30#0"}),
    ("green hydrogen order book", {"BHARATOIL-ER-Q2FY27#1"}),
    ("net interest margin compressed higher deposit costs", {"NAVBANK-ER-Q2FY27#1"}),
    ("qualified opinion going concern subsidiary auditor", {"VIKRAMINFRA-AR-FY26#0"}),
    ("record ARPU net subscriber additions", {"SANCHARNET-ER-Q2FY27#0"}),
]
PARAPHRASE = [
    ("watchdog questioned undisclosed dealings with affiliates", {"VIKRAMINFRA-LODR-R30#0"}),
    ("lenders worried about heavy borrowing and rising share collateral", {"VIKRAMINFRA-ER-Q2FY27#1", "VIKRAMINFRA-LODR-R30#1"}),
    ("American inspectors criticised factory data practices", {"AROGYAPHARMA-LODR-R30#0", "AROGYAPHARMA-ER-Q2FY27#1"}),
    ("carmaker pulls back vehicles over defective safety part", {"GATIMOTORS-LODR-R30#0"}),
    ("rating firm cut credit view because of soft demand", {"GATIMOTORS-LODR-R30#1"}),
    ("accountants doubt the project company can keep operating", {"VIKRAMINFRA-AR-FY26#0"}),
    ("bank profitability squeezed by deposit funding costs", {"NAVBANK-ER-Q2FY27#1"}),
]


def _score(rag, queries, mode, k=3):
    rr, rec = [], []
    for q, gold in queries:
        hits = rag.search(q, k=10, mode=mode)
        ids = [h.chunk.chunk_id for h in hits]
        rank = next((i + 1 for i, c in enumerate(ids) if c in gold), None)
        rr.append(1 / rank if rank else 0.0)
        rec.append(1.0 if rank and rank <= k else 0.0)
    return {"recall@3": round(sum(rec) / len(rec), 3), "mrr": round(sum(rr) / len(rr), 3)}


def evaluate(rag) -> dict:
    out = {"n_queries": len(LEXICAL) + len(PARAPHRASE), "corpus_passages": len(rag.chunks), "sets": {}}
    for name, qs in (("lexical", LEXICAL), ("paraphrase", PARAPHRASE)):
        out["sets"][name] = {"n": len(qs), **{m: _score(rag, qs, m) for m in ("bm25", "vector", "hybrid")}}
    return out

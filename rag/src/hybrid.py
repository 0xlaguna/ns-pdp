"""Reciprocal rank fusion of the dense and BM25 rankings.

score(d) = sum 1/(k + rank(d)) over rankings that include d.
Ranks start at 1. A document missing from a list does not get that list's term.
Ties break by score descending, then by id.

sentence-transformers ReciprocalRankFusionEvaluator uses the same
1/(rrf_k + rank) with rrf_k=60 and one-based ranks, and it also skips a
list when the document is absent. Its sort is score-only. This function
breaks remaining ties by id, which that evaluator does not.
"""

from __future__ import annotations


def rrf(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def hybrid_search(dense, lexical, query: str, k_each: int = 20, k: int = 60) -> list[dict]:
    dense_hits = dense.search(query, k=k_each)
    lexical_hits = lexical.search(query, k=k_each)
    dense_ids = [str(hit["id"]) for hit in dense_hits]
    lexical_ids = [str(hit["id"]) for hit in lexical_hits]
    dense_rank = {doc_id: rank for rank, doc_id in enumerate(dense_ids, start=1)}
    lexical_rank = {doc_id: rank for rank, doc_id in enumerate(lexical_ids, start=1)}
    texts: dict[str, str] = {}
    for hit in [*dense_hits, *lexical_hits]:
        texts.setdefault(str(hit["id"]), str(hit["text"]))
    hits: list[dict] = []
    for doc_id, score in rrf([dense_ids, lexical_ids], k=k):
        hits.append(
            {
                "id": doc_id,
                "text": texts[doc_id],
                "score": score,
                "dense_rank": dense_rank.get(doc_id),
                "lexical_rank": lexical_rank.get(doc_id),
            }
        )
    return hits

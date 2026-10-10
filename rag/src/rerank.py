"""Cross-encoder second stage over a fused shortlist.

``CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")`` is loaded once.
That name is an alias of the canonical repo
``cross-encoder/ms-marco-MiniLM-L6-v2`` (same snapshot). The model card's
sample calls ``predict`` and prints raw logits. sentence-transformers 6.1
would use ``Sigmoid`` for ``num_labels=1`` when the checkpoint does not say
otherwise. This checkpoint's ``config.json`` sets
``sbert_ce_default_activation_function`` to
``torch.nn.modules.linear.Identity``, the pre-v4 key, so ``predict`` returns
logits. That matches the model card sample, not the retrieve-and-rerank
page's "score between 0 and 1". This module does not override the activation.
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import CrossEncoder

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_model: CrossEncoder | None = None


def get_model() -> CrossEncoder:
    global _model
    if _model is None:
        _model = CrossEncoder(MODEL_NAME)
    return _model


def _fusion_score(hit: dict) -> float:
    if "fusion_score" in hit:
        return float(hit["fusion_score"])
    return float(hit["score"])


def rerank(query: str, hits: list[dict], top_n: int = 5) -> list[dict]:
    top_n = int(top_n)
    if top_n <= 0 or not hits:
        return []
    pairs = [(query, hit["text"]) for hit in hits]
    scores = np.asarray(get_model().predict(pairs, show_progress_bar=False), dtype=np.float64).reshape(-1)
    rows: list[dict] = []
    for hit, score in zip(hits, scores):
        rows.append(
            {
                "id": str(hit["id"]),
                "text": str(hit["text"]),
                "rerank_score": float(score),
                "fusion_score": _fusion_score(hit),
            }
        )
    # Stable: equal cross-encoder scores keep the fused order.
    rows.sort(key=lambda row: row["rerank_score"], reverse=True)
    return rows[:top_n]

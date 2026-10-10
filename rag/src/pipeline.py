"""Hybrid retrieval then cross-encoder rerank.

``search`` always fuses the dense and BM25 top 20 (``k_each=20``), then
returns ``k`` reranked rows. Indexes and the cross-encoder load once.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dense import DenseIndex
from hybrid import hybrid_search
from lexical import LexicalIndex, load_chunks
from rerank import get_model, rerank

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Pipeline:
    dense: DenseIndex
    lexical: LexicalIndex
    chunks: int


_pipeline: Pipeline | None = None


def load() -> Pipeline:
    global _pipeline
    if _pipeline is not None:
        return _pipeline
    rows = load_chunks(ROOT / "data" / "chunks.jsonl")
    lexical = LexicalIndex()
    lexical.fit(rows)
    dense = DenseIndex.load(ROOT / "data" / "index")
    dense.load_model()
    get_model()
    _pipeline = Pipeline(dense=dense, lexical=lexical, chunks=len(rows))
    return _pipeline


def chunk_count() -> int:
    return load().chunks


def search(query: str, k: int = 5) -> list[dict]:
    state = load()
    fused = hybrid_search(state.dense, state.lexical, query, k_each=20)
    return rerank(query, fused, top_n=k)

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
_stamp: tuple[float, float] | None = None


def _stamp_now() -> tuple[float, float]:
    index = ROOT / "data" / "index"
    chunks = ROOT / "data" / "chunks.jsonl"
    index_mtime = index.stat().st_mtime if index.exists() else 0.0
    chunks_mtime = chunks.stat().st_mtime if chunks.exists() else 0.0
    return (index_mtime, chunks_mtime)


def load() -> Pipeline:
    global _pipeline, _stamp
    stamp = _stamp_now()
    if _pipeline is not None and stamp == _stamp:
        return _pipeline
    rows = load_chunks(ROOT / "data" / "chunks.jsonl")
    lexical = LexicalIndex()
    lexical.fit(rows)
    old_model = _pipeline.dense.model if _pipeline is not None else None
    dense = DenseIndex.load(ROOT / "data" / "index")
    if old_model is not None:
        dense.model = old_model
    else:
        dense.load_model()
    get_model()
    _pipeline = Pipeline(dense=dense, lexical=lexical, chunks=len(rows))
    _stamp = stamp
    return _pipeline


def chunk_count() -> int:
    return load().chunks


def search(query: str, k: int = 5) -> list[dict]:
    state = load()
    fused = hybrid_search(state.dense, state.lexical, query, k_each=20)
    return rerank(query, fused, top_n=k)

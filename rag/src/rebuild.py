"""Rebuild chunks and indexes, then swap the index into place.

A path that contains ``..`` fails before any file is written. The live
``chunks.jsonl`` and ``data/index`` change only after the next index has
been saved.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from chunk import load_corpus
from dense import DenseIndex
from lexical import LexicalIndex


def assert_raw_path(root: Path, path: str) -> Path:
    if ".." in Path(path).parts:
        raise ValueError("path escapes data/raw")
    raw = (root / "data" / "raw").resolve()
    candidate = (root / path).resolve()
    if not candidate.is_relative_to(raw):
        raise ValueError("path escapes data/raw")
    if not candidate.is_file():
        raise ValueError("path is not a file under data/raw")
    return candidate


def swap_index(tmp: Path, live: Path) -> None:
    if live.exists():
        bak = live.with_name("index.bak")
        if bak.exists():
            shutil.rmtree(bak)
        live.replace(bak)
    tmp.replace(live)


def rebuild_from_raw(root: Path, path: str) -> dict:
    assert_raw_path(root, path)
    rows = load_corpus(root / "data" / "raw")
    data = root / "data"
    chunks_next = data / "chunks.jsonl.next"
    index_next = data / "index.next"
    if index_next.exists():
        shutil.rmtree(index_next)
    with chunks_next.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    dense = DenseIndex()
    dense.fit(rows)
    dense.save(index_next)
    lexical = LexicalIndex()
    lexical.fit(rows)
    (index_next / "lexical.json").write_text(
        json.dumps(
            {"ids": lexical.ids, "texts": lexical.texts},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    chunks = data / "chunks.jsonl"
    chunks_bak = data / "chunks.jsonl.bak"
    if chunks_bak.exists():
        chunks_bak.unlink()
    if chunks.exists():
        chunks.replace(chunks_bak)
    chunks_next.replace(chunks)
    try:
        swap_index(index_next, data / "index")
    except Exception:
        if chunks_bak.exists():
            if chunks.exists():
                chunks.unlink()
            chunks_bak.replace(chunks)
        raise
    return {"chunks": len(rows)}

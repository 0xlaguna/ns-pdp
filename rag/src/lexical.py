"""BM25 index over the same chunk ids as the dense index.

Pattern kept: ``[a-z0-9_./+-]+`` on lowercased text, so ``mlx_lm.lora``,
``train.jsonl``, ``/search``, and ``2-4`` each stay one token.

rank_bm25 does not tokenize. ``BM25Okapi`` still takes a list of token
lists and ``get_scores`` still returns one score per document, as in the
library README. The index is rebuilt from ``chunks.jsonl`` in memory.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_./+-]+", text.lower())


def load_chunks(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class LexicalIndex:
    def __init__(self) -> None:
        self.ids: list[str] = []
        self.texts: list[str] = []
        self._bm25: BM25Okapi | None = None

    def fit(self, rows: list[dict]) -> None:
        self.ids = [str(row["id"]) for row in rows]
        self.texts = [str(row["text"]) for row in rows]
        if not rows:
            self._bm25 = None
            return
        tokenized = [tokenize(text) for text in self.texts]
        self._bm25 = BM25Okapi(tokenized)

    def search(self, query: str, k: int = 5) -> list[dict]:
        if self._bm25 is None or not self.ids:
            return []
        k = max(0, min(int(k), len(self.ids)))
        if k == 0:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        ranked = [
            (float(scores[index]), self.ids[index], index)
            for index in range(len(self.ids))
            if float(scores[index]) > 0.0
        ]
        ranked.sort(key=lambda item: (-item[0], item[1]))
        return [
            {
                "id": chunk_id,
                "text": self.texts[index],
                "score": score,
            }
            for score, chunk_id, index in ranked[:k]
        ]

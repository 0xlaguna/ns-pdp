"""Local inner-product index over Fieldguide chunk embeddings.

BAAI/bge-small-en-v1.5 model card (table row) still lists this query
instruction. The v1.5 FAQ says retrieval works without an instruction, with a
small drop, and recommends the instruction for short queries against longer
passages. Passages are encoded without it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# Torch before faiss. On macOS both ship libomp; importing faiss first aborts
# with OMP Error #15. sentence-transformers pulls torch in anyway.
import torch  # noqa: F401

# "Represent this sentence for searching relevant passages: " — bge-small-en-v1.5 card.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
MODEL_NAME = "BAAI/bge-small-en-v1.5"


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.ascontiguousarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    np.maximum(norms, 1e-12, out=norms)
    return matrix / norms


class DenseIndex:
    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self.model_name = model_name
        self.model = None
        self.ids: list[str] = []
        self.texts: list[str] = []
        self.vectors: np.ndarray | None = None
        self.index = None
        self.backend = "faiss"

    def load_model(self):
        if self.model is None:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(self.model_name)
        return self.model

    def fit(self, rows: list[dict]) -> None:
        self.load_model()
        self.ids = [str(row["id"]) for row in rows]
        self.texts = [str(row["text"]) for row in rows]
        encoded = self.model.encode(
            self.texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        self.vectors = _l2_normalize(np.asarray(encoded))
        self._build_index(self.vectors)

    def _build_index(self, vectors: np.ndarray) -> None:
        try:
            import faiss
        except ImportError:
            self.backend = "numpy"
            self.index = None
            return
        index = faiss.IndexFlatIP(int(vectors.shape[1]))
        index.add(vectors)
        self.index = index
        self.backend = "faiss"

    def search(self, query: str, k: int = 5) -> list[dict]:
        if self.vectors is None or not self.ids:
            return []
        self.load_model()
        k = max(0, min(int(k), len(self.ids)))
        if k == 0:
            return []
        encoded = self.model.encode(
            [QUERY_PREFIX + query],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        query_vector = _l2_normalize(np.asarray(encoded, dtype=np.float32))
        if self.backend == "faiss" and self.index is not None:
            scores, positions = self.index.search(query_vector.astype("float32"), k)
            score_row = scores[0]
            position_row = positions[0]
        else:
            similarities = self.vectors @ query_vector[0]
            position_row = np.argsort(-similarities)[:k]
            score_row = similarities[position_row]
        hits: list[dict] = []
        for score, position in zip(score_row, position_row):
            position = int(position)
            if position < 0:
                continue
            hits.append(
                {
                    "id": self.ids[position],
                    "text": self.texts[position],
                    "score": float(score),
                }
            )
        return hits

    def save(self, directory: str | Path) -> None:
        if self.vectors is None:
            raise RuntimeError("fit the index before saving it")
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "vectors.npy", self.vectors)
        records = [{"id": chunk_id, "text": text} for chunk_id, text in zip(self.ids, self.texts)]
        (out / "ids.json").write_text(
            json.dumps(records, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (out / "backend.txt").write_text(self.backend + "\n", encoding="utf-8")
        if self.backend == "faiss" and self.index is not None:
            import faiss

            faiss.write_index(self.index, str(out / "index.faiss"))

    @classmethod
    def load(cls, directory: str | Path, model_name: str = MODEL_NAME) -> DenseIndex:
        src = Path(directory)
        index = cls(model_name=model_name)
        records = json.loads((src / "ids.json").read_text(encoding="utf-8"))
        index.ids = [str(row["id"]) for row in records]
        index.texts = [str(row["text"]) for row in records]
        index.vectors = np.load(src / "vectors.npy")
        backend_path = src / "backend.txt"
        recorded = backend_path.read_text(encoding="utf-8").strip() if backend_path.exists() else "numpy"
        faiss_path = src / "index.faiss"
        if recorded == "faiss" and faiss_path.exists():
            try:
                import faiss
            except ImportError:
                index.backend = "numpy"
                index.index = None
            else:
                index.index = faiss.read_index(str(faiss_path))
                index.backend = "faiss"
        else:
            index.backend = "numpy"
            index.index = None
        return index

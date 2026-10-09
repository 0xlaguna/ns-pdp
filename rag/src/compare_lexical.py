"""Print top-3 lexical and dense ids for the five comparison queries."""

from __future__ import annotations

from pathlib import Path

from dense import DenseIndex
from lexical import load_chunks, tokenize, LexicalIndex

QUERIES = [
    "What is a cross-encoder?",
    "mlx_lm.lora",
    "POST /search",
    "train.jsonl valid.jsonl test.jsonl",
    "2-4 sentences",
]


def _snippet(text: str, limit: int = 90) -> str:
    flat = " ".join(text.split())
    if len(flat) <= limit:
        return flat
    return flat[: limit - 3] + "..."


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    rows = load_chunks(root / "data" / "chunks.jsonl")
    lexical = LexicalIndex()
    lexical.fit(rows)
    dense = DenseIndex.load(root / "data" / "index")

    kept = tokenize("mlx_lm.lora")
    print("tokenizer: [a-z0-9_./+-]+")
    print(f"token_check: mlx_lm.lora -> {kept}")
    print(f"chunks: {len(rows)}")
    print(f"dense_backend: {dense.backend}")

    for query in QUERIES:
        lexical_hits = lexical.search(query, k=3)
        dense_hits = dense.search(query, k=3)
        print(f"query: {query}")
        print("lexical:")
        for hit in lexical_hits:
            print(f"  {hit['id']}  {hit['score']:.4f}  {_snippet(hit['text'])}")
        if not lexical_hits:
            print("  (none)")
        print("dense:")
        for hit in dense_hits:
            print(f"  {hit['id']}  {hit['score']:.4f}  {_snippet(hit['text'])}")
        if not dense_hits:
            print("  (none)")
        lexical_top = lexical_hits[0]["id"] if lexical_hits else "(none)"
        dense_top = dense_hits[0]["id"] if dense_hits else "(none)"
        print(f"top1: lexical={lexical_top} dense={dense_top}")


if __name__ == "__main__":
    main()

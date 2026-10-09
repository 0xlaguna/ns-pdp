"""Write eval/fusion.md for the five queries from the BM25 comparison."""

from __future__ import annotations

from pathlib import Path

from compare_lexical import QUERIES
from dense import DenseIndex
from hybrid import hybrid_search
from lexical import LexicalIndex, load_chunks


def _ids(hits: list[dict], n: int = 5) -> list[str]:
    return [str(hit["id"]) for hit in hits[:n]]


def _join(ids: list[str]) -> str:
    return ", ".join(ids) if ids else "(none)"


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    rows = load_chunks(root / "data" / "chunks.jsonl")
    lexical = LexicalIndex()
    lexical.fit(rows)
    dense = DenseIndex.load(root / "data" / "index")

    sections: list[str] = [
        "# Reciprocal rank fusion",
        "",
        "RRF constant k=60. Each parent list is that index's top 20.",
        "A fused top-5 differs from both parents when its id order matches neither parent top-5.",
        "",
    ]
    differed: list[tuple[str, list[str]]] = []
    for query in QUERIES:
        dense_hits = dense.search(query, k=20)
        lexical_hits = lexical.search(query, k=20)
        fused_hits = hybrid_search(dense, lexical, query, k_each=20, k=60)
        dense_top = _ids(dense_hits)
        lexical_top = _ids(lexical_hits)
        fused_top = _ids(fused_hits)
        differs = fused_top != dense_top and fused_top != lexical_top
        print(f"query: {query}")
        print(f"dense_top5: {_join(dense_top)}")
        print(f"lexical_top5: {_join(lexical_top)}")
        print(f"fused_top5: {_join(fused_top)}")
        print(f"differs_from_both: {str(differs).lower()}")
        sections.append(f"## {query}")
        sections.append("")
        sections.append(f"- dense top-5: {_join(dense_top)}")
        sections.append(f"- lexical top-5: {_join(lexical_top)}")
        sections.append(f"- fused top-5: {_join(fused_top)}")
        sections.append(f"- differs from both parents: {str(differs).lower()}")
        sections.append("")
        sections.append("| fused id | score | dense_rank | lexical_rank |")
        sections.append("| --- | --- | --- | --- |")
        for hit in fused_hits[:5]:
            dense_rank = hit["dense_rank"] if hit["dense_rank"] is not None else "absent"
            lexical_rank = hit["lexical_rank"] if hit["lexical_rank"] is not None else "absent"
            sections.append(
                f"| {hit['id']} | {hit['score']:.6f} | {dense_rank} | {lexical_rank} |"
            )
        sections.append("")
        if differs:
            differed.append((query, fused_top))

    if not differed:
        raise SystemExit("no fused top-5 differed from both parents")

    query, fused_top = differed[0]
    summary = f"Fusion differed on query: {query}. Top ids: {_join(fused_top)}."
    sections.append("## Reported difference")
    sections.append("")
    sections.append(summary)
    sections.append("")
    text = "\n".join(sections)
    out = root / "eval" / "fusion.md"
    out.write_text(text, encoding="utf-8")
    print(summary)
    print(f"wrote {out.relative_to(root)}")


if __name__ == "__main__":
    main()

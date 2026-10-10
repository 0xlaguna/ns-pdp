"""Score dense, BM25, hybrid, and hybrid-plus-rerank on eval/gold.jsonl.

Recall@5 is 1 when any gold id is in the top 5. MRR is the mean of 1/rank of
the first gold id in that system's returned list, or 0 when the id is absent.
Dense and BM25 return 5 hits. Hybrid fuses with k_each=20 and keeps the full
fused list. Rerank returns 5.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hybrid import hybrid_search
from lexical import load_chunks
from pipeline import load, search


def recall_at_k(hits: list[dict], relevant: set[str], k: int = 5) -> float:
    top = {h["id"] for h in hits[:k]}
    return 1.0 if relevant & top else 0.0


def reciprocal_rank(hits: list[dict], relevant: set[str]) -> float:
    for rank, hit in enumerate(hits, start=1):
        if hit["id"] in relevant:
            return 1.0 / rank
    return 0.0


def _load_gold(path: Path, known_ids: set[str]) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            relevant = [str(chunk_id) for chunk_id in row["relevant_ids"]]
            missing = [chunk_id for chunk_id in relevant if chunk_id not in known_ids]
            if missing:
                raise SystemExit(f"gold line {line_number} has unknown ids: {missing}")
            rows.append({"query": str(row["query"]), "relevant_ids": relevant})
    return rows


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 6)


def main() -> None:
    rows = load_chunks(ROOT / "data" / "chunks.jsonl")
    known_ids = {str(row["id"]) for row in rows}
    gold = _load_gold(ROOT / "eval" / "gold.jsonl", known_ids)
    state = load()

    names = ("dense", "bm25", "hybrid", "hybrid_rerank")
    recalls = {name: [] for name in names}
    reciprocal = {name: [] for name in names}

    for row in gold:
        query = row["query"]
        relevant = set(row["relevant_ids"])
        dense_hits = state.dense.search(query, k=5)
        bm25_hits = state.lexical.search(query, k=5)
        hybrid_hits = hybrid_search(state.dense, state.lexical, query, k_each=20)
        rerank_hits = search(query, k=5)
        lists = {
            "dense": dense_hits,
            "bm25": bm25_hits,
            "hybrid": hybrid_hits,
            "hybrid_rerank": rerank_hits,
        }
        print(f"query: {query}")
        print(f"gold: {', '.join(row['relevant_ids'])}")
        for name, hits in lists.items():
            recall = recall_at_k(hits, relevant, k=5)
            rr = reciprocal_rank(hits, relevant)
            recalls[name].append(recall)
            reciprocal[name].append(rr)
            top_ids = ", ".join(hit["id"] for hit in hits[:5]) or "(none)"
            print(f"  {name}: recall_at_5={recall} rr={rr} top5={top_ids}")

    systems = {
        name: {
            "recall_at_5": _mean(recalls[name]),
            "mrr": _mean(reciprocal[name]),
            "n": len(gold),
        }
        for name in names
    }
    try:
        command = str(Path(sys.executable).resolve().relative_to(ROOT))
    except ValueError:
        command = sys.executable
    payload = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "command": f"{command} eval/run_eval.py",
        "n": len(gold),
        "systems": systems,
    }
    out = ROOT / "eval" / "results.json"
    rendered = json.dumps(payload, indent=2) + "\n"
    out.write_text(rendered, encoding="utf-8")
    print(f"n={len(gold)}")
    for name in names:
        metrics = systems[name]
        print(
            f"{name} recall_at_5={metrics['recall_at_5']} mrr={metrics['mrr']} n={metrics['n']}"
        )
    print(rendered, end="")
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

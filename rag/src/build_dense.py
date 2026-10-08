"""Embed chunks.jsonl and write eval/dense_baseline.json."""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

from dense import MODEL_NAME, QUERY_PREFIX, DenseIndex


def load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    rows = load_rows(root / "data" / "chunks.jsonl")
    index = DenseIndex()
    print(f"loading {index.model_name}", flush=True)
    index.load_model()

    started = time.perf_counter()
    index.fit(rows)
    embed_seconds = time.perf_counter() - started

    index_dir = root / "data" / "index"
    index.save(index_dir)

    query = "What is a cross-encoder?"
    samples_ms: list[float] = []
    last_hits: list[dict] = []
    for _ in range(20):
        tick = time.perf_counter()
        last_hits = index.search(query, k=5)
        samples_ms.append((time.perf_counter() - tick) * 1000.0)
    query_ms_p50 = float(statistics.median(samples_ms))

    payload = {
        "chunks": len(rows),
        "dim": int(index.vectors.shape[1]),
        "embed_seconds": round(embed_seconds, 4),
        "query_ms_p50": round(query_ms_p50, 4),
        "model": MODEL_NAME,
        "backend": index.backend,
    }
    eval_path = root / "eval" / "dense_baseline.json"
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, indent=2) + "\n"
    eval_path.write_text(rendered, encoding="utf-8")

    print(f"embed_seconds_raw={embed_seconds:.6f}")
    print("query_ms_samples=" + ",".join(f"{sample:.4f}" for sample in samples_ms))
    print(f"query_ms_median={query_ms_p50:.6f}")
    print(f"backend_file={(index_dir / 'backend.txt').read_text(encoding='utf-8').strip()}")
    print(rendered, end="")
    print("timed_query_top")
    for hit in last_hits:
        print(f"  {hit['score']:.4f} {hit['id']}")
    print("spot mlx_lm.lora")
    for hit in index.search("mlx_lm.lora", k=3):
        print(f"  {hit['score']:.4f} {hit['id']}")
    print(f"query_prefix={QUERY_PREFIX!r}")


if __name__ == "__main__":
    main()

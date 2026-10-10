"""Time 10 reranks of one shortlist built outside the timer."""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

from hybrid import hybrid_search
from pipeline import load
from rerank import MODEL_NAME, get_model, rerank

QUERY = "Why not run the cross-encoder on every chunk?"
REPEATS = 10


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    state = load()
    shortlist = hybrid_search(state.dense, state.lexical, QUERY, k_each=20)[:20]
    if not shortlist:
        raise SystemExit("shortlist is empty")

    samples_ms: list[float] = []
    last_hits: list[dict] = []
    for _ in range(REPEATS):
        tick = time.perf_counter()
        last_hits = rerank(QUERY, shortlist, top_n=5)
        samples_ms.append((time.perf_counter() - tick) * 1000.0)
    rerank_ms_p50 = round(float(statistics.median(samples_ms)), 4)

    payload = {
        "pairs": len(shortlist),
        "rerank_ms_p50": rerank_ms_p50,
        "model": MODEL_NAME,
    }
    eval_dir = root / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    latency_path = eval_dir / "rerank_latency.json"
    rendered = json.dumps(payload, indent=2) + "\n"
    latency_path.write_text(rendered, encoding="utf-8")

    top = last_hits[0]
    first_line = top["text"].splitlines()[0]
    spot = "\n".join(
        [
            "# Rerank spot-check",
            "",
            f"Query: {QUERY}",
            "",
            f"Top id: {top['id']}",
            "",
            f"> {first_line}",
            "",
        ]
    )
    spot_path = eval_dir / "rerank_spotcheck.md"
    spot_path.write_text(spot, encoding="utf-8")

    encoder = get_model()
    resolved_cross = getattr(getattr(encoder, "model", None), "name_or_path", MODEL_NAME)

    print(f"corpus_chunks={state.chunks}")
    print(f"shortlist_pairs={len(shortlist)}")
    print(f"repeats={REPEATS}")
    print("rerank_ms_samples=" + ",".join(f"{sample:.4f}" for sample in samples_ms))
    print(f"rerank_ms_median={statistics.median(samples_ms):.6f}")
    print(f"rerank_ms_p50={rerank_ms_p50}")
    print(f"model={MODEL_NAME}")
    print(f"resolved_model={resolved_cross}")
    print(rendered, end="")
    print(f"spot_top_id={top['id']}")
    print(f"spot_rerank_score={top['rerank_score']:.6f}")
    print(f"spot_first_line={first_line}")
    print(f"wrote {latency_path.relative_to(root)}")
    print(f"wrote {spot_path.relative_to(root)}")


if __name__ == "__main__":
    main()

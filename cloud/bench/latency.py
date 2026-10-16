"""Local latency for search, ingest, and the token stream.

Search is 20 sequential calls. Ingest is 3 jobs of queue-note.md, from POST
until GET says done. Token TTFT is 5 calls after one warmup.
"""

from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
API = "http://127.0.0.1:8000"
STREAM = "http://127.0.0.1:8001/v1/chat/completions"


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] + (ordered[high] - ordered[low]) * weight


def post(url: str, payload: dict, timeout: float = 120) -> tuple[dict, float]:
    data = json.dumps(payload).encode()
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.load(response)
    return body, time.perf_counter() - started


def ttft(prompt: str) -> float:
    payload = json.dumps(
        {"messages": [{"role": "user", "content": prompt}], "max_tokens": 16, "stream": True}
    ).encode()
    request = urllib.request.Request(STREAM, data=payload, headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=120) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            body = json.loads(line[len("data: ") :])
            if body["choices"][0]["delta"].get("content"):
                return time.perf_counter() - started
    raise SystemExit("no token")


def ingest_seconds() -> float:
    started = time.perf_counter()
    body, _ = post(API + "/jobs/ingest", {"path": "data/raw/queue-note.md"})
    job_id = body["job_id"]
    while True:
        with urllib.request.urlopen(API + f"/jobs/{job_id}", timeout=30) as response:
            status = json.load(response)
        if status.get("status") in {"done", "failed"}:
            if status["status"] != "done":
                raise SystemExit(status)
            return time.perf_counter() - started
        time.sleep(0.05)


def main() -> None:
    search = []
    for _ in range(20):
        _, elapsed = post(API + "/search", {"query": "What is a cross-encoder?", "k": 5})
        search.append(elapsed * 1000)
    ingest = [ingest_seconds() for _ in range(3)]
    ttft("Say hello in one sentence.")
    tokens = [ttft("Say hello in one sentence.") for _ in range(5)]
    result = {
        "search_p50_ms": percentile(search, 0.50),
        "search_p95_ms": percentile(search, 0.95),
        "ingest_p50_s": percentile(ingest, 0.50),
        "ttft_p50_s": percentile(tokens, 0.50),
        "search_ms": search,
        "ingest_s": ingest,
        "ttft_s": tokens,
    }
    path = ROOT / "bench" / "latency.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("search_p50_ms", "search_p95_ms", "ingest_p50_s", "ttft_p50_s")}))


if __name__ == "__main__":
    main()

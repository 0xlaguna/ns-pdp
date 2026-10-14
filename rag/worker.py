"""arq ingest worker.

The job rebuilds chunks and swaps the index. ``max_tries`` is 2.
A path that escapes ``data/raw`` is marked failed and does not write chunks.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from arq.connections import RedisSettings

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from rebuild import rebuild_from_raw  # noqa: E402


def redis_settings() -> RedisSettings:
    url = os.environ.get("REDIS_URL", "redis://127.0.0.1:6379")
    return RedisSettings.from_dsn(url)


async def ingest(ctx, path: str, job_id: str) -> None:
    out = ROOT / "data" / "jobs" / f"{job_id}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = rebuild_from_raw(ROOT, path)
        payload = {"path": path, "job_id": job_id, "status": "done", "chunks": result["chunks"]}
    except Exception as exc:
        payload = {"path": path, "job_id": job_id, "status": "failed", "error": str(exc)}
    out.write_text(json.dumps(payload), encoding="utf-8")


class WorkerSettings:
    functions = [ingest]
    redis_settings = redis_settings()
    max_tries = 2
    job_timeout = 600

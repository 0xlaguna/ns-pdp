"""RAG search API.

The day check serves this module with ``uvicorn src.api:app --port 8000``.
The current FastAPI first-steps page starts an app with ``fastapi dev`` instead.
``src`` is inserted on ``sys.path`` so ``pipeline`` can import ``dense`` the
same way ``python src/time_rerank.py`` does.
"""

from __future__ import annotations

import json
import os
import sys
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import pipeline
from fastapi import FastAPI
from pydantic import BaseModel


@asynccontextmanager
async def lifespan(_app: FastAPI):
    pipeline.load()
    yield


app = FastAPI(lifespan=lifespan)


class SearchRequest(BaseModel):
    query: str
    k: int = 5


class IngestRequest(BaseModel):
    path: str


def _job_file(job_id: str) -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "jobs" / f"{job_id}.json"


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "chunks": pipeline.chunk_count()}


@app.post("/search")
def search(body: SearchRequest) -> dict:
    hits = pipeline.search(body.query, k=body.k)
    return {
        "hits": [
            {
                "id": hit["id"],
                "text": hit["text"],
                "rerank_score": float(hit["rerank_score"]),
            }
            for hit in hits
        ]
    }


@app.post("/jobs/ingest")
async def enqueue_ingest(body: IngestRequest) -> dict:
    from arq import create_pool
    from arq.connections import RedisSettings

    job_id = uuid.uuid4().hex
    pool = await create_pool(
        RedisSettings.from_dsn(os.environ.get("REDIS_URL", "redis://127.0.0.1:6379"))
    )
    try:
        await pool.enqueue_job("ingest", body.path, job_id, _job_id=job_id)
    finally:
        await pool.aclose()
    return {"job_id": job_id}


@app.get("/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    path = _job_file(job_id)
    if not path.is_file():
        return {"status": "queued"}
    return json.loads(path.read_text(encoding="utf-8"))

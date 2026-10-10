"""RAG search API.

The day check serves this module with ``uvicorn src.api:app --port 8000``.
The current FastAPI first-steps page starts an app with ``fastapi dev`` instead.
``src`` is inserted on ``sys.path`` so ``pipeline`` can import ``dense`` the
same way ``python src/time_rerank.py`` does.
"""

from __future__ import annotations

import sys
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

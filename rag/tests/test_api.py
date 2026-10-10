from fastapi.testclient import TestClient

import pipeline
from api import app


def test_health_reports_loaded_chunk_count(monkeypatch):
    monkeypatch.setattr(pipeline, "load", lambda: None)
    monkeypatch.setattr(pipeline, "chunk_count", lambda: 18)
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "chunks": 18}


def test_search_returns_rerank_hits(monkeypatch):
    monkeypatch.setattr(pipeline, "load", lambda: None)
    seen: dict = {}

    def fake_search(query: str, k: int = 5) -> list[dict]:
        seen["query"] = query
        seen["k"] = k
        return [
            {
                "id": "fieldguide-runbook:0",
                "text": "Search is POST /search.",
                "rerank_score": 0.82,
                "fusion_score": 0.032,
            }
        ]

    monkeypatch.setattr(pipeline, "search", fake_search)
    with TestClient(app) as client:
        response = client.post("/search", json={"query": "POST /search", "k": 4})
    assert response.status_code == 200
    assert response.json() == {
        "hits": [
            {
                "id": "fieldguide-runbook:0",
                "text": "Search is POST /search.",
                "rerank_score": 0.82,
            }
        ]
    }
    assert seen == {"query": "POST /search", "k": 4}

"""Contract checks against a running Fieldguide API.

``FIELDGUIDE_BASE`` defaults to http://127.0.0.1:8000.
"""

from __future__ import annotations

import json
import os
import urllib.request

BASE = os.environ.get("FIELDGUIDE_BASE", "http://127.0.0.1:8000").rstrip("/")


def _get(path: str) -> dict:
    with urllib.request.urlopen(BASE + path, timeout=60) as response:
        return json.load(response)


def _post(path: str, payload: dict) -> dict:
    data = json.dumps(payload).encode()
    request = urllib.request.Request(
        BASE + path,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def test_health_ok_and_chunks():
    body = _get("/health")
    assert body["status"] == "ok"
    assert body["chunks"] > 0


def test_search_hit_has_id_and_text():
    body = _post("/search", {"query": "What is a cross-encoder?", "k": 5})
    assert body["hits"]
    hit = body["hits"][0]
    assert hit["id"]
    assert hit["text"]


if __name__ == "__main__":
    test_health_ok_and_chunks()
    test_search_hit_has_id_and_text()
    print("contract ok")

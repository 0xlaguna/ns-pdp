"""Prompt for a spoken turn. The model only sees the retrieved notes."""

from __future__ import annotations

import json
import urllib.error
import urllib.request


def build_messages(transcript: str, hits: list[dict]) -> list[dict]:
    notes = "\n\n".join(str(hit.get("text", "")) for hit in hits)
    content = (
        "You answer only from the notes below. If the notes do not contain the answer, "
        "say you do not know. Do not guess, and do not describe a system the notes do not mention.\n"
        "Keep the answer to 2-4 sentences.\n"
        "Notes:\n"
        f"{notes}\n"
        "Question:\n"
        f"{transcript}"
    )
    return [{"role": "user", "content": content}]


def search_hits(transcript: str, rag_url: str, k: int = 5) -> list[dict]:
    url = rag_url.rstrip("/") + "/search"
    payload = json.dumps({"query": transcript, "k": k}).encode()
    request = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = json.load(response)
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(f"RAG server unreachable at {url}: {exc}") from exc
    return list(body.get("hits") or [])

"""Read token deltas from the MLX stream on port 8001."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request


def iter_tokens(messages: list[dict], base_url: str, max_tokens: int = 64):
    url = base_url.rstrip("/") + "/v1/chat/completions"
    payload = json.dumps(
        {"messages": messages, "max_tokens": max_tokens, "stream": True}
    ).encode()
    request = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        response = urllib.request.urlopen(request, timeout=30)
    except (urllib.error.URLError, TimeoutError) as exc:
        raise SystemExit(f"stream server unreachable at {url}: {exc}") from exc
    with response:
        for raw in response:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data: "):
                continue
            data = line[len("data: ") :]
            if data == "[DONE]":
                return
            body = json.loads(data)
            text = body["choices"][0]["delta"].get("content") or ""
            if not text:
                continue
            yield text


def time_to_first_token(messages: list[dict], base_url: str) -> float:
    for _ in iter_tokens(messages, base_url):
        break
    started = time.perf_counter()
    for _ in iter_tokens(messages, base_url):
        return time.perf_counter() - started
    raise SystemExit("second call produced no token")

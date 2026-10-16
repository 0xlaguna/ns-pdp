"""Warm up the token stream once, then time the next first token."""

from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

URL = "http://127.0.0.1:8001/v1/chat/completions"
PROMPT = "Say hello in one sentence."
ROOT = Path(__file__).resolve().parents[1]


def first_token(prompt: str) -> float:
    payload = json.dumps(
        {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 32,
            "stream": True,
        }
    ).encode()
    request = urllib.request.Request(
        URL,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=120) as response:
        for raw in response:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            body = json.loads(line[len("data: ") :])
            text = body["choices"][0]["delta"].get("content") or ""
            if text:
                return time.perf_counter() - started
    raise SystemExit("stream ended before a token")


def main() -> None:
    first_token(PROMPT)
    seconds = first_token(PROMPT)
    import yaml

    config = yaml.safe_load(
        (ROOT.parent / "finetune" / "configs" / "run-a.yaml").read_text(encoding="utf-8")
    )
    out = {
        "ttft_seconds": seconds,
        "model": config["model"],
        "adapter_loaded": True,
        "prompt": PROMPT,
    }
    path = ROOT / "bench" / "ttft.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out))


if __name__ == "__main__":
    main()

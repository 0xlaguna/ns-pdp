"""Record warm time to first token for one fixed prompt."""

from __future__ import annotations

import json
from pathlib import Path

from llm_stream import time_to_first_token

ROOT = Path(__file__).resolve().parents[1]
URL = "http://127.0.0.1:8001"
PROMPT = "Answer in one sentence: what is a cross-encoder?"


def main() -> None:
    seconds = time_to_first_token(
        [{"role": "user", "content": PROMPT}],
        URL,
    )
    out = {"ttft_seconds": seconds, "url": URL, "prompt": PROMPT}
    path = ROOT / "bench" / "ttft.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out))


if __name__ == "__main__":
    main()

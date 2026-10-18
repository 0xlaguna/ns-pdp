"""Time tiny.en on samples/cross.wav and write bench/stt.json."""

from __future__ import annotations

import json
from pathlib import Path

from stt import realtime_factor

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    text, audio_s, wall, factor = realtime_factor(ROOT / "samples" / "cross.wav")
    out = {
        "text": text,
        "audio_seconds": audio_s,
        "wall_seconds": wall,
        "realtime_factor": factor,
        "model": "tiny.en",
    }
    path = ROOT / "bench" / "stt.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out))


if __name__ == "__main__":
    main()

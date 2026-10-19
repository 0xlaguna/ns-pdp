"""Time one Piper sentence and write bench/tts.json."""

from __future__ import annotations

import json
import time
from pathlib import Path

from tts import synthesize

ROOT = Path(__file__).resolve().parents[1]
SENTENCE = "A cross encoder reads the query and the passage together."


def main() -> None:
    started = time.perf_counter()
    wav = synthesize(SENTENCE)
    wall = time.perf_counter() - started
    out_path = ROOT / "bench" / "tts-sentence.wav"
    out_path.write_bytes(wav)
    result = {
        "wall_seconds": wall,
        "wav_bytes": len(wav),
        "voice": "en_US-lessac-medium",
        "text": SENTENCE,
    }
    path = ROOT / "bench" / "tts.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()

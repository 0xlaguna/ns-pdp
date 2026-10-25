"""Median stage times for one spoken turn on samples/cross.wav.

One warmup, then three measured runs. first_wav_ms is from the start of the
turn until the first sentence wav exists.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import yaml

from llm_stream import iter_tokens
from segment import SentenceBuffer
from stt import transcribe
from tts import synthesize
from turn import build_messages, search_hits
from vad import trim_wav

ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
CLIP = ROOT / "samples" / "cross.wav"


def median(values: list[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def one_turn() -> dict:
    started = time.perf_counter()
    trimmed, _stamps = trim_wav(CLIP)
    vad_ms = (time.perf_counter() - started) * 1000
    stt_started = time.perf_counter()
    text = transcribe(trimmed)
    stt_ms = (time.perf_counter() - stt_started) * 1000
    if trimmed != CLIP:
        trimmed.unlink(missing_ok=True)
    search_started = time.perf_counter()
    hits = search_hits(text, CONFIG["rag_url"])
    search_ms = (time.perf_counter() - search_started) * 1000
    messages = build_messages(text, hits)
    token_started = time.perf_counter()
    buffer = SentenceBuffer()
    ttft_ms = None
    wav = b""
    for token in iter_tokens(messages, CONFIG["stream_url"], max_tokens=80):
        if ttft_ms is None:
            ttft_ms = (time.perf_counter() - token_started) * 1000
        ready = buffer.push(token)
        if ready:
            wav = synthesize(ready[0])
            break
    if not wav:
        leftover = buffer.flush() or text
        wav = synthesize(leftover)
    first_wav_ms = (time.perf_counter() - started) * 1000
    return {
        "vad_ms": vad_ms,
        "stt_ms": stt_ms,
        "search_ms": search_ms,
        "ttft_ms": ttft_ms or 0.0,
        "first_wav_ms": first_wav_ms,
        "wav_bytes": len(wav),
    }


def main() -> None:
    one_turn()
    runs = [one_turn() for _ in range(3)]
    medians = {
        key: median([run[key] for run in runs])
        for key in ("vad_ms", "stt_ms", "search_ms", "ttft_ms", "first_wav_ms")
    }
    out = {"medians": medians, "runs": runs, "n": 3, "warmup": 1}
    path = ROOT / "bench" / "turn_latency.json"
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(medians))


if __name__ == "__main__":
    main()

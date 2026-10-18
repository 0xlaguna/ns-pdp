"""Transcribe a wav with faster-whisper tiny.en on CPU."""

from __future__ import annotations

import time
import wave
from pathlib import Path

from faster_whisper import WhisperModel

_model: WhisperModel | None = None


def get_model() -> WhisperModel:
    global _model
    if _model is None:
        _model = WhisperModel("tiny.en", device="cpu", compute_type="int8")
    return _model


def audio_seconds(path: str | Path) -> float:
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes() / float(handle.getframerate())


def transcribe(path: str | Path) -> str:
    segments, _info = get_model().transcribe(str(path), vad_filter=True)
    return " ".join(segment.text for segment in segments).strip()


def realtime_factor(path: str | Path) -> tuple[str, float, float, float]:
    duration = audio_seconds(path)
    started = time.perf_counter()
    text = transcribe(path)
    wall = time.perf_counter() - started
    return text, duration, wall, duration / wall

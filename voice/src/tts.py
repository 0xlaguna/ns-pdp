"""Piper synthesis. The current package is piper-tts and ``PiperVoice.synthesize_wav``."""

from __future__ import annotations

import io
import wave
from pathlib import Path

from piper import PiperVoice

ROOT = Path(__file__).resolve().parents[1]
VOICE_PATH = ROOT / "voices" / "en_US-lessac-medium.onnx"

_voice: PiperVoice | None = None


def get_voice() -> PiperVoice:
    global _voice
    if _voice is None:
        _voice = PiperVoice.load(str(VOICE_PATH))
    return _voice


def synthesize(text: str) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        get_voice().synthesize_wav(text, handle)
    return buffer.getvalue()

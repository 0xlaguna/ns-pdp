"""Silero speech timestamps.

``read_audio`` in silero-vad 6 needs torchaudio or torchcodec, and neither is
installed. The model itself does not. This module reads 16-bit PCM with the
stdlib ``wave`` module and passes a float tensor to ``get_speech_timestamps``.
"""

from __future__ import annotations

import wave
from pathlib import Path

import torch
from silero_vad import get_speech_timestamps, load_silero_vad

_model = None


def load_model():
    global _model
    if _model is None:
        _model = load_silero_vad()
    return _model


def read_wav(path: str | Path) -> tuple[torch.Tensor, int]:
    with wave.open(str(path), "rb") as handle:
        rate = handle.getframerate()
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        frames = handle.readframes(handle.getnframes())
    if width != 2:
        raise ValueError(f"expected 16-bit wav, got width {width}")
    samples = torch.frombuffer(bytearray(frames), dtype=torch.int16).to(torch.float32) / 32768.0
    if channels > 1:
        samples = samples.view(-1, channels).mean(dim=1)
    return samples, rate


def speech_timestamps(path: str | Path) -> list[dict]:
    audio, rate = read_wav(path)
    if rate != 16000:
        raise ValueError(f"expected 16000 Hz, got {rate}")
    stamps = get_speech_timestamps(audio, load_model(), sampling_rate=rate, return_seconds=True)
    return list(stamps)


def trim_wav(path: str | Path) -> tuple[Path, list[dict]]:
    source = Path(path)
    stamps = speech_timestamps(source)
    audio, rate = read_wav(source)
    if not stamps:
        return source, []
    pieces = []
    for stamp in stamps:
        start = int(float(stamp["start"]) * rate)
        end = int(float(stamp["end"]) * rate)
        pieces.append(audio[start:end])
    trimmed = torch.cat(pieces) if pieces else audio[:0]
    pcm = (trimmed.clamp(-1, 1) * 32767).to(torch.int16).numpy().tobytes()
    dest = source.with_name(source.stem + "-speech.wav")
    with wave.open(str(dest), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(pcm)
    return dest, stamps

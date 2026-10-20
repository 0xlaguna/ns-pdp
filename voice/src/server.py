"""Voice socket at ws://127.0.0.1:8010/ws.

A text turn streams tokens from port 8001 and a Piper wav for each closed
sentence. An audio turn transcribes first. barge_in cancels the turn and
sends cancel_audio before any further token or audio.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import tempfile
import threading
from pathlib import Path

import yaml
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from llm_stream import iter_tokens
from messages import parse_client
from segment import SentenceBuffer
from stt import transcribe
from tts import synthesize
from turn import build_messages, search_hits
from vad import trim_wav

ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
STREAM_URL = CONFIG["stream_url"]

app = FastAPI()


def messages_for(text: str) -> list[dict]:
    hits = search_hits(text, CONFIG["rag_url"])
    print("hits " + ",".join(str(hit.get("id", "")) for hit in hits), flush=True)
    return build_messages(text, hits)


async def speak(ws: WebSocket, text: str, stop: threading.Event) -> None:
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def produce() -> None:
        try:
            for token in iter_tokens(messages_for(text), STREAM_URL, max_tokens=80):
                if stop.is_set():
                    break
                loop.call_soon_threadsafe(queue.put_nowait, ("token", token))
        except SystemExit as exc:
            loop.call_soon_threadsafe(queue.put_nowait, ("error", str(exc)))
        except Exception as exc:
            loop.call_soon_threadsafe(queue.put_nowait, ("error", str(exc)))
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, ("end", ""))

    worker = asyncio.create_task(asyncio.to_thread(produce))
    buffer = SentenceBuffer()
    try:
        while True:
            if stop.is_set():
                return
            kind, value = await queue.get()
            if stop.is_set():
                return
            if kind == "end":
                break
            if kind == "error":
                await ws.send_json({"type": "error", "message": value})
                return
            await ws.send_json({"type": "token", "text": value})
            for sentence in buffer.push(value):
                if stop.is_set():
                    return
                wav = await asyncio.to_thread(synthesize, sentence)
                if stop.is_set():
                    return
                await ws.send_json(
                    {"type": "audio", "wav_b64": base64.b64encode(wav).decode("ascii")}
                )
        leftover = buffer.flush()
        if leftover and not stop.is_set():
            wav = await asyncio.to_thread(synthesize, leftover)
            if not stop.is_set():
                await ws.send_json(
                    {"type": "audio", "wav_b64": base64.b64encode(wav).decode("ascii")}
                )
        if not stop.is_set():
            await ws.send_json({"type": "turn_done"})
    finally:
        stop.set()
        worker.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await worker


@app.websocket("/ws")
async def socket(ws: WebSocket) -> None:
    await ws.accept()
    task: asyncio.Task | None = None
    stop = threading.Event()

    async def cancel_turn() -> None:
        nonlocal task, stop
        stop.set()
        current = task
        task = None
        if current is not None and not current.done():
            current.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await current

    try:
        while True:
            payload = json.loads(await ws.receive_text())
            try:
                message = parse_client(payload)
            except ValueError as exc:
                await ws.send_json({"type": "error", "message": str(exc)})
                continue
            if message.type == "barge_in":
                await cancel_turn()
                await ws.send_json({"type": "cancel_audio"})
                stop = threading.Event()
                continue
            if message.type in {"start", "stop", "pcm"}:
                continue
            if message.type == "text":
                utterance = message.text
            elif message.type == "audio":
                raw = base64.b64decode(message.wav_b64)
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
                    handle.write(raw)
                    wav_path = Path(handle.name)
                trimmed = wav_path
                try:
                    trimmed, stamps = trim_wav(wav_path)
                    print(f"vad {stamps}", flush=True)
                    transcript = await asyncio.to_thread(transcribe, str(trimmed))
                finally:
                    wav_path.unlink(missing_ok=True)
                    if trimmed != wav_path:
                        trimmed.unlink(missing_ok=True)
                await ws.send_json({"type": "transcript", "text": transcript})
                if not transcript.strip():
                    await ws.send_json({"type": "error", "message": "no speech"})
                    continue
                utterance = transcript
            else:
                continue
            await cancel_turn()
            stop = threading.Event()
            task = asyncio.create_task(speak(ws, utterance, stop))
    except WebSocketDisconnect:
        await cancel_turn()


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8010)


if __name__ == "__main__":
    main()

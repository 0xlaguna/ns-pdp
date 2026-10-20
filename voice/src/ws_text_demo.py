"""Send one text turn and save the first wav."""

from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path

import websockets

ROOT = Path(__file__).resolve().parents[1]
UTTERANCE = "What is a cross-encoder?"


async def main() -> None:
    tokens: list[str] = []
    wav: bytes | None = None
    async with websockets.connect("ws://127.0.0.1:8010/ws") as socket:
        await socket.send(json.dumps({"type": "text", "text": UTTERANCE}))
        async for raw in socket:
            message = json.loads(raw)
            if message["type"] == "token":
                tokens.append(message["text"])
                print(message["text"], end="", flush=True)
            elif message["type"] == "audio" and wav is None:
                wav = base64.b64decode(message["wav_b64"])
            elif message["type"] == "turn_done":
                break
            elif message["type"] == "error":
                raise SystemExit(message["message"])
    print()
    if not wav:
        raise SystemExit("no wav")
    path = ROOT / "bench" / "ws-sentence.wav"
    path.write_bytes(wav)
    print(f"tokens={len(tokens)} wav_bytes={len(wav)} path={path}")


if __name__ == "__main__":
    asyncio.run(main())

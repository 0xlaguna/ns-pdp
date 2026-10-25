"""Send a long text turn and barge in after several tokens or two wavs."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import websockets

ROOT = Path(__file__).resolve().parents[1]


async def main() -> None:
    tokens_before = 0
    audios_before = 0
    tokens_after = 0
    audios_after = 0
    cancel = False
    lines = []
    async with websockets.connect("ws://127.0.0.1:8010/ws") as socket:
        await socket.send(json.dumps({"type": "text", "text": "Explain retrieve and rerank in detail."}))
        while True:
            try:
                raw = await asyncio.wait_for(socket.recv(), timeout=3)
            except TimeoutError:
                break
            message = json.loads(raw)
            kind = message["type"]
            if kind == "cancel_audio":
                cancel = True
                lines.append("cancel_audio")
                continue
            if kind == "token":
                if cancel:
                    tokens_after += 1
                    lines.append("token-after " + message["text"])
                else:
                    tokens_before += 1
                    lines.append("token " + message["text"])
                    if tokens_before >= 6 or audios_before >= 2:
                        await socket.send(json.dumps({"type": "barge_in"}))
            elif kind == "audio":
                if cancel:
                    audios_after += 1
                    lines.append("audio-after")
                else:
                    audios_before += 1
                    lines.append("audio")
                    if tokens_before >= 6 or audios_before >= 2:
                        await socket.send(json.dumps({"type": "barge_in"}))
            elif kind == "turn_done" and not cancel:
                lines.append("turn_done before cancel")
                break
            elif kind == "error":
                lines.append("error " + message.get("message", ""))
    summary = {
        "tokens_before": tokens_before,
        "audios_before": audios_before,
        "cancel_audio": cancel,
        "tokens_after": tokens_after,
        "audios_after": audios_after,
    }
    text = "\n".join(lines + ["", json.dumps(summary)])
    path = ROOT / "notes" / "barge.md"
    path.write_text("# Barge-in\n\n```\n" + text + "\n```\n", encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    asyncio.run(main())

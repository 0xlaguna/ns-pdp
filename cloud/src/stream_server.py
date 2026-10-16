"""OpenAI-style token stream from the MLX model on port 8001.

``FIELDGUIDE_LLM`` must be ``mlx``. The winning adapter is run A. mlx-lm 0.32
``stream_generate`` yields objects whose text is ``response.text``. Temperature
goes through ``make_sampler``, not a ``temp`` argument.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import yaml
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler
from pydantic import BaseModel, Field

EVIDENCE = Path(__file__).resolve().parents[2]
CONFIG = EVIDENCE / "finetune" / "configs" / "run-a.yaml"
ADAPTER = EVIDENCE / "finetune" / "adapters" / "run-a"

if os.environ.get("FIELDGUIDE_LLM", "mlx") != "mlx":
    raise SystemExit("FIELDGUIDE_LLM must be mlx for this server")

_cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
MODEL_ID = _cfg["model"]
model, tokenizer = load(MODEL_ID, adapter_path=str(ADAPTER))
ADAPTER_LOADED = True

app = FastAPI()


class ChatRequest(BaseModel):
    messages: list[dict]
    max_tokens: int = Field(default=64)
    stream: bool = False


def _prompt(messages: list[dict]):
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return tokenizer.encode(rendered, add_special_tokens=False)


def _pieces(messages: list[dict], max_tokens: int):
    # mlx-lm's generation stream is thread-local and is created when this module
    # is imported. Starlette runs a sync generator on a worker thread, which then
    # raises "There is no Stream(gpu, 1)". An async generator stays on the
    # event-loop thread, which is the thread that imported mlx_lm.
    prompt = _prompt(messages)
    for response in stream_generate(
        model,
        tokenizer,
        prompt,
        max_tokens=max_tokens,
        sampler=make_sampler(0.0),
    ):
        if response.text:
            yield response.text


async def _sse(messages: list[dict], max_tokens: int):
    for text in _pieces(messages, max_tokens):
        payload = {"choices": [{"delta": {"content": text}}]}
        yield f"data: {json.dumps(payload)}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
async def chat(body: ChatRequest):
    if body.stream:
        return StreamingResponse(_sse(body.messages, body.max_tokens), media_type="text/event-stream")
    text = "".join(_pieces(body.messages, body.max_tokens))
    return {"choices": [{"message": {"role": "assistant", "content": text}}]}

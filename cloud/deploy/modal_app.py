"""Same SSE chat contract on a Modal GPU.

Not deployed from this machine. The GPU guide lists T4 first, and it says a
payment method is required. ``scaledown_window`` is the current idle timeout.
"""

import modal

image = (
    modal.Image.debian_slim()
    .pip_install("fastapi[standard]", "transformers", "torch", "accelerate")
)
app = modal.App("fieldguide-stream")
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


@app.function(image=image, gpu="T4", scaledown_window=5)
@modal.concurrent(max_inputs=4)
@modal.asgi_app()
def web():
    import json

    import torch
    from fastapi import FastAPI
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel, Field
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
    model.to("cuda")
    model.eval()
    api = FastAPI()

    class ChatRequest(BaseModel):
        messages: list[dict]
        max_tokens: int = Field(default=64)
        stream: bool = False

    def pieces(messages: list[dict], max_tokens: int):
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        output = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
        new_tokens = output[0][inputs["input_ids"].shape[-1] :]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        if text:
            yield text

    @api.get("/health")
    def health():
        return {"status": "ok", "model": MODEL_ID}

    @api.post("/v1/chat/completions")
    def chat(body: ChatRequest):
        if body.stream:
            def events():
                for text in pieces(body.messages, body.max_tokens):
                    payload = {"choices": [{"delta": {"content": text}}]}
                    yield f"data: {json.dumps(payload)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(events(), media_type="text/event-stream")
        text = "".join(pieces(body.messages, body.max_tokens))
        return {"choices": [{"message": {"role": "assistant", "content": text}}]}

    return api

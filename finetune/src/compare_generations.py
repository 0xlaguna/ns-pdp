"""Generate the 6 test prompts with the base model and with the chosen adapter.

This follows ``mlx_lm.generate`` as installed: chat template on, temperature 0,
``--max-tokens 80``. The chosen adapter is the run with the lower final val loss.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

ROOT = Path(__file__).resolve().parents[1]
MODEL = "mlx-community/Llama-3.2-1B-Instruct-4bit"
MAX_TOKENS = 80


def prompts() -> list[str]:
    rows = []
    for line in (ROOT / "data" / "test.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line)["messages"][0]["content"])
    return rows


def encode(tokenizer, text: str) -> list[int]:
    messages = [{"role": "user", "content": text}]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return tokenizer.encode(prompt, add_special_tokens=False)


def run_split(adapter: str | None, questions: list[str]) -> list[str]:
    model, tokenizer = load(MODEL, adapter_path=adapter)
    answers = []
    for question in questions:
        text = generate(
            model,
            tokenizer,
            encode(tokenizer, question),
            max_tokens=MAX_TOKENS,
            sampler=make_sampler(0.0),
            verbose=False,
        )
        answers.append(text.strip())
    del model
    mx.clear_cache()
    return answers


def main() -> None:
    questions = prompts()
    base = run_split(None, questions)
    adapter = run_split(str(ROOT / "adapters" / "run-a"), questions)
    parts = [
        "# Base vs rank-8 adapter",
        "",
        "Chosen run: A. It had the lower validation loss (3.507 vs 3.608).",
        "Command shape: `mlx_lm.generate --model mlx-community/Llama-3.2-1B-Instruct-4bit "
        "--max-tokens 80 --temp 0 --prompt \"<user>\"`, then the same command with "
        "`--adapter-path adapters/run-a`.",
        "`generate()` in mlx-lm 0.32.0 does not take `temp`. The CLI builds `make_sampler(temp)` and passes `sampler`. This script does the same thing, with temperature 0.",
        "The weights load twice, once for the base model and once with the adapter, instead of twelve times.",
        "",
    ]
    for index, question in enumerate(questions, start=1):
        parts.extend(
            [
                f"## Prompt {index}",
                "",
                question,
                "",
                "### Base",
                "",
                base[index - 1],
                "",
                "### Adapter",
                "",
                adapter[index - 1],
                "",
            ]
        )
    out = ROOT / "runs" / "generations.md"
    out.write_text("\n".join(parts), encoding="utf-8")
    print(f"wrote {out} prompts={len(questions)} completions={len(questions) * 2}")


if __name__ == "__main__":
    main()

"""Check the chat jsonl splits before a training run.

Exits non-zero on a schema error, an assistant length outside 80–500
characters, a 2–4 sentence miss, a bullet list, an exact user-string leak
across splits, or a count other than 24/6/6. On success prints one JSON
object with the split counts, the mean assistant length, and ok true.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
EXPECTED = {"train": 24, "valid": 6, "test": 6}
SPLITS = ("train", "valid", "test")


def _sentences(text: str) -> int:
    # A boundary is punctuation followed by whitespace or the end of the
    # string. A dot inside a name such as mlx_lm.lora is not a boundary.
    parts = re.split(r"[.!?]+(?:\s+|$)", text.strip())
    return len([part for part in parts if part.strip()])


def _has_bullet(text: str) -> bool:
    for line in text.splitlines():
        stripped = line.lstrip()
        if re.match(r"([-*•]|\d+[.)])\s+", stripped):
            return True
    return False


def _load(name: str) -> list[dict]:
    path = DATA_DIR / f"{name}.jsonl"
    if not path.is_file():
        raise SystemExit(f"missing {path}")
    rows: list[dict] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            raise SystemExit(f"{name}:{number} blank line")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{name}:{number} invalid json: {exc}") from exc
        messages = row.get("messages") if isinstance(row, dict) else None
        if (
            not isinstance(messages, list)
            or len(messages) != 2
            or not isinstance(messages[0], dict)
            or not isinstance(messages[1], dict)
        ):
            raise SystemExit(f"{name}:{number} schema")
        user, assistant = messages
        if user.get("role") != "user" or assistant.get("role") != "assistant":
            raise SystemExit(f"{name}:{number} schema")
        user_text = user.get("content")
        assistant_text = assistant.get("content")
        if not isinstance(user_text, str) or not isinstance(assistant_text, str):
            raise SystemExit(f"{name}:{number} schema")
        if not user_text.strip() or not assistant_text.strip():
            raise SystemExit(f"{name}:{number} schema")
        length = len(assistant_text)
        if length < 80 or length > 500:
            raise SystemExit(f"{name}:{number} assistant length {length}")
        count = _sentences(assistant_text)
        if count < 2 or count > 4:
            raise SystemExit(f"{name}:{number} sentence count {count}")
        if _has_bullet(user_text) or _has_bullet(assistant_text):
            raise SystemExit(f"{name}:{number} bullet list")
        rows.append(row)
    return rows


def _users(rows: list[dict]) -> list[str]:
    return [row["messages"][0]["content"] for row in rows]


def main() -> None:
    loaded = {name: _load(name) for name in SPLITS}
    for name, expected in EXPECTED.items():
        if len(loaded[name]) != expected:
            raise SystemExit(f"{name} count {len(loaded[name])} != {expected}")

    seen: dict[str, str] = {}
    for name in SPLITS:
        for text in _users(loaded[name]):
            if text in seen:
                raise SystemExit("leak")
            seen[text] = name

    lengths = [
        len(row["messages"][1]["content"])
        for name in SPLITS
        for row in loaded[name]
    ]
    mean = sum(lengths) / len(lengths)
    print(
        json.dumps(
            {
                "train": len(loaded["train"]),
                "valid": len(loaded["valid"]),
                "test": len(loaded["test"]),
                "mean_assistant_chars": mean,
                "ok": True,
            },
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sys.exit(0)

"""Write data/chunks.jsonl from data/raw."""

from __future__ import annotations

import json
from pathlib import Path

from chunk import load_corpus


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    rows = load_corpus(root / "data" / "raw")
    out = root / "data" / "chunks.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    first = rows[0]["id"] if rows else ""
    print(f"wrote {out.relative_to(root)} lines={len(rows)} first_id={first}")


if __name__ == "__main__":
    main()

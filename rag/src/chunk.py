"""Character-window chunker for the Fieldguide notes."""

from __future__ import annotations

from pathlib import Path


def chunk_text(text: str, size: int = 800, overlap: int = 120) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            snap = text.rfind(" ", max(0, end - 80), end)
            if snap > start:
                end = snap
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def load_corpus(raw_dir: str | Path) -> list[dict]:
    """Read raw notes in filename order and assign `{stem}:{index}` ids from 0."""
    raw_path = Path(raw_dir)
    files = sorted(
        (
            path
            for path in raw_path.iterdir()
            if path.is_file() and not path.name.startswith(".")
        ),
        key=lambda path: path.name,
    )
    rows: list[dict] = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        for index, piece in enumerate(chunk_text(text)):
            rows.append(
                {
                    "id": f"{path.stem}:{index}",
                    "source": path.name,
                    "text": piece,
                }
            )
    return rows

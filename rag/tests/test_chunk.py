from pathlib import Path

import pytest

from chunk import chunk_text, load_corpus


def test_empty_text():
    assert chunk_text("") == []
    assert chunk_text(" \n\t") == []


def test_short_text():
    assert chunk_text("hello") == ["hello"]
    assert chunk_text("  hello world  ") == ["hello world"]
    assert chunk_text("x" * 800) == ["x" * 800]


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError, match="overlap must be smaller than size"):
        chunk_text("abcdef", size=10, overlap=10)
    with pytest.raises(ValueError, match="overlap must be smaller than size"):
        chunk_text("abcdef", size=10, overlap=12)


def test_overlapping_long_text_with_no_spaces():
    size, overlap = 30, 10
    text = "a" * 100
    chunks = chunk_text(text, size=size, overlap=overlap)
    assert chunks == [
        "a" * 30,
        "a" * 30,
        "a" * 30,
        "a" * 30,
        "a" * 20,
    ]
    for previous, current in zip(chunks, chunks[1:]):
        assert previous[-overlap:] == current[:overlap]


def test_snaps_end_to_a_space_in_the_last_80_characters():
    sentence = "Windows snap backward onto a nearby space. "
    text = sentence * 30
    size, overlap = 200, 40
    chunks = chunk_text(text, size=size, overlap=overlap)
    assert len(chunks) > 1
    for piece in chunks:
        assert piece in text
        assert len(piece) <= size
    for piece in chunks[:-1]:
        assert " " in piece
        assert not piece.endswith(" ")


def test_stable_ids_across_two_files(tmp_path: Path):
    (tmp_path / "b.md").write_text("b" * 2000, encoding="utf-8")
    (tmp_path / "a.md").write_text("a" * 2000, encoding="utf-8")
    (tmp_path / ".hidden.md").write_text("skip me", encoding="utf-8")
    first = load_corpus(tmp_path)
    second = load_corpus(tmp_path)
    assert first == second
    ids = [row["id"] for row in first]
    a_ids = [chunk_id for chunk_id in ids if chunk_id.startswith("a:")]
    b_ids = [chunk_id for chunk_id in ids if chunk_id.startswith("b:")]
    assert len(a_ids) > 1
    assert len(b_ids) > 1
    assert a_ids == [f"a:{index}" for index in range(len(a_ids))]
    assert b_ids == [f"b:{index}" for index in range(len(b_ids))]
    assert ids == a_ids + b_ids
    assert first[0]["source"] == "a.md"
    assert {row["source"] for row in first} == {"a.md", "b.md"}

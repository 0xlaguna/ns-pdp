import pytest

from hybrid import hybrid_search, rrf


def test_ranked_one_and_one_beats_ranked_one_and_absent():
    fused = dict(rrf([["shared"], ["shared"], ["only"]], k=60))
    assert fused["shared"] == pytest.approx(2 / 61)
    assert fused["only"] == pytest.approx(1 / 61)
    assert fused["shared"] > fused["only"]


def test_rank_one_contributes_one_over_61_when_k_is_60():
    fused = rrf([["only"]], k=60)
    assert len(fused) == 1
    assert fused[0][0] == "only"
    assert fused[0][1] == pytest.approx(1 / 61)


def test_equal_scores_sort_by_id():
    fused = rrf([["b"], ["a"]], k=60)
    assert [doc_id for doc_id, _score in fused] == ["a", "b"]
    assert fused[0][1] == fused[1][1]


class _Stub:
    def __init__(self, hits: list[dict]) -> None:
        self.hits = hits

    def search(self, query: str, k: int = 5) -> list[dict]:
        return self.hits[:k]


def test_hybrid_search_leaves_absent_rank_as_none():
    dense = _Stub(
        [
            {"id": "b", "text": "from dense", "score": 0.9},
            {"id": "a", "text": "from dense", "score": 0.1},
        ]
    )
    lexical = _Stub(
        [
            {"id": "a", "text": "from lexical", "score": 4.0},
        ]
    )
    hits = hybrid_search(dense, lexical, "query", k_each=20, k=60)
    by_id = {hit["id"]: hit for hit in hits}
    assert [hit["id"] for hit in hits] == ["a", "b"]
    assert by_id["a"]["dense_rank"] == 2
    assert by_id["a"]["lexical_rank"] == 1
    assert by_id["a"]["text"] == "from dense"
    assert by_id["a"]["score"] == pytest.approx(1 / 62 + 1 / 61)
    assert by_id["b"]["dense_rank"] == 1
    assert by_id["b"]["lexical_rank"] is None
    assert by_id["b"]["text"] == "from dense"
    assert by_id["b"]["score"] == pytest.approx(1 / 61)

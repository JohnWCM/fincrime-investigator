"""Fast isolated checks for reciprocal rank fusion; no model downloads needed."""
from unittest.mock import patch
from types import SimpleNamespace
import sys

import pytest

from src.retrieval_advanced import hybrid_search, reranked_search


@pytest.fixture
def corpus():
    return [
        {"chunk_id": key, "text": key, "title": "Sample"}
        for key in ("a", "b", "c", "d")
    ]


def test_rrf_deduplicates_and_preserves_metadata(corpus):
    b = [corpus[i] for i in (0, 1, 2)]  # a, b, c
    d = [corpus[i] for i in (1, 3, 0)]  # b, d, a
    with patch("src.retrieval_advanced.bm25_search", return_value=b), \
         patch("src.retrieval_advanced.dense_search", return_value=d):
        hits = hybrid_search("sample", corpus, k=4, candidate_k=4)
    assert [h["chunk_id"] for h in hits] == ["b", "a", "d", "c"]
    assert len({h["chunk_id"] for h in hits}) == 4
    assert hits[0]["bm25_rank"] == 2
    assert hits[0]["dense_rank"] == 1
    assert hits[0]["rrf_score"] > hits[2]["rrf_score"]


def test_rejects_invalid_configuration(corpus):
    with pytest.raises(ValueError):
        hybrid_search("x", corpus, k=5, candidate_k=4)
    with pytest.raises(ValueError):
        hybrid_search(" ", corpus)


def test_cross_encoder_reorders_candidates_and_keeps_citations(monkeypatch, corpus):
    class FakeCrossEncoder:
        def __init__(self, model_name):
            self.model_name = model_name

        def predict(self, pairs, **kwargs):
            assert len(pairs) == 2
            return [0.1, 2.5]

    monkeypatch.setitem(
        sys.modules, "sentence_transformers",
        SimpleNamespace(CrossEncoder=FakeCrossEncoder),
    )
    candidates = [
        {**corpus[0], "rrf_score": 0.4, "page_number": 1},
        {**corpus[1], "rrf_score": 0.2, "page_number": 2},
    ]
    with patch("src.retrieval_advanced.hybrid_search", return_value=candidates):
        hits = reranked_search("example question", k=2, candidate_k=2)
    assert [h["chunk_id"] for h in hits] == ["b", "a"]
    assert hits[0]["page_number"] == 2
    assert hits[0]["method"] == "hybrid_rerank"

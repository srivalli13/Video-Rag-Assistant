"""Unit tests for retrieval/retriever.py — the hallucination-guard threshold logic."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import retrieval.retriever as retriever_module
from retrieval.retriever import retrieve


class FakeManager:
    """Stands in for FaissManager so the test doesn't need a real index."""

    def __init__(self, results):
        self._results = results

    def search(self, query_vector, k):
        return self._results[:k]


def test_retrieve_drops_chunks_below_threshold(monkeypatch):
    fake_results = [
        {"text": "on-topic chunk", "score": 0.9, "source": "a.pdf", "page": 1},
        {"text": "borderline chunk", "score": 0.4, "source": "a.pdf", "page": 2},
    ]
    monkeypatch.setattr(retriever_module, "embed_texts", lambda texts: [[0.0]])
    monkeypatch.setattr(retriever_module, "SIMILARITY_THRESHOLD", 0.5)

    result = retrieve("some question", FakeManager(fake_results))

    assert len(result) == 1
    assert result[0]["text"] == "on-topic chunk"


def test_retrieve_returns_empty_when_nothing_passes_threshold(monkeypatch):
    fake_results = [{"text": "off-topic", "score": 0.1, "source": "a.pdf", "page": 1}]
    monkeypatch.setattr(retriever_module, "embed_texts", lambda texts: [[0.0]])
    monkeypatch.setattr(retriever_module, "SIMILARITY_THRESHOLD", 0.5)

    result = retrieve("some question", FakeManager(fake_results))

    assert result == []


def test_retrieve_keeps_all_chunks_at_or_above_threshold(monkeypatch):
    fake_results = [
        {"text": "exact threshold", "score": 0.5, "source": "a.pdf", "page": 1},
    ]
    monkeypatch.setattr(retriever_module, "embed_texts", lambda texts: [[0.0]])
    monkeypatch.setattr(retriever_module, "SIMILARITY_THRESHOLD", 0.5)

    result = retrieve("some question", FakeManager(fake_results))

    assert len(result) == 1

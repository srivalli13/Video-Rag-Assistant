"""Unit tests for processing/chunker.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from processing.chunker import chunk_records, group_segments
from utils import config


def test_chunk_records_splits_long_text_with_overlap():
    long_text = "a" * (config.CHUNK_SIZE * 2 + 50)
    records = [{"text": long_text, "page": 1, "source": "doc.pdf"}]

    chunks = chunk_records(records)

    assert len(chunks) > 1
    assert all(c["page"] == 1 and c["source"] == "doc.pdf" for c in chunks)
    assert all(len(c["text"]) <= config.CHUNK_SIZE for c in chunks)


def test_chunk_records_short_text_stays_one_chunk():
    records = [{"text": "short text", "page": 1, "source": "doc.pdf"}]
    chunks = chunk_records(records)
    assert len(chunks) == 1
    assert chunks[0]["text"] == "short text"


def test_chunk_records_skips_empty_text():
    records = [{"text": "   ", "page": 1, "source": "doc.pdf"}]
    chunks = chunk_records(records)
    assert chunks == []


def test_group_segments_merges_until_chunk_size():
    segments = [
        {"text": "a" * 300, "timestamp": 0.0, "source": "lecture.mp4"},
        {"text": "b" * 300, "timestamp": 5.0, "source": "lecture.mp4"},
    ]
    chunks = group_segments(segments)

    # combined length (600) exceeds CHUNK_SIZE (500), so they should merge into one chunk
    assert len(chunks) == 1
    assert chunks[0]["timestamp"] == 0.0  # keeps the FIRST segment's timestamp
    assert "a" * 300 in chunks[0]["text"]
    assert "b" * 300 in chunks[0]["text"]


def test_group_segments_keeps_leftover_tail():
    segments = [{"text": "short", "timestamp": 1.0, "source": "lecture.mp4"}]
    chunks = group_segments(segments)
    assert len(chunks) == 1
    assert chunks[0]["text"] == "short"
    assert chunks[0]["timestamp"] == 1.0

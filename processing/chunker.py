"""Split or group text records into ~CHUNK_SIZE chunks, preserving metadata."""

from utils.config import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_records(records: list[dict]) -> list[dict]:
    """
    Slice large text records (e.g. PDF pages) into overlapping chunks.
    Each chunk inherits its parent's metadata.
    """
    chunks = []
    for record in records:
        text = record["text"]
        metadata = {k: v for k, v in record.items() if k != "text"}

        start = 0
        while start < len(text):
            piece = text[start : start + CHUNK_SIZE].strip()
            if piece:
                chunks.append({"text": piece, **metadata})
            if start + CHUNK_SIZE >= len(text):
                break
            start += CHUNK_SIZE - CHUNK_OVERLAP  # step forward, minus overlap

    return chunks


def group_segments(records: list[dict]) -> list[dict]:
    """
    Merge small timestamped segments (e.g. Whisper output) into
    ~CHUNK_SIZE-char chunks. Each chunk keeps its FIRST segment's timestamp.
    """
    chunks = []
    current_texts = []
    current_start = None
    current_length = 0

    for record in records:
        if current_start is None:
            current_start = record["timestamp"]

        current_texts.append(record["text"])
        current_length += len(record["text"])

        if current_length >= CHUNK_SIZE:
            chunks.append({
                "text": " ".join(current_texts),
                "timestamp": current_start,
                "source": record["source"],
            })
            current_texts = []
            current_start = None
            current_length = 0

    if current_texts:  # leftover tail smaller than CHUNK_SIZE
        chunks.append({
            "text": " ".join(current_texts),
            "timestamp": current_start,
            "source": records[-1]["source"],
        })

    return chunks


if __name__ == "__main__":
    import sys

    path = sys.argv[1]
    if path.lower().endswith(".pdf"):
        from ingest.pdf_ingest import extract_pdf
        chunks = chunk_records(extract_pdf(path))
    else:
        from ingest.video_ingest import transcribe_video
        chunks = group_segments(transcribe_video(path))

    print(f"{len(chunks)} chunks")
    print("--- Chunk 0 ---")
    print(chunks[0])
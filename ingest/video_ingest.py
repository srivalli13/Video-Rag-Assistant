"""Transcribe video/audio to timestamped text segments using Whisper."""

from pathlib import Path

import whisper

from utils.config import WHISPER_MODEL

_model = None  # same lazy-singleton pattern as the embedder


def get_model():
    global _model
    if _model is None:
        print(f"Loading Whisper '{WHISPER_MODEL}' (first time only)...")
        _model = whisper.load_model(WHISPER_MODEL)
    return _model


def transcribe_video(video_path: str) -> list[dict]:
    """
    Return records: {"text": str, "timestamp": float, "source": str}
    timestamp = start of the segment, in seconds.
    """
    video_path = Path(video_path)
    model = get_model()

    print(f"Transcribing {video_path.name} (roughly realtime on CPU)...")
    result = model.transcribe(str(video_path), fp16=False)

    records = []
    for segment in result["segments"]:
        text = segment["text"].strip()
        if not text:
            continue
        records.append({
            "text": text,
            "timestamp": round(segment["start"], 1),
            "source": video_path.name,
        })
    return records


def format_timestamp(seconds: float) -> str:
    """754.2 -> '12:34' for display."""
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes}:{secs:02d}"


if __name__ == "__main__":
    import sys

    records = transcribe_video(sys.argv[1])
    print(f"\n{len(records)} segments")
    for r in records[:5]:
        print(f"[{format_timestamp(r['timestamp'])}] {r['text']}")
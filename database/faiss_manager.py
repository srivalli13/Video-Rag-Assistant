"""FAISS vector index + parallel metadata store, kept in sync."""

import json

import faiss
import numpy as np

from utils.config import INDEX_DIR

INDEX_FILE = INDEX_DIR / "chunks.faiss"
META_FILE = INDEX_DIR / "chunks_meta.json"
DIMENSION = 1024  # BGE-M3 output size


class FaissManager:
    def __init__(self):
        if INDEX_FILE.exists() and META_FILE.exists():
            self.index = faiss.read_index(str(INDEX_FILE))
            self.metadata = json.loads(META_FILE.read_text(encoding="utf-8"))
        else:
            self.index = faiss.IndexFlatIP(DIMENSION)
            self.metadata = []

    def add(self, vectors: np.ndarray, records: list[dict]):
        """Add vectors and their chunk records. Positions must correspond 1:1."""
        assert len(vectors) == len(records), "vectors/records length mismatch"
        self.index.add(np.asarray(vectors, dtype="float32"))
        self.metadata.extend(records)
        self._save()

    def search(self, query_vector: np.ndarray, k: int) -> list[dict]:
        """Return up to k records, each with its similarity score attached."""
        if self.index.ntotal == 0:
            return []
        query = np.asarray([query_vector], dtype="float32")
        scores, positions = self.index.search(query, min(k, self.index.ntotal))

        results = []
        for score, pos in zip(scores[0], positions[0]):
            record = dict(self.metadata[pos])
            record["score"] = float(score)
            results.append(record)
        return results

    def _save(self):
        faiss.write_index(self.index, str(INDEX_FILE))
        META_FILE.write_text(
            json.dumps(self.metadata, ensure_ascii=False), encoding="utf-8"
        )

    @property
    def size(self) -> int:
        return self.index.ntotal


if __name__ == "__main__":
    # End-to-end Day 1 test: PDF -> chunks -> vectors -> FAISS -> semantic search
    import sys
    from ingest.pdf_ingest import extract_pdf
    from processing.chunker import chunk_records
    from embeddings.embedder import embed_texts

    manager = FaissManager()
    if manager.size == 0:
        pages = extract_pdf(sys.argv[1])
        chunks = chunk_records(pages)
        vectors = embed_texts([c["text"] for c in chunks])
        manager.add(vectors, chunks)
    print(f"Index contains {manager.size} vectors")

    query = input("\nAsk something about your PDF: ")
    query_vector = embed_texts([query])[0]
    for r in manager.search(query_vector, k=3):
        print(f"\n[score {r['score']:.3f}] page {r['page']} of {r['source']}")
        print(r["text"][:200])
"""Turn a question into the chunks Gemini is allowed to see."""

from database.faiss_manager import FaissManager
from embeddings.embedder import embed_texts
from utils.config import TOP_K, SIMILARITY_THRESHOLD


def retrieve(question: str, manager: FaissManager) -> list[dict]:
    """
    Embed the question, search the index, and return only chunks
    scoring >= SIMILARITY_THRESHOLD. Empty list = nothing relevant.
    """
    query_vector = embed_texts([question])[0]
    results = manager.search(query_vector, k=TOP_K)
    return [r for r in results if r["score"] >= SIMILARITY_THRESHOLD]


if __name__ == "__main__":
    manager = FaissManager()
    print(f"Index has {manager.size} vectors")

    while True:
        question = input("\nQuestion (or 'quit'): ")
        if question.lower() == "quit":
            break
        chunks = retrieve(question, manager)
        if not chunks:
            print(">> Nothing above threshold - the bot would refuse to answer.")
            continue
        for c in chunks:
            location = f"page {c['page']}" if "page" in c else f"time {c['timestamp']}s"
            print(f"[{c['score']:.3f}] {location} of {c['source']} | {c['text'][:100]}")
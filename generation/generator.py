"""Build a grounded prompt from retrieved chunks and ask Gemini."""

from google import genai

from utils.config import GEMINI_API_KEY, GEMINI_MODEL

_client = None


def get_client():
    global _client
    if _client is None:
        if not GEMINI_API_KEY:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Add it to a .env file in the project root."
            )
        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client


def build_context(chunks: list[dict]) -> str:
    """Turn retrieved chunks into a labeled block of text for the prompt."""
    parts = []
    for c in chunks:
        location = f"page {c['page']}" if "page" in c else f"{c['timestamp']}s"
        parts.append(f"[Source: {c['source']}, {location}]\n{c['text']}")
    return "\n\n".join(parts)


def generate_answer(question: str, chunks: list[dict], history: list[dict] = None) -> str:
    """
    chunks: retrieved, threshold-passed chunks (possibly empty)
    history: list of {"role": "user"/"assistant", "text": ...}
    """
    if not chunks:
        return "I couldn't find relevant information in your uploaded content to answer that."

    context = build_context(chunks)
    history_text = ""
    if history:
        history_text = "\n".join(f"{h['role']}: {h['text']}" for h in history[-4:])

    prompt = f"""You are a study assistant. Answer the question using ONLY the context below.
If the context does not contain the answer, say clearly that you could not find it in the
uploaded materials. Do not use outside knowledge. Cite the source (page or timestamp) you used.

Context:
{context}

Conversation so far:
{history_text}

Question: {question}
Answer:"""

    client = get_client()
    try:
        response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        return response.text
    except Exception as e:
        return f"Gemini request failed: {e}"


if __name__ == "__main__":
    from database.faiss_manager import FaissManager
    from retrieval.retriever import retrieve

    manager = FaissManager()
    while True:
        q = input("\nQuestion (or 'quit'): ")
        if q.lower() == "quit":
            break
        chunks = retrieve(q, manager)
        print("\n" + generate_answer(q, chunks))
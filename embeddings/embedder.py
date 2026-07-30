"""Wrap the BGE-M3 embedding model. Loads once, encodes anything."""

from sentence_transformers import SentenceTransformer

from utils.config import EMBEDDING_MODEL

_model = None  # module-level cache: loaded on first use, reused forever


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"Loading embedding model {EMBEDDING_MODEL} (first time only)...")
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def embed_texts(texts: list[str]):
    """Encode a list of texts into normalized vectors (numpy array, shape [n, 1024])."""
    model = get_model()
    return model.encode(
        texts,
        batch_size=16,
        normalize_embeddings=True,   # unit length -> cosine == dot product
        show_progress_bar=True,
    )


if __name__ == "__main__":
    # Self-test: see a vector, then watch semantic similarity beat keyword matching
    vectors = embed_texts([
        "The professor explained gradient descent optimization.",
        "The teacher described how models minimize loss step by step.",
        "I had paneer for lunch today.",
    ])

    print(f"\nShape: {vectors.shape}")          # expect (3, 1024)
    print(f"First 8 numbers of vector 0:\n{vectors[0][:8]}")

    sim_related = vectors[0] @ vectors[1]        # @ = dot product
    sim_unrelated = vectors[0] @ vectors[2]
    print(f"\nSimilarity gradient-descent vs loss-minimization: {sim_related:.3f}")
    print(f"Similarity gradient-descent vs paneer-lunch:        {sim_unrelated:.3f}")
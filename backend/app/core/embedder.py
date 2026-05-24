"""
Embedding wrapper around sentence-transformers.
Supports separate query vs document embedding with optional prefixes.
Includes embedding cache integration for repeated queries.

Design note: We use a bi-encoder (SentenceTransformer) for fast retrieval.
Models like intfloat/e5-base-v2 use "query: " and "passage: " prefixes
for asymmetric retrieval — the embedder supports this pattern via config.
"""
from sentence_transformers import SentenceTransformer
from app.config import EMBEDDING_MODEL, QUERY_PREFIX, DOCUMENT_PREFIX
from app.core.cache import embedding_cache

# Module-level singleton — loaded once, reused across requests
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """Lazy-load the embedding model (singleton)."""
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def embed_documents(texts: list[str]) -> list[list[float]]:
    """
    Embed a batch of document texts.
    Applies DOCUMENT_PREFIX if configured (e.g., "passage: " for E5 models).
    Returns normalized embeddings as lists of floats.
    """
    model = _get_model()
    prefixed = [f"{DOCUMENT_PREFIX}{t}" for t in texts] if DOCUMENT_PREFIX else texts
    embeddings = model.encode(
        prefixed,
        normalize_embeddings=True,
        show_progress_bar=len(texts) > 100,
        batch_size=64,
    )
    return embeddings.tolist()


def embed_query(text: str) -> list[float]:
    """
    Embed a single query text with cache support.
    Applies QUERY_PREFIX if configured (e.g., "query: " for E5 models).
    Returns a normalized embedding as a list of floats.
    """
    # Check cache first
    cached = embedding_cache.get(text)
    if cached is not None:
        return cached

    model = _get_model()
    prefixed = f"{QUERY_PREFIX}{text}" if QUERY_PREFIX else text
    embedding = model.encode(
        prefixed,
        normalize_embeddings=True,
    )
    result = embedding.tolist()

    # Store in cache
    embedding_cache.put(text, result)
    return result


def get_model_info() -> dict:
    """Return model metadata for health checks."""
    model = _get_model()
    return {
        "model_name": EMBEDDING_MODEL,
        "embedding_dim": model.get_sentence_embedding_dimension(),
        "max_seq_length": model.max_seq_length,
    }

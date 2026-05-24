"""
Cross-encoder reranker for second-stage retrieval precision.

Architecture note: Bi-encoders (used in retrieval) encode query and document
independently — fast but less accurate. Cross-encoders process the
(query, document) pair jointly through a transformer, giving much higher
accuracy at the cost of speed. This is why we rerank only the top-K
candidates from retrieval, not the entire corpus.
"""
from sentence_transformers import CrossEncoder
from app.config import RERANKER_MODEL, RERANK_TOP_N

# Module-level singleton
_reranker: CrossEncoder | None = None


def _get_reranker() -> CrossEncoder:
    """Lazy-load the cross-encoder model (singleton)."""
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANKER_MODEL)
    return _reranker


def rerank(query: str, candidates: list[dict], top_n: int = RERANK_TOP_N) -> list[dict]:
    """
    Rerank candidate documents using the cross-encoder.

    Args:
        query: The user's question
        candidates: List of dicts with at least 'document' key
        top_n: Number of top results to return after reranking

    Returns:
        List of candidates sorted by rerank_score, truncated to top_n.
        Each candidate gets a 'rerank_score' field added.
    """
    if not candidates:
        return []

    model = _get_reranker()

    # Create (query, document) pairs for the cross-encoder
    pairs = [(query, doc["document"]) for doc in candidates]

    # Batch predict scores
    scores = model.predict(pairs)

    # Attach scores to candidates
    for candidate, score in zip(candidates, scores):
        candidate["rerank_score"] = float(score)

    # Sort by rerank score descending
    reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)

    return reranked[:top_n]


def get_reranker_info() -> dict:
    """Return reranker metadata for health checks."""
    return {
        "model_name": RERANKER_MODEL,
        "top_n": RERANK_TOP_N,
    }

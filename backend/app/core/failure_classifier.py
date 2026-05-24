"""
Failure classification for RAG pipeline diagnostics.

V3 final polish:
  - Mutually exclusive categories (strict priority chain)
  - root_cause field: "retrieval" | "ranking" | "generation"
  - Frequency distribution of failure types over time
  - Top recurring root causes tracking
  - Supporting metrics dict for structured logging
"""
from collections import deque
from datetime import datetime, timezone
from app.config import RERANK_SCORE_THRESHOLD


# --- Failure frequency tracker ---
class _FailureTracker:
    """Track failure type distribution over a rolling window."""

    def __init__(self, window_size: int = 200):
        self._window: deque[dict] = deque(maxlen=window_size)

    def record(self, failure_class: str, root_cause: str | None):
        self._window.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "class": failure_class,
            "root_cause": root_cause,
        })

    def get_distribution(self) -> dict:
        """Get frequency distribution of failure types."""
        n = len(self._window)
        if n == 0:
            return {"total": 0, "types": {}, "root_causes": {}}

        types = {}
        root_causes = {}
        for entry in self._window:
            fc = entry["class"]
            rc = entry.get("root_cause")
            types[fc] = types.get(fc, 0) + 1
            if rc:
                root_causes[rc] = root_causes.get(rc, 0) + 1

        # Sort by frequency descending
        types_sorted = dict(sorted(types.items(), key=lambda x: -x[1]))
        root_causes_sorted = dict(sorted(root_causes.items(), key=lambda x: -x[1]))

        return {
            "total": n,
            "types": {k: {"count": v, "rate": round(v / n, 4)} for k, v in types_sorted.items()},
            "root_causes": {k: {"count": v, "rate": round(v / n, 4)} for k, v in root_causes_sorted.items()},
            "top_recurring": list(root_causes_sorted.keys())[:3],
        }


_failure_tracker = _FailureTracker()


def get_failure_distribution() -> dict:
    """Public accessor for failure frequency distribution."""
    return _failure_tracker.get_distribution()


# Failure categories — mutually exclusive, ordered by priority
# Only ONE category is returned per query. First match wins.
FAILURE_CATEGORIES = {
    "retrieval_miss": "retrieval",    # root cause
    "rerank_failure": "ranking",
    "generation_error": "generation",
    "generation_refusal": "generation",
    "success": None,
}


def classify_failure(
    candidates: list[dict],
    reranked_chunks: list[dict],
    generation_output: str,
    generation_success: bool = True,
    query_overlap_ratio: float = 0.0,
    query_variation_count: int = 0,
) -> dict:
    """
    Classify the query outcome based on pipeline signals.
    Categories are MUTUALLY EXCLUSIVE — first match in priority chain wins.

    Returns:
        dict with 'class', 'root_cause', 'reason', and 'metrics'
    """
    metrics = _compute_metrics(candidates, reranked_chunks, generation_output, query_overlap_ratio)

    # --- Priority 1: No candidates at all ---
    if not candidates:
        reason = (
            "No candidates returned from hybrid retrieval. "
            f"Knowledge base returned 0 results across {query_variation_count + 1} query variants. "
            "The query is likely out of scope or the knowledge base is empty."
        )
        return _result("retrieval_miss", reason, metrics)

    # --- Priority 2: All reranked scores below threshold ---
    if reranked_chunks:
        rerank_scores = [c.get("rerank_score", -100) for c in reranked_chunks]
        top_rerank = max(rerank_scores)
        avg_rerank = sum(rerank_scores) / len(rerank_scores)

        if top_rerank < RERANK_SCORE_THRESHOLD:
            overlap_signal = ""
            if query_variation_count > 0:
                if query_overlap_ratio < 0.1:
                    overlap_signal = (
                        f" Cross-query overlap was very low ({query_overlap_ratio:.0%}), "
                        "suggesting query expansions found different (but equally irrelevant) content."
                    )
                else:
                    overlap_signal = (
                        f" Cross-query overlap was {query_overlap_ratio:.0%}, "
                        "indicating consistent retrieval across variants — but all content scored poorly."
                    )

            reason = (
                f"Best rerank score ({top_rerank:.2f}) and average ({avg_rerank:.2f}) "
                f"are below threshold ({RERANK_SCORE_THRESHOLD}). "
                f"Retrieved {len(candidates)} candidates but none were semantically relevant."
                f"{overlap_signal}"
            )
            return _result("retrieval_miss", reason, metrics)

    # --- Priority 3: Candidates exist but reranking eliminated too many ---
    if candidates and len(reranked_chunks) <= 1:
        rrf_scores = [c.get("rrf_score", 0) for c in candidates]
        avg_rrf = sum(rrf_scores) / len(rrf_scores) if rrf_scores else 0
        top_rrf = max(rrf_scores) if rrf_scores else 0
        top_rerank = reranked_chunks[0].get("rerank_score", -100) if reranked_chunks else -100

        if avg_rrf > 0.01 and top_rerank < RERANK_SCORE_THRESHOLD:
            semantic_count = sum(1 for c in candidates if c.get("match_type") == "semantic")
            keyword_count = sum(1 for c in candidates if c.get("match_type") == "keyword")
            hybrid_count = sum(1 for c in candidates if c.get("match_type") == "hybrid")

            reason = (
                f"Retrieval found {len(candidates)} candidates (top RRF: {top_rrf:.4f}, avg: {avg_rrf:.4f}) "
                f"with mix of {semantic_count} semantic, {keyword_count} keyword, {hybrid_count} hybrid matches, "
                f"but reranker scored top result at {top_rerank:.2f} (threshold: {RERANK_SCORE_THRESHOLD}). "
                "Query may be ambiguous or the retrieved content is topically related but doesn't answer the question."
            )
            return _result("rerank_failure", reason, metrics)

    # --- Priority 4: Generation exception ---
    if not generation_success:
        reason = (
            "LLM generation failed with an exception. "
            f"Retrieval succeeded with {len(reranked_chunks)} reranked chunks available. "
            "Check API key, provider status, and rate limits."
        )
        return _result("generation_error", reason, metrics)

    # --- Priority 5: LLM produced fallback/refusal despite good retrieval ---
    fallback_indicators = [
        "I don't have enough information",
        "No LLM available",
        "LLM generation failed",
        "not enough context",
        "Generation error",
    ]
    if any(indicator.lower() in generation_output.lower() for indicator in fallback_indicators):
        top_rerank = max((c.get("rerank_score", -100) for c in reranked_chunks), default=-100)
        reason = (
            f"LLM refused to answer despite having {len(reranked_chunks)} reranked chunks "
            f"(top rerank: {top_rerank:.2f}). "
            "The retrieved context likely doesn't contain a direct answer. "
            "Consider rephrasing the query or indexing more relevant documents."
        )
        return _result("generation_refusal", reason, metrics)

    # --- Success ---
    return _result("success", "", metrics)


def _result(failure_class: str, reason: str, metrics: dict) -> dict:
    """Build a standardized result dict with root_cause. Records to tracker."""
    root_cause = FAILURE_CATEGORIES.get(failure_class)
    _failure_tracker.record(failure_class, root_cause)
    return {
        "class": failure_class,
        "root_cause": root_cause,
        "reason": reason,
        "metrics": metrics,
    }


def _compute_metrics(
    candidates: list[dict],
    reranked_chunks: list[dict],
    generation_output: str,
    query_overlap_ratio: float = 0.0,
) -> dict:
    """Compute structured metrics for logging."""
    rerank_scores = [c.get("rerank_score", 0) for c in reranked_chunks] if reranked_chunks else []
    rrf_scores = [c.get("rrf_score", 0) for c in candidates] if candidates else []

    match_types = {}
    for c in candidates:
        mt = c.get("match_type", "unknown")
        match_types[mt] = match_types.get(mt, 0) + 1

    return {
        "candidate_count": len(candidates),
        "reranked_count": len(reranked_chunks),
        "top_rerank_score": round(max(rerank_scores), 4) if rerank_scores else None,
        "avg_rerank_score": round(sum(rerank_scores) / len(rerank_scores), 4) if rerank_scores else None,
        "rerank_spread": round(max(rerank_scores) - min(rerank_scores), 4) if len(rerank_scores) > 1 else 0.0,
        "top_rrf_score": round(max(rrf_scores), 6) if rrf_scores else None,
        "match_type_distribution": match_types,
        "query_overlap_ratio": round(query_overlap_ratio, 4),
        "answer_length": len(generation_output),
    }

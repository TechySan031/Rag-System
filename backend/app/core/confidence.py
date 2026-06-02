"""
Multi-signal confidence scoring for RAG responses.

V3 final polish:
  - Reliability curve tracking (confidence vs correctness binned stats)
  - Calibration history persistence to disk
  - Bucket distribution (high/medium/low)
  - p25/p50/p75 percentiles
  - Rolling window calibration
"""
import json
import math
from collections import deque
from datetime import datetime, timezone
from app.config import CONFIDENCE_THRESHOLDS, RERANK_SCORE_THRESHOLD, CALIBRATION_HISTORY_DIR

# --- Calibration tracker (rolling window) ---
_calibration_window: deque[float] = deque(maxlen=100)

# --- Reliability curve: confidence vs correctness tracking ---
# Bins: [0.0-0.2), [0.2-0.4), [0.4-0.6), [0.6-0.8), [0.8-1.0]
_reliability_bins: dict[str, dict] = {
    "0.0-0.2": {"total": 0, "correct": 0},
    "0.2-0.4": {"total": 0, "correct": 0},
    "0.4-0.6": {"total": 0, "correct": 0},
    "0.6-0.8": {"total": 0, "correct": 0},
    "0.8-1.0": {"total": 0, "correct": 0},
}
_persist_counter = 0
_PERSIST_INTERVAL = 50  # save calibration every N queries


def _get_bin_key(score: float) -> str:
    """Map a confidence score to its bin key."""
    if score < 0.2:
        return "0.0-0.2"
    elif score < 0.4:
        return "0.2-0.4"
    elif score < 0.6:
        return "0.4-0.6"
    elif score < 0.8:
        return "0.6-0.8"
    else:
        return "0.8-1.0"


def record_correctness(score: float, is_correct: bool):
    """
    Record whether a query with a given confidence was actually correct.
    Used for reliability curve calibration (external call, e.g. from evaluation).
    """
    bin_key = _get_bin_key(score)
    _reliability_bins[bin_key]["total"] += 1
    if is_correct:
        _reliability_bins[bin_key]["correct"] += 1


def get_reliability_curve() -> dict:
    """
    Get the reliability curve: expected calibration per confidence bin.
    Returns {bin_key: {total, correct, accuracy}} for each bin.
    """
    result = {}
    for key, val in _reliability_bins.items():
        total = val["total"]
        correct = val["correct"]
        result[key] = {
            "total": total,
            "correct": correct,
            "accuracy": round(correct / total, 4) if total > 0 else None,
        }
    return result


def compute_confidence(
    reranked_chunks: list[dict],
    candidates: list[dict],
    query_overlap_ratio: float = 0.0,
) -> dict:
    """
    Compute confidence score from multiple signals.

    Returns:
        dict with 'score' (0.0–1.0), 'label', 'signals', 'calibration'
    """
    global _persist_counter

    if not reranked_chunks:
        # Still record in calibration and provide full signal breakdown
        # so the empty-retrieval case is visible in diagnostics
        _calibration_window.append(0.0)
        _persist_counter += 1
        if _persist_counter >= _PERSIST_INTERVAL:
            _persist_calibration()
            _persist_counter = 0

        return {
            "score": 0.0,
            "label": "low",
            "signals": {
                "score_magnitude": 0.0,
                "score_spread": 0.0,
                "retrieval_agreement": 0.0,
                "cross_query_consistency": round(min(max(query_overlap_ratio, 0.0), 1.0), 4),
                "support_ratio": 0.0,
                "reason": "no_reranked_chunks",
            },
            "calibration": _get_calibration(),
        }

    # --- Signal 1: Reranker score magnitude (0.0–1.0) ---
    top_score = reranked_chunks[0].get("rerank_score", -10)
    score_magnitude = _sigmoid_normalize(top_score, midpoint=-4.0, steepness=0.5)

    # --- Signal 2: Score spread (0.0–1.0) ---
    if len(reranked_chunks) > 1:
        last_score = reranked_chunks[-1].get("rerank_score", -10)
        spread = top_score - last_score
        score_spread = min(max(spread / 6.0, 0.0), 1.0)
    else:
        score_spread = 0.5

    # --- Signal 3: Retrieval agreement (0.0–1.0) ---
    hybrid_count = sum(
        1 for c in reranked_chunks
        if c.get("match_type") == "hybrid"
    )
    agreement = hybrid_count / len(reranked_chunks) if reranked_chunks else 0.0

    # --- Signal 4: Cross-query consistency (0.0–1.0) ---
    cross_query = min(max(query_overlap_ratio, 0.0), 1.0)

    # --- Signal 5: Supporting chunk count (0.0–1.0) ---
    positive_chunks = sum(1 for c in reranked_chunks if c.get("rerank_score", -100) > RERANK_SCORE_THRESHOLD)
    support_ratio = positive_chunks / len(reranked_chunks) if reranked_chunks else 0.0

    # --- Weighted combination ---
    weights = {
        "score_magnitude": 0.30,
        "score_spread": 0.20,
        "retrieval_agreement": 0.20,
        "cross_query_consistency": 0.15,
        "support_ratio": 0.15,
    }
    raw_score = (
        weights["score_magnitude"] * score_magnitude
        + weights["score_spread"] * score_spread
        + weights["retrieval_agreement"] * agreement
        + weights["cross_query_consistency"] * cross_query
        + weights["support_ratio"] * support_ratio
    )

    score = round(min(max(raw_score, 0.0), 1.0), 4)

    # --- Calibration: track distribution ---
    _calibration_window.append(score)
    calibration = _get_calibration()

    # --- Periodic persistence ---
    _persist_counter += 1
    if _persist_counter >= _PERSIST_INTERVAL:
        _persist_calibration()
        _persist_counter = 0

    # --- Label ---
    if score >= CONFIDENCE_THRESHOLDS["high"]:
        label = "high"
    elif score >= CONFIDENCE_THRESHOLDS["medium"]:
        label = "medium"
    else:
        label = "low"

    return {
        "score": score,
        "label": label,
        "signals": {
            "score_magnitude": round(score_magnitude, 4),
            "score_spread": round(score_spread, 4),
            "retrieval_agreement": round(agreement, 4),
            "cross_query_consistency": round(cross_query, 4),
            "support_ratio": round(support_ratio, 4),
        },
        "calibration": calibration,
    }


def _sigmoid_normalize(x: float, midpoint: float = 0.0, steepness: float = 1.0) -> float:
    """Map an unbounded value to [0, 1] using a sigmoid curve."""
    return 1.0 / (1.0 + math.exp(-steepness * (x - midpoint)))


def _get_calibration() -> dict:
    """
    Return rolling calibration stats with bucket distribution and reliability curve.
    """
    if not _calibration_window:
        return {"count": 0, "mean": 0.0, "min": 0.0, "max": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0,
                "buckets": {"high": 0, "medium": 0, "low": 0},
                "reliability_curve": get_reliability_curve()}

    scores = sorted(_calibration_window)
    n = len(scores)

    high_count = sum(1 for s in scores if s >= CONFIDENCE_THRESHOLDS["high"])
    medium_count = sum(1 for s in scores if CONFIDENCE_THRESHOLDS["medium"] <= s < CONFIDENCE_THRESHOLDS["high"])
    low_count = n - high_count - medium_count

    return {
        "count": n,
        "mean": round(sum(scores) / n, 4),
        "min": round(scores[0], 4),
        "max": round(scores[-1], 4),
        "p25": round(scores[max(0, n // 4)], 4),
        "p50": round(scores[n // 2], 4),
        "p75": round(scores[min(n - 1, 3 * n // 4)], 4),
        "buckets": {
            "high": high_count,
            "medium": medium_count,
            "low": low_count,
            "high_pct": round(high_count / n, 4) if n else 0,
            "medium_pct": round(medium_count / n, 4) if n else 0,
            "low_pct": round(low_count / n, 4) if n else 0,
        },
        "reliability_curve": get_reliability_curve(),
    }


def _persist_calibration():
    """Save calibration snapshot to disk for historical analysis."""
    try:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        snapshot = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "calibration": _get_calibration(),
        }
        path = CALIBRATION_HISTORY_DIR / f"calibration_{ts}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, ensure_ascii=False, indent=2)
    except Exception:
        pass  # Non-critical


def get_calibration_stats() -> dict:
    """Public accessor for calibration data (for health endpoint)."""
    return _get_calibration()

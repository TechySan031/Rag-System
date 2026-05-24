"""
Rolling metrics aggregation for production monitoring.

V3 final polish:
  - Anomaly flags (is_anomalous: true/false) in metrics output
  - Persist metrics snapshots to disk for historical analysis
  - Drift detection + trend tracking
  - Per-model token pricing
  - p50/p95/p99 latency distributions
  - Fully decoupled from pipeline
"""
import json
import re
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from app.config import (
    DRIFT_CONFIDENCE_DROP_THRESHOLD, DRIFT_FAILURE_SPIKE_THRESHOLD,
    METRICS_HISTORY_DIR, CONFIG_HASH,
)


# --- Per-model pricing ($ per 1K tokens) ---
MODEL_PRICING = {
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-4o": {"input": 0.005, "output": 0.015},
    "gemini-2.0-flash": {"input": 0.0, "output": 0.0},
    "ollama": {"input": 0.0, "output": 0.0},
}

DEFAULT_PRICING = {"input": 0.0, "output": 0.0}


def _get_model_pricing(model: str) -> dict:
    base = model.split("/")[0] if "/" in model else model
    return MODEL_PRICING.get(base, MODEL_PRICING.get(model, DEFAULT_PRICING))


def _percentile(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        return 0.0
    k = (len(sorted_values) - 1) * (p / 100.0)
    f = int(k)
    c = f + 1
    if c >= len(sorted_values):
        return sorted_values[-1]
    return sorted_values[f] + (k - f) * (sorted_values[c] - sorted_values[f])


def _count_citations(text: str) -> int:
    return len(re.findall(r'\[Source:.*?\]', text, re.IGNORECASE))


def _count_sentences(text: str) -> int:
    sentences = re.split(r'[.!?]+', text)
    return max(len([s for s in sentences if s.strip()]), 1)


class MetricsCollector:
    """
    Rolling window metrics with drift detection, anomaly flags, and persistence.
    Independent of pipeline — accepts any dict with the expected fields.
    """

    def __init__(self, window_size: int = 500):
        self._window: deque[dict] = deque(maxlen=window_size)
        self._total_queries = 0
        self._total_tokens = {"input": 0, "output": 0}
        self._total_cost = 0.0
        self._prev_snapshot: dict | None = None
        self._snapshot_interval = max(window_size // 5, 20)
        self._since_snapshot = 0
        self._persist_interval = max(window_size // 2, 50)
        self._since_persist = 0

    def record(self, trace: dict):
        """Record a pipeline trace for metrics computation."""
        self._total_queries += 1

        answer = trace.get("generation_output", "")
        citations = _count_citations(answer)
        sentences = _count_sentences(answer)
        citation_coverage = citations / sentences if sentences > 0 else 0.0

        usage = trace.get("token_usage", {})
        model = usage.get("model", "unknown")
        pricing = _get_model_pricing(model)
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        cost = (prompt_tokens / 1000 * pricing["input"]) + (completion_tokens / 1000 * pricing["output"])

        self._total_tokens["input"] += prompt_tokens
        self._total_tokens["output"] += completion_tokens
        self._total_cost += cost

        entry = {
            "timestamp": trace.get("created_at", datetime.now(timezone.utc).isoformat()),
            "total_latency_ms": trace.get("total_latency_ms", 0),
            "retrieval_latency_ms": trace.get("latency_ms", {}).get("retrieval", 0),
            "generation_latency_ms": trace.get("latency_ms", {}).get("generation", 0),
            "confidence_score": trace.get("confidence_score", 0),
            "failure_class": trace.get("failure_class", "success"),
            "candidate_count": trace.get("candidate_count", 0),
            "citation_coverage": round(citation_coverage, 4),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost": cost,
        }
        self._window.append(entry)

        # Periodic snapshot for drift comparison
        self._since_snapshot += 1
        if self._since_snapshot >= self._snapshot_interval:
            self._prev_snapshot = self._compute_core_metrics()
            self._since_snapshot = 0

        # Periodic persist for historical analysis
        self._since_persist += 1
        if self._since_persist >= self._persist_interval:
            self._persist_snapshot()
            self._since_persist = 0

    def _compute_core_metrics(self) -> dict:
        n = len(self._window)
        if n == 0:
            return {"avg_confidence": 0.0, "failure_rate": 0.0, "avg_latency_ms": 0.0}
        entries = list(self._window)
        avg_conf = sum(e["confidence_score"] for e in entries) / n
        failure_rate = sum(1 for e in entries if e["failure_class"] != "success") / n
        avg_lat = sum(e["total_latency_ms"] for e in entries) / n
        return {
            "avg_confidence": round(avg_conf, 4),
            "failure_rate": round(failure_rate, 4),
            "avg_latency_ms": round(avg_lat, 2),
        }

    def _detect_drift(self, current: dict) -> dict:
        if not self._prev_snapshot:
            return {"status": "no_baseline", "alerts": []}

        alerts = []
        prev = self._prev_snapshot

        conf_drop = prev["avg_confidence"] - current["avg_confidence"]
        if conf_drop > DRIFT_CONFIDENCE_DROP_THRESHOLD:
            alerts.append({
                "type": "confidence_drop",
                "severity": "warning",
                "message": f"Avg confidence dropped {conf_drop:.3f} ({prev['avg_confidence']:.3f} → {current['avg_confidence']:.3f})",
                "previous": prev["avg_confidence"],
                "current": current["avg_confidence"],
                "delta": round(-conf_drop, 4),
            })

        failure_spike = current["failure_rate"] - prev["failure_rate"]
        if failure_spike > DRIFT_FAILURE_SPIKE_THRESHOLD:
            alerts.append({
                "type": "failure_rate_spike",
                "severity": "warning",
                "message": f"Failure rate spiked {failure_spike:.3f} ({prev['failure_rate']:.3f} → {current['failure_rate']:.3f})",
                "previous": prev["failure_rate"],
                "current": current["failure_rate"],
                "delta": round(failure_spike, 4),
            })

        status = "alert" if alerts else "ok"
        return {"status": status, "alerts": alerts}

    def _persist_snapshot(self):
        """Save current metrics snapshot to disk for historical analysis."""
        try:
            metrics = self.get_metrics()
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            path = METRICS_HISTORY_DIR / f"metrics_{ts}.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, ensure_ascii=False, indent=2)
        except Exception:
            pass  # Non-critical — don't crash pipeline for metrics persistence

    def get_metrics(self) -> dict:
        """
        Compute current metrics snapshot from rolling window.
        Includes drift detection, trend comparison, and anomaly flags.
        """
        n = len(self._window)
        if n == 0:
            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "window_size": 0,
                "total_queries": self._total_queries,
                "config_hash": CONFIG_HASH,
            }

        entries = list(self._window)

        total_latencies = sorted(e["total_latency_ms"] for e in entries)
        retrieval_latencies = sorted(e["retrieval_latency_ms"] for e in entries)
        generation_latencies = sorted(e["generation_latency_ms"] for e in entries)

        failure_count = sum(1 for e in entries if e["failure_class"] != "success")
        empty_count = sum(1 for e in entries if e["candidate_count"] == 0)
        avg_confidence = sum(e["confidence_score"] for e in entries) / n
        avg_citation = sum(e["citation_coverage"] for e in entries) / n

        window_prompt = sum(e["prompt_tokens"] for e in entries)
        window_completion = sum(e["completion_tokens"] for e in entries)
        window_cost = sum(e["cost"] for e in entries)

        core = {
            "avg_confidence": round(avg_confidence, 4),
            "failure_rate": round(failure_count / n, 4),
            "avg_latency_ms": round(sum(total_latencies) / n, 2),
        }

        # Drift detection
        drift = self._detect_drift(core)

        # Trend vs previous window
        trend = "stable"
        if self._prev_snapshot:
            delta_conf = core["avg_confidence"] - self._prev_snapshot["avg_confidence"]
            delta_fail = core["failure_rate"] - self._prev_snapshot["failure_rate"]
            if delta_conf > 0.05 and delta_fail < 0:
                trend = "improving"
            elif delta_conf < -0.05 or delta_fail > 0.05:
                trend = "degrading"

        # Anomaly flags
        p99_lat = _percentile(total_latencies, 99)
        is_anomalous = (
            drift["status"] == "alert"
            or core["failure_rate"] > 0.3
            or core["avg_confidence"] < 0.2
            or p99_lat > 300000  # 5 min p99
        )

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "config_hash": CONFIG_HASH,
            "window_size": n,
            "total_queries": self._total_queries,
            "trend": trend,
            "is_anomalous": is_anomalous,
            # Latency
            "avg_latency_ms": core["avg_latency_ms"],
            "p50_latency_ms": round(_percentile(total_latencies, 50), 2),
            "p95_latency_ms": round(_percentile(total_latencies, 95), 2),
            "p99_latency_ms": round(p99_lat, 2),
            # Stage latency
            "avg_retrieval_latency_ms": round(sum(retrieval_latencies) / n, 2),
            "p95_retrieval_latency_ms": round(_percentile(retrieval_latencies, 95), 2),
            "avg_generation_latency_ms": round(sum(generation_latencies) / n, 2),
            "p95_generation_latency_ms": round(_percentile(generation_latencies, 95), 2),
            # Quality
            "avg_confidence": core["avg_confidence"],
            "avg_citation_coverage": round(avg_citation, 4),
            "failure_rate": core["failure_rate"],
            "empty_retrieval_rate": round(empty_count / n, 4),
            # Cost
            "window_prompt_tokens": window_prompt,
            "window_completion_tokens": window_completion,
            "window_cost_usd": round(window_cost, 6),
            "total_cost_usd": round(self._total_cost, 6),
            # Drift
            "drift": drift,
        }


# --- Singleton ---
metrics_collector = MetricsCollector()

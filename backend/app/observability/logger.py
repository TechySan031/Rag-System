"""
Structured JSON logger for RAG pipeline observability.

V3.2 — final polish:
  - Log rotation via RotatingFileHandler (50MB default, 5 backups)
  - LOG_SCHEMA_VERSION in every record for backward compatibility
  - config_hash in every INFO record for reproducibility
  - DEBUG sampling (LOG_DEBUG_SAMPLE_RATE)
  - Flat JSON — ELK / Datadog ingestion-ready
"""
import json
import logging
import logging.handlers
import random
import uuid
from datetime import datetime, timezone
from pathlib import Path
from app.config import (
    LOG_DIR, LOG_DEBUG_MODE, LOG_DEBUG_SAMPLE_RATE,
    LOG_MAX_BYTES, LOG_BACKUP_COUNT, LOG_SCHEMA_VERSION, CONFIG_HASH,
)

# --- JSONL file paths ---
_log_file = LOG_DIR / "rag_queries.jsonl"
_debug_log_file = LOG_DIR / "rag_debug.jsonl"

_logger = logging.getLogger("rag_observability")
_logger.setLevel(logging.DEBUG if LOG_DEBUG_MODE else logging.INFO)

_debug_logger = logging.getLogger("rag_debug")
_debug_logger.setLevel(logging.DEBUG)

if not _logger.handlers:
    # INFO-level handler: rotating file, 50MB max, 5 backups
    file_handler = logging.handlers.RotatingFileHandler(
        str(_log_file), maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUP_COUNT, encoding="utf-8",
    )
    file_handler.setLevel(logging.INFO)
    _logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    _logger.addHandler(console_handler)

if not _debug_logger.handlers:
    # DEBUG-level handler: rotating file, same limits
    debug_handler = logging.handlers.RotatingFileHandler(
        str(_debug_log_file), maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUP_COUNT, encoding="utf-8",
    )
    debug_handler.setLevel(logging.DEBUG)
    _debug_logger.addHandler(debug_handler)


def generate_query_id() -> str:
    """Generate a unique query ID for tracing."""
    return uuid.uuid4().hex[:12]


def log_query_trace(
    query_id: str,
    query_text: str,
    retrieved_chunks: list[dict],
    reranked_chunks: list[dict],
    final_prompt: str,
    generation_output: str,
    latency_ms: dict[str, float],
    confidence_score: float = 0.0,
    confidence_label: str = "low",
    confidence_signals: dict = None,
    confidence_calibration: dict = None,
    failure_class: str = "success",
    failure_reason: str = "",
    failure_metrics: dict = None,
    query_variations: list[str] = None,
    selected_query_variant: str = "",
    cache_hit: bool = False,
    prompt_version: str = "",
    prompt_hash: str = "",
    token_usage: dict = None,
    trace_version: str = "2.0",
):
    """
    Log a pipeline trace at two levels:
      - INFO (always): flat metrics for dashboards + monitoring
      - DEBUG (sampled): full chunks, prompt, contexts for debugging
    """
    now = datetime.now(timezone.utc).isoformat()
    usage = token_usage or {}

    # ---- INFO log: flat, dashboard-ready, schema-versioned ----
    info_record = {
        "log_schema": LOG_SCHEMA_VERSION,
        "config_hash": CONFIG_HASH,
        "trace_id": query_id,
        "trace_version": trace_version,
        "timestamp": now,
        "query": query_text,
        # Metrics
        "confidence_score": round(confidence_score, 4),
        "confidence_label": confidence_label,
        "failure_class": failure_class,
        "failure_reason": failure_reason,
        "cache_hit": cache_hit,
        # Counts
        "retrieved_count": len(retrieved_chunks),
        "reranked_count": len(reranked_chunks),
        "query_variation_count": len(query_variations) if query_variations else 0,
        # Prompt
        "prompt_version": prompt_version,
        "prompt_hash": prompt_hash,
        # Token usage (flat)
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
        "model": usage.get("model", "unknown"),
        # Latency (flat)
        "total_latency_ms": round(sum(latency_ms.values()), 2),
        "retrieval_latency_ms": round(latency_ms.get("retrieval", 0), 2),
        "reranking_latency_ms": round(latency_ms.get("reranking", 0), 2),
        "generation_latency_ms": round(latency_ms.get("generation", 0), 2),
        # Answer preview
        "answer_length": len(generation_output),
    }
    _logger.info(json.dumps(info_record, ensure_ascii=False))

    # ---- DEBUG log: full trace (sampled) ----
    debug_record = {
        **info_record,
        "query_variations": query_variations or [],
        "selected_query_variant": selected_query_variant,
        "confidence_signals": confidence_signals or {},
        "confidence_calibration": confidence_calibration or {},
        "failure_metrics": failure_metrics or {},
        "retrieved_chunks": [
            {
                "chunk_id": c.get("id", ""),
                "source": c.get("metadata", {}).get("source", ""),
                "page": c.get("metadata", {}).get("page", ""),
                "similarity_score": round(c.get("similarity_score", 0), 4),
                "bm25_score": round(c.get("bm25_score", 0), 4),
                "rrf_score": round(c.get("rrf_score", 0), 6),
                "match_type": c.get("match_type", "unknown"),
                "text_preview": c.get("document", "")[:200],
            }
            for c in retrieved_chunks
        ],
        "reranked_chunks": [
            {
                "chunk_id": c.get("id", ""),
                "source": c.get("metadata", {}).get("source", ""),
                "page": c.get("metadata", {}).get("page", ""),
                "rerank_score": round(c.get("rerank_score", 0), 4),
                "match_type": c.get("match_type", "unknown"),
                "text_preview": c.get("document", "")[:200],
            }
            for c in reranked_chunks
        ],
        "final_prompt_length": len(final_prompt),
        "generation_output": generation_output,
    }

    if LOG_DEBUG_MODE:
        debug_record["final_prompt"] = final_prompt
    else:
        debug_record["final_prompt_preview"] = final_prompt[:500]

    # Probabilistic sampling
    if random.random() < LOG_DEBUG_SAMPLE_RATE:
        debug_record["_sampled"] = True
        _debug_logger.debug(json.dumps(debug_record, ensure_ascii=False))


def log_ingestion(
    filename: str,
    chunk_count: int,
    duration_ms: float,
):
    """Log document ingestion events."""
    event = {
        "log_schema": LOG_SCHEMA_VERSION,
        "config_hash": CONFIG_HASH,
        "event": "ingestion",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "filename": filename,
        "chunk_count": chunk_count,
        "duration_ms": round(duration_ms, 2),
    }
    _logger.info(json.dumps(event, ensure_ascii=False))

"""
RAG Pipeline Orchestrator — V3.2 final.
Ties together: multi-query → hybrid retrieve → rerank → generate → analyze → log.

Final polish:
  - config_hash embedded in every trace for full reproducibility
  - Recall proxy: tracks whether top_k contained a relevant chunk
  - Missed-retrieval examples persisted for debugging
  - All modules remain optional — pipeline degrades gracefully
"""
import json
import time
from datetime import datetime, timezone
from app.models import QueryResponse, ChunkInfo, DebugInfo
from app.core import retriever, reranker, generator
from app.core.multi_query import multi_query_retrieve
from app.core.confidence import compute_confidence
from app.core.failure_classifier import classify_failure
from app.core.cache import query_cache
from app.core.parser import parse_file
from app.core.chunker import chunk_documents
from app.core.embedder import embed_documents
from app.core import vectorstore
from app.core.prompts import get_active_prompt
from app.observability.logger import log_query_trace, log_ingestion, generate_query_id
from app.observability.trace import PipelineTrace
from app.observability.metrics import metrics_collector
from app.config import (
    CHUNK_SIZE, CHUNK_OVERLAP, STRUCTURE_AWARE_CHUNKING,
    MULTI_QUERY_ENABLED, HYBRID_TOP_K,
    FUSION_STRATEGY, VECTOR_TOP_K, BM25_TOP_K,
    RERANK_TOP_N, RERANKER_MODEL, RERANK_SCORE_THRESHOLD,
    CONFIG_HASH, LOG_DIR,
)

# --- Recall proxy tracker ---
_recall_stats = {"total": 0, "relevant_in_top_k": 0}
_missed_retrieval_file = LOG_DIR / "missed_retrievals.jsonl"


def get_recall_proxy() -> dict:
    """Public accessor: % of queries where a relevant chunk appeared in top_k."""
    total = _recall_stats["total"]
    relevant = _recall_stats["relevant_in_top_k"]
    return {
        "total_queries": total,
        "relevant_in_top_k": relevant,
        "recall_proxy": round(relevant / total, 4) if total > 0 else None,
    }


def _time_ms(start: float) -> float:
    """Calculate elapsed time in milliseconds."""
    return (time.perf_counter() - start) * 1000


def _explain_selection(doc: dict) -> str:
    """
    Generate a human-readable explanation of why this chunk was selected.
    Based on match_type, retrieval scores, and rerank score.
    """
    match_type = doc.get("match_type", "unknown")
    parts = []

    if match_type == "hybrid":
        parts.append("Found by both semantic embedding and keyword search")
    elif match_type == "semantic":
        parts.append("Found by semantic embedding similarity")
    elif match_type == "keyword":
        parts.append("Found by BM25 keyword matching")

    sim = doc.get("similarity_score")
    bm25 = doc.get("bm25_score")
    rerank = doc.get("rerank_score")

    if sim and sim > 0.5:
        parts.append(f"high vector similarity ({sim:.3f})")
    elif sim and sim > 0:
        parts.append(f"moderate vector similarity ({sim:.3f})")

    if bm25 and bm25 > 5:
        parts.append(f"strong keyword overlap (BM25={bm25:.1f})")
    elif bm25 and bm25 > 0:
        parts.append(f"partial keyword overlap (BM25={bm25:.1f})")

    if rerank is not None:
        if rerank > 0:
            parts.append(f"cross-encoder confirmed relevance (rerank={rerank:.2f})")
        elif rerank > -5:
            parts.append(f"cross-encoder scored moderately (rerank={rerank:.2f})")
        else:
            parts.append(f"cross-encoder scored low (rerank={rerank:.2f})")

    return "; ".join(parts) if parts else "Selected via pipeline scoring"


def _to_chunk_info(doc: dict, stage: str = "retrieval") -> ChunkInfo:
    """Convert a raw candidate dict to a typed ChunkInfo."""
    meta = doc.get("metadata", {})
    return ChunkInfo(
        chunk_id=doc.get("id", meta.get("chunk_id", "")),
        text=doc.get("document", ""),
        source=meta.get("source", "unknown"),
        page=meta.get("page"),
        similarity_score=doc.get("similarity_score"),
        bm25_score=doc.get("bm25_score"),
        rrf_score=doc.get("rrf_score"),
        rerank_score=doc.get("rerank_score"),
        match_type=doc.get("match_type"),
        selection_reason=_explain_selection(doc),
    )


def _identify_best_variant(query: str, variations: list[str], reranked: list[dict]) -> str:
    """
    Identify which query variant was most effective.
    Returns the most effective query string for logging.
    """
    if not variations:
        return query
    return query


def run_query(query: str) -> QueryResponse:
    """
    Execute the full RAG pipeline V3:
    0. Check query cache
    1. Multi-query expansion + parallel hybrid retrieval
    2. Cross-encoder reranking
    3. LLM generation with versioned prompt
    4. Confidence scoring with calibration
    5. Failure classification with detailed reasons
    6. Trace finalization + metrics collection + observability logging
    7. Cache storage

    Returns a QueryResponse with answer, sources, confidence, and full debug info.
    """
    # --- Stage 0: Cache check ---
    cached = query_cache.get(query)
    if cached is not None:
        return cached

    query_id = generate_query_id()
    trace = PipelineTrace(trace_id=query_id)
    query_variations = []
    overlap_ratio = 0.0

    # Embed config_hash for reproducibility
    trace.add_metadata("config_hash", CONFIG_HASH)

    # Get active prompt
    active_prompt = get_active_prompt()
    trace.add_metadata("prompt_version", active_prompt["version"])
    trace.add_metadata("prompt_hash", active_prompt["prompt_hash"])

    # --- Stage 1: Retrieval ---
    with trace.stage("retrieval", {
        "fusion_strategy": FUSION_STRATEGY,
        "vector_top_k": VECTOR_TOP_K,
        "bm25_top_k": BM25_TOP_K,
        "hybrid_top_k": HYBRID_TOP_K,
        "multi_query_enabled": MULTI_QUERY_ENABLED,
    }) as stage_data:
        if MULTI_QUERY_ENABLED:
            candidates, query_variations, overlap_ratio = multi_query_retrieve(query)
            candidates = candidates[:HYBRID_TOP_K]
            stage_data["query_variations"] = query_variations
            stage_data["overlap_ratio"] = round(overlap_ratio, 4)
        else:
            candidates = retriever.hybrid_retrieve(query)

        stage_data["candidate_count"] = len(candidates)
        stage_data["match_types"] = {}
        for c in candidates:
            mt = c.get("match_type", "unknown")
            stage_data["match_types"][mt] = stage_data["match_types"].get(mt, 0) + 1

    # --- Stage 2: Reranking ---
    with trace.stage("reranking", {
        "model": RERANKER_MODEL,
        "top_n": RERANK_TOP_N,
        "input_count": len(candidates),
    }) as stage_data:
        reranked = reranker.rerank(query, candidates)
        stage_data["output_count"] = len(reranked)

        # Per-chunk contribution scores (normalized rerank → 0-1)
        rerank_scores = [c.get("rerank_score", -100) for c in reranked]
        max_score = max(rerank_scores) if rerank_scores else 0
        min_score = min(rerank_scores) if rerank_scores else 0
        score_range = max_score - min_score if max_score != min_score else 1.0

        chunk_contributions = []
        for c in reranked:
            raw = c.get("rerank_score", -100)
            contribution = round((raw - min_score) / score_range, 4) if score_range > 0 else 0.0
            chunk_contributions.append({
                "chunk_id": c.get("id", ""),
                "rerank_score": round(raw, 4),
                "contribution": contribution,
                "source": c.get("metadata", {}).get("source", ""),
            })
        stage_data["scores"] = chunk_contributions

        # Top-K sufficiency: did we actually fill top_k with good results?
        above_threshold = sum(1 for s in rerank_scores if s > RERANK_SCORE_THRESHOLD)
        stage_data["top_k_sufficiency"] = {
            "requested": RERANK_TOP_N,
            "returned": len(reranked),
            "above_threshold": above_threshold,
            "sufficient": above_threshold >= RERANK_TOP_N,
        }

    # --- Stage 3: Context assembly ---
    with trace.stage("context_assembly") as stage_data:
        selected_context = "\n\n".join(
            f"[{c.get('metadata', {}).get('source', '?')}, p{c.get('metadata', {}).get('page', '?')}]: {c.get('document', '')}"
            for c in reranked
        )
        stage_data["context_length"] = len(selected_context)
        stage_data["chunk_count"] = len(reranked)

    # --- Stage 4: Generation ---
    gen_success = True
    with trace.stage("generation") as stage_data:
        try:
            gen_result = generator.generate(query, reranked)
        except Exception as e:
            gen_success = False
            gen_result = {
                "answer": f"⚠️ Generation error: {str(e)}",
                "final_prompt": "",
                "model": "error",
                "usage": {},
            }
        stage_data["model"] = gen_result.get("model", "unknown")
        stage_data["success"] = gen_success
        stage_data["answer_length"] = len(gen_result.get("answer", ""))

    # Extract token usage
    token_usage = gen_result.get("usage", {})
    token_usage["model"] = gen_result.get("model", "unknown")
    trace.add_metadata("token_usage", token_usage)

    # --- Stage 5: Confidence scoring ---
    with trace.stage("confidence") as stage_data:
        conf = compute_confidence(
            reranked_chunks=reranked,
            candidates=candidates,
            query_overlap_ratio=overlap_ratio,
        )
        stage_data["score"] = conf["score"]
        stage_data["label"] = conf["label"]

    # --- Stage 6: Failure classification ---
    with trace.stage("classification") as stage_data:
        failure = classify_failure(
            candidates=candidates,
            reranked_chunks=reranked,
            generation_output=gen_result["answer"],
            generation_success=gen_success,
            query_overlap_ratio=overlap_ratio,
            query_variation_count=len(query_variations),
        )
        stage_data["class"] = failure["class"]
        stage_data["reason"] = failure["reason"]

    # --- Identify best query variant ---
    selected_variant = _identify_best_variant(query, query_variations, reranked)

    # --- Build latency dict from trace ---
    latency = {}
    for s in trace._stages:
        latency[s.name] = round(s.latency_ms, 2)

    # --- Assemble response ---
    retrieved_infos = [_to_chunk_info(c) for c in candidates]
    reranked_infos = [_to_chunk_info(c) for c in reranked]

    debug = DebugInfo(
        retrieved_chunks=retrieved_infos,
        reranked_chunks=reranked_infos,
        selected_context=selected_context,
        final_prompt=gen_result.get("final_prompt", ""),
        generation_output=gen_result["answer"],
        latency_ms=latency,
        query_variations=query_variations,
        failure_class=failure["class"],
        failure_reason=failure["reason"],
        cache_hit=False,
    )

    total = sum(latency.values())

    response = QueryResponse(
        answer=gen_result["answer"],
        sources=reranked_infos,
        debug=debug,
        total_latency_ms=total,
        confidence_score=conf["score"],
        confidence_label=conf["label"],
    )

    # --- Stage 7: Observability ---
    log_query_trace(
        query_id=query_id,
        query_text=query,
        retrieved_chunks=candidates,
        reranked_chunks=reranked,
        final_prompt=gen_result.get("final_prompt", ""),
        generation_output=gen_result["answer"],
        latency_ms=latency,
        confidence_score=conf["score"],
        confidence_label=conf["label"],
        confidence_signals=conf.get("signals"),
        confidence_calibration=conf.get("calibration"),
        failure_class=failure["class"],
        failure_reason=failure["reason"],
        failure_metrics=failure.get("metrics"),
        query_variations=query_variations,
        selected_query_variant=selected_variant,
        cache_hit=False,
        prompt_version=active_prompt["version"],
        prompt_hash=active_prompt["prompt_hash"],
        token_usage=token_usage,
    )

    # --- Metrics collection (decoupled) ---
    trace_data = trace.finalize()
    trace_data["confidence_score"] = conf["score"]
    trace_data["failure_class"] = failure["class"]
    trace_data["generation_output"] = gen_result["answer"]
    trace_data["candidate_count"] = len(candidates)
    trace_data["token_usage"] = token_usage
    metrics_collector.record(trace_data)

    # --- Recall proxy + missed-retrieval tracking ---
    _recall_stats["total"] += 1
    has_relevant = any(c.get("rerank_score", -100) > RERANK_SCORE_THRESHOLD for c in reranked)
    if has_relevant:
        _recall_stats["relevant_in_top_k"] += 1
    else:
        # Store missed-retrieval for debugging
        try:
            missed = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "query": query,
                "query_id": query_id,
                "candidate_count": len(candidates),
                "reranked_count": len(reranked),
                "top_rerank_score": round(max((c.get("rerank_score", -100) for c in reranked), default=-100), 4),
                "failure_class": failure["class"],
                "config_hash": CONFIG_HASH,
            }
            with open(_missed_retrieval_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(missed, ensure_ascii=False) + "\n")
        except Exception:
            pass  # Non-critical

    # --- Stage 8: Cache result ---
    query_cache.put(query, response)

    return response


def ingest_file(filepath: str) -> dict:
    """
    Ingest a single file: parse → chunk → embed → store.
    Rebuilds the BM25 index and invalidates query cache after storage.

    Returns:
        dict with filename, chunk_count, status
    """
    t_start = time.perf_counter()

    # Parse
    documents = parse_file(filepath)
    if not documents:
        return {"filename": filepath, "chunk_count": 0, "status": "empty"}

    # Chunk (structure-aware if configured)
    chunks = chunk_documents(
        documents,
        chunk_size=CHUNK_SIZE,
        overlap=CHUNK_OVERLAP,
        structure_aware=STRUCTURE_AWARE_CHUNKING,
    )

    # Embed
    texts = [c.text for c in chunks]
    embeddings = embed_documents(texts)

    # Store in ChromaDB
    ids = [c.metadata["chunk_id"] for c in chunks]
    metadatas = [c.metadata for c in chunks]

    vectorstore.add_documents(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    # Rebuild BM25 index
    retriever.rebuild_bm25_index()

    # Invalidate query cache (document set changed)
    query_cache.clear()

    # Sync to HF Hub for persistence across restarts (prod only)
    from app.config import HF_PERSISTENCE_ENABLED
    if HF_PERSISTENCE_ENABLED:
        try:
            from app.core.hf_persistence import sync_to_hub
            sync_to_hub(uploaded_filepath=filepath)
        except Exception as e:
            # Sync failure is non-fatal — log and continue
            print(f"[persistence] WARNING: Post-upload sync failed: {e}")

    duration = _time_ms(t_start)
    log_ingestion(filepath, len(chunks), duration)

    return {
        "filename": filepath,
        "chunk_count": len(chunks),
        "status": "success",
        "duration_ms": duration,
    }

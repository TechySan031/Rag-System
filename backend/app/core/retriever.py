"""
Hybrid retriever combining BM25 (lexical) and vector (semantic) search.
Supports multiple fusion strategies: RRF and Weighted Sum.
Annotates each result with match_type for debugging visibility.

Architecture:
  1. Vector search via ChromaDB (cosine similarity)
  2. BM25 search via rank_bm25 (lexical matching)
  3. Fusion: RRF (default) or weighted normalization
  4. Deduplication before returning to reranker
"""
from rank_bm25 import BM25Okapi
from app.config import (
    VECTOR_TOP_K, BM25_TOP_K, RRF_K, HYBRID_TOP_K,
    FUSION_STRATEGY, BM25_WEIGHT, VECTOR_WEIGHT,
)
from app.core import vectorstore, embedder

# In-memory BM25 index — rebuilt on upload
_bm25_index: BM25Okapi | None = None
_bm25_corpus: list[dict] | None = None  # [{id, document, metadata}, ...]


def rebuild_bm25_index():
    """
    Rebuild the BM25 index from all documents in ChromaDB.
    Called after each document upload.
    """
    global _bm25_index, _bm25_corpus

    all_docs = vectorstore.get_all_documents()
    if not all_docs["documents"]:
        _bm25_index = None
        _bm25_corpus = None
        return

    _bm25_corpus = [
        {"id": id_, "document": doc, "metadata": meta}
        for id_, doc, meta in zip(
            all_docs["ids"], all_docs["documents"], all_docs["metadatas"]
        )
    ]

    # Tokenize for BM25 (simple whitespace + lowercase)
    tokenized = [doc["document"].lower().split() for doc in _bm25_corpus]
    _bm25_index = BM25Okapi(tokenized)


def _vector_search(query: str, top_k: int = VECTOR_TOP_K) -> list[dict]:
    """Run vector similarity search via ChromaDB."""
    query_embedding = embedder.embed_query(query)
    results = vectorstore.query_by_embedding(query_embedding, top_k=top_k)

    candidates = []
    for i, (id_, doc, meta, dist) in enumerate(zip(
        results["ids"], results["documents"],
        results["metadatas"], results["distances"]
    )):
        candidates.append({
            "id": id_,
            "document": doc,
            "metadata": meta,
            "similarity_score": 1.0 - dist,  # cosine distance → similarity
            "vector_rank": i + 1,
        })
    return candidates


def _bm25_search(query: str, top_k: int = BM25_TOP_K) -> list[dict]:
    """Run BM25 lexical search over the in-memory index."""
    if _bm25_index is None or _bm25_corpus is None:
        return []

    tokenized_query = query.lower().split()
    scores = _bm25_index.get_scores(tokenized_query)

    # Get top-K indices sorted by score descending
    scored_indices = sorted(
        enumerate(scores), key=lambda x: x[1], reverse=True
    )[:top_k]

    candidates = []
    for rank, (idx, score) in enumerate(scored_indices):
        if score <= 0:
            continue
        doc = _bm25_corpus[idx]
        candidates.append({
            "id": doc["id"],
            "document": doc["document"],
            "metadata": doc["metadata"],
            "bm25_score": float(score),
            "bm25_rank": rank + 1,
        })
    return candidates


def _reciprocal_rank_fusion(
    vector_results: list[dict],
    bm25_results: list[dict],
    k: int = RRF_K,
) -> list[dict]:
    """
    Merge rankings using Reciprocal Rank Fusion.
    RRF_score(d) = sum(1 / (k + rank(d, method))) for each method.
    Annotates each result with match_type.
    """
    fused = {}
    vector_ids = set()
    bm25_ids = set()

    for doc in vector_results:
        doc_id = doc["id"]
        vector_ids.add(doc_id)
        fused[doc_id] = {
            **doc,
            "rrf_score": 1.0 / (k + doc["vector_rank"]),
        }

    for doc in bm25_results:
        doc_id = doc["id"]
        bm25_ids.add(doc_id)
        if doc_id in fused:
            fused[doc_id]["rrf_score"] += 1.0 / (k + doc["bm25_rank"])
            fused[doc_id]["bm25_score"] = doc.get("bm25_score", 0)
            fused[doc_id]["bm25_rank"] = doc.get("bm25_rank")
        else:
            fused[doc_id] = {
                **doc,
                "rrf_score": 1.0 / (k + doc["bm25_rank"]),
                "similarity_score": 0.0,
                "vector_rank": None,
            }

    # Annotate match_type
    for doc_id, doc in fused.items():
        if doc_id in vector_ids and doc_id in bm25_ids:
            doc["match_type"] = "hybrid"
        elif doc_id in vector_ids:
            doc["match_type"] = "semantic"
        else:
            doc["match_type"] = "keyword"

    sorted_results = sorted(
        fused.values(), key=lambda x: x["rrf_score"], reverse=True
    )
    return sorted_results[:HYBRID_TOP_K]


def _weighted_fusion(
    vector_results: list[dict],
    bm25_results: list[dict],
) -> list[dict]:
    """
    Merge using weighted score normalization.
    Normalizes both score sets to [0,1] and combines with configurable weights.
    """
    fused = {}
    vector_ids = set()
    bm25_ids = set()

    # Normalize vector scores to [0, 1]
    vec_scores = [d.get("similarity_score", 0) for d in vector_results]
    vec_min = min(vec_scores) if vec_scores else 0
    vec_max = max(vec_scores) if vec_scores else 1
    vec_range = vec_max - vec_min if vec_max != vec_min else 1.0

    # Normalize BM25 scores to [0, 1]
    bm25_scores = [d.get("bm25_score", 0) for d in bm25_results]
    bm25_min = min(bm25_scores) if bm25_scores else 0
    bm25_max = max(bm25_scores) if bm25_scores else 1
    bm25_range = bm25_max - bm25_min if bm25_max != bm25_min else 1.0

    for doc in vector_results:
        doc_id = doc["id"]
        vector_ids.add(doc_id)
        norm_score = (doc.get("similarity_score", 0) - vec_min) / vec_range
        fused[doc_id] = {
            **doc,
            "rrf_score": VECTOR_WEIGHT * norm_score,  # reuse rrf_score field
        }

    for doc in bm25_results:
        doc_id = doc["id"]
        bm25_ids.add(doc_id)
        norm_score = (doc.get("bm25_score", 0) - bm25_min) / bm25_range
        if doc_id in fused:
            fused[doc_id]["rrf_score"] += BM25_WEIGHT * norm_score
            fused[doc_id]["bm25_score"] = doc.get("bm25_score", 0)
        else:
            fused[doc_id] = {
                **doc,
                "rrf_score": BM25_WEIGHT * norm_score,
                "similarity_score": 0.0,
            }

    # Annotate match_type
    for doc_id, doc in fused.items():
        if doc_id in vector_ids and doc_id in bm25_ids:
            doc["match_type"] = "hybrid"
        elif doc_id in vector_ids:
            doc["match_type"] = "semantic"
        else:
            doc["match_type"] = "keyword"

    sorted_results = sorted(
        fused.values(), key=lambda x: x["rrf_score"], reverse=True
    )
    return sorted_results[:HYBRID_TOP_K]


def hybrid_retrieve(query: str, strategy: str = FUSION_STRATEGY) -> list[dict]:
    """
    Run hybrid retrieval: vector + BM25, fused with configurable strategy.

    Args:
        query: The user's question
        strategy: "rrf" (default) or "weighted_sum"

    Returns a list of candidate documents sorted by fused score,
    each annotated with match_type.
    """
    vector_results = _vector_search(query, top_k=VECTOR_TOP_K)
    bm25_results = _bm25_search(query, top_k=BM25_TOP_K)

    if strategy == "weighted_sum":
        fused = _weighted_fusion(vector_results, bm25_results)
    else:
        fused = _reciprocal_rank_fusion(vector_results, bm25_results)

    return fused

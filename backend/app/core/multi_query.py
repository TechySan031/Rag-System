"""
Multi-query retrieval for improved recall.

Uses the active LLM to generate query variations, runs hybrid retrieval
for each in parallel, and merges results with weighted deduplication.

Design decisions:
  - Original query gets HIGHER weight (2x) than expansions
  - Low-quality expansions are filtered by semantic similarity to original
  - Near-duplicate expansions are removed via inter-query similarity
  - Retrieval runs in parallel threads to control latency
  - Results are deduplicated by chunk_id before reranking
  - Falls back to single-query if LLM is unavailable
"""
import os
import openai
from concurrent.futures import ThreadPoolExecutor, as_completed
from app.config import (
    MULTI_QUERY_COUNT,
    MULTI_QUERY_WEIGHT_ORIGINAL,
    MULTI_QUERY_WEIGHT_EXPANSION,
    OPENAI_API_KEY,
)
from app.core import retriever, embedder

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")

_EXPANSION_PROMPT = """Generate {count} alternative search queries for the following question.
Each query should approach the topic from a different angle to maximize retrieval coverage.
Return ONLY the queries, one per line, no numbering or bullet points.

Original question: {query}"""


def _get_llm_client() -> tuple[openai.OpenAI, str] | None:
    """Get an OpenAI-compatible client for query expansion."""
    try:
        # Try Ollama first (local, free)
        client = openai.OpenAI(api_key="ollama", base_url=OLLAMA_BASE_URL)
        client.models.list()
        return client, OLLAMA_MODEL
    except Exception:
        pass

    if GEMINI_API_KEY:
        client = openai.OpenAI(
            api_key=GEMINI_API_KEY,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
        return client, "gemini-2.0-flash"

    if OPENAI_API_KEY:
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        return client, "gpt-4o-mini"

    return None


def generate_query_variations(query: str, count: int = MULTI_QUERY_COUNT) -> list[str]:
    """
    Generate alternative query phrasings using the active LLM.
    Filters out low-quality and near-duplicate expansions.
    Returns list of unique expansion queries (excluding original).
    """
    llm = _get_llm_client()
    if llm is None:
        return []

    client, model = llm

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": _EXPANSION_PROMPT.format(count=count, query=query),
            }],
            temperature=0.7,
            max_tokens=256,
        )

        raw = response.choices[0].message.content.strip()
        variations = [
            line.strip().lstrip("0123456789.-) ")
            for line in raw.split("\n")
            if line.strip() and len(line.strip()) > 5
        ]

        # Stage 1: Filter semantically distant expansions
        variations = _filter_expansions(query, variations)

        # Stage 2: Remove near-duplicate expansions (inter-query dedup)
        variations = _deduplicate_expansions(variations, threshold=0.85)

        return variations[:count]

    except Exception:
        return []


def _filter_expansions(
    original: str,
    expansions: list[str],
    min_similarity: float = 0.3,
) -> list[str]:
    """
    Remove low-quality expansions that are too far from the original query.
    Uses embedding similarity as a quality gate.
    """
    if not expansions:
        return []

    try:
        original_emb = embedder.embed_query(original)
        expansion_embs = [embedder.embed_query(q) for q in expansions]

        filtered = []
        for exp, emb in zip(expansions, expansion_embs):
            sim = _cosine_similarity(original_emb, emb)
            if sim >= min_similarity:
                filtered.append(exp)
        return filtered
    except Exception:
        return expansions  # on failure, keep all


def _deduplicate_expansions(
    expansions: list[str],
    threshold: float = 0.85,
) -> list[str]:
    """
    Remove near-duplicate queries from expansion list.
    If two queries have cosine similarity > threshold, keep only the first.
    This ensures each expansion adds unique retrieval signal.
    """
    if len(expansions) <= 1:
        return expansions

    try:
        embs = [embedder.embed_query(q) for q in expansions]
        unique = [expansions[0]]
        unique_embs = [embs[0]]

        for i in range(1, len(expansions)):
            is_duplicate = False
            for u_emb in unique_embs:
                if _cosine_similarity(embs[i], u_emb) > threshold:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique.append(expansions[i])
                unique_embs.append(embs[i])

        return unique
    except Exception:
        return expansions


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    import math
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _retrieve_for_query(query: str) -> list[dict]:
    """Run hybrid retrieval for a single query (thread target)."""
    return retriever.hybrid_retrieve(query)


def multi_query_retrieve(query: str) -> tuple[list[dict], list[str], float]:
    """
    Run multi-query retrieval with parallel execution:
    1. Generate query variations via LLM
    2. Run hybrid_retrieve for each in parallel threads
    3. Merge with weighted deduplication
    4. Compute cross-query overlap ratio

    Returns:
        (merged_candidates, query_variations, overlap_ratio)
    """
    variations = generate_query_variations(query)

    # All queries: original (weighted higher) + expansions
    all_queries = [(query, MULTI_QUERY_WEIGHT_ORIGINAL)]
    for v in variations:
        all_queries.append((v, MULTI_QUERY_WEIGHT_EXPANSION))

    # --- Parallel retrieval using ThreadPoolExecutor ---
    query_results: list[tuple[str, float, list[dict]]] = []

    with ThreadPoolExecutor(max_workers=len(all_queries)) as executor:
        futures = {}
        for q, weight in all_queries:
            future = executor.submit(_retrieve_for_query, q)
            futures[future] = (q, weight)

        for future in as_completed(futures):
            q, weight = futures[future]
            try:
                candidates = future.result(timeout=30)
                query_results.append((q, weight, candidates))
            except Exception:
                # Skip failed retrievals
                pass

    # --- Weighted merge with deduplication ---
    all_results: dict[str, dict] = {}  # chunk_id → merged info
    chunk_query_counts: dict[str, int] = {}  # chunk_id → how many queries found it

    for q, weight, candidates in query_results:
        for c in candidates:
            cid = c.get("id", "")
            if cid in all_results:
                # Sum weighted RRF scores
                all_results[cid]["rrf_score"] += c.get("rrf_score", 0) * weight
                # Keep the higher of each score
                for key in ["similarity_score", "bm25_score", "rerank_score"]:
                    existing = all_results[cid].get(key) or 0
                    new_val = c.get(key) or 0
                    all_results[cid][key] = max(existing, new_val)
                # Keep best match_type (hybrid > semantic > keyword)
                if c.get("match_type") == "hybrid" or all_results[cid].get("match_type") == "hybrid":
                    all_results[cid]["match_type"] = "hybrid"
                chunk_query_counts[cid] = chunk_query_counts.get(cid, 0) + 1
            else:
                all_results[cid] = {**c}
                all_results[cid]["rrf_score"] = c.get("rrf_score", 0) * weight
                chunk_query_counts[cid] = 1

    # Sort by accumulated weighted RRF score
    merged = sorted(all_results.values(), key=lambda x: x.get("rrf_score", 0), reverse=True)

    # Compute overlap ratio: fraction of chunks found by multiple queries
    total_queries = len(all_queries)
    if total_queries > 1 and chunk_query_counts:
        multi_hit = sum(1 for count in chunk_query_counts.values() if count > 1)
        overlap_ratio = multi_hit / len(chunk_query_counts)
    else:
        overlap_ratio = 0.0

    return merged, variations, overlap_ratio

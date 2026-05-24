"""
In-memory caching for embeddings and query results.
Uses LRU eviction with TTL-based expiry for query cache.

Design:
  - EmbeddingCache: hash(text) → vector. Avoids re-computing embeddings.
  - QueryCache: hash(query + retrieval_config) → full pipeline response.
    Includes retrieval configuration in cache key to avoid stale/mismatched
    results when config changes (fusion strategy, top_k, etc.).
  - Both use simple dict + deque for O(1) eviction without external deps.
"""
import hashlib
import time
from collections import OrderedDict
from app.config import (
    CACHE_MAX_SIZE, CACHE_TTL_SECONDS,
    FUSION_STRATEGY, VECTOR_TOP_K, BM25_TOP_K,
    HYBRID_TOP_K, RERANK_TOP_N, MULTI_QUERY_ENABLED,
)


def _retrieval_config_hash() -> str:
    """Hash the current retrieval configuration to include in cache keys."""
    config_str = (
        f"fusion={FUSION_STRATEGY}"
        f"|vtop={VECTOR_TOP_K}"
        f"|btop={BM25_TOP_K}"
        f"|htop={HYBRID_TOP_K}"
        f"|rerank_n={RERANK_TOP_N}"
        f"|mq={MULTI_QUERY_ENABLED}"
    )
    return hashlib.sha256(config_str.encode()).hexdigest()[:8]


def _hash_key(text: str, include_config: bool = False) -> str:
    """Create a deterministic hash key from text, optionally including config."""
    if include_config:
        text = f"{text}|cfg:{_retrieval_config_hash()}"
    return hashlib.sha256(text.encode()).hexdigest()[:16]


class EmbeddingCache:
    """
    LRU cache for embedding vectors.
    No TTL — embeddings are deterministic for the same model.
    """

    def __init__(self, max_size: int = CACHE_MAX_SIZE):
        self._cache: OrderedDict[str, list[float]] = OrderedDict()
        self._max_size = max_size
        self.hits = 0
        self.misses = 0

    def get(self, text: str) -> list[float] | None:
        key = _hash_key(text)
        if key in self._cache:
            self._cache.move_to_end(key)
            self.hits += 1
            return self._cache[key]
        self.misses += 1
        return None

    def put(self, text: str, embedding: list[float]):
        key = _hash_key(text)
        self._cache[key] = embedding
        self._cache.move_to_end(key)
        if len(self._cache) > self._max_size:
            self._cache.popitem(last=False)

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "size": len(self._cache),
            "max_size": self._max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 4) if total > 0 else 0.0,
        }


class QueryCache:
    """
    LRU cache for full query responses with TTL expiry.
    Cache key includes retrieval configuration to prevent stale results
    when pipeline settings change (fusion strategy, top_k, etc.).
    Invalidated when documents change (call clear() after upload).
    """

    def __init__(self, max_size: int = CACHE_MAX_SIZE, ttl: int = CACHE_TTL_SECONDS):
        self._cache: OrderedDict[str, tuple[float, dict]] = OrderedDict()
        self._max_size = max_size
        self._ttl = ttl
        self.hits = 0
        self.misses = 0

    def get(self, query: str) -> dict | None:
        key = _hash_key(query, include_config=True)
        if key in self._cache:
            timestamp, result = self._cache[key]
            if time.time() - timestamp < self._ttl:
                self._cache.move_to_end(key)
                self.hits += 1
                return result
            else:
                # Expired entry
                del self._cache[key]
        self.misses += 1
        return None

    def put(self, query: str, result: dict):
        key = _hash_key(query, include_config=True)
        self._cache[key] = (time.time(), result)
        self._cache.move_to_end(key)
        if len(self._cache) > self._max_size:
            self._cache.popitem(last=False)

    def clear(self):
        """Invalidate all entries (call after document upload)."""
        self._cache.clear()

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "size": len(self._cache),
            "max_size": self._max_size,
            "ttl_seconds": self._ttl,
            "config_hash": _retrieval_config_hash(),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 4) if total > 0 else 0.0,
        }


# --- Module-level singletons ---
embedding_cache = EmbeddingCache()
query_cache = QueryCache()

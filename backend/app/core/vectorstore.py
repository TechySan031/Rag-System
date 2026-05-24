"""
ChromaDB vector store operations.
Handles document storage, querying, and collection management.
"""
from __future__ import annotations
import chromadb
from app.config import CHROMA_DIR, COLLECTION_NAME, DISTANCE_METRIC

# Module-level singleton
_client: chromadb.PersistentClient | None = None
_collection: chromadb.Collection | None = None


def _get_collection() -> chromadb.Collection:
    """Get or create the ChromaDB collection (singleton)."""
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = _client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": DISTANCE_METRIC},
        )
    return _collection


def add_documents(
    ids: list[str],
    documents: list[str],
    embeddings: list[list[float]],
    metadatas: list[dict],
) -> int:
    """
    Add documents to ChromaDB with embeddings and metadata.
    Uses upsert to handle re-ingestion of the same document.
    Returns the number of documents added.
    """
    collection = _get_collection()

    # ChromaDB has a batch limit, process in chunks of 5000
    batch_size = 5000
    total = 0

    for i in range(0, len(ids), batch_size):
        batch_end = min(i + batch_size, len(ids))
        collection.upsert(
            ids=ids[i:batch_end],
            documents=documents[i:batch_end],
            embeddings=embeddings[i:batch_end],
            metadatas=metadatas[i:batch_end],
        )
        total += batch_end - i

    return total


def query_by_embedding(
    embedding: list[float],
    top_k: int = 20,
) -> dict:
    """
    Query ChromaDB by embedding vector.
    Returns dict with keys: ids, documents, metadatas, distances.
    """
    collection = _get_collection()
    results = collection.query(
        query_embeddings=[embedding],
        n_results=min(top_k, collection.count() or top_k),
        include=["documents", "metadatas", "distances"],
    )
    return {
        "ids": results["ids"][0] if results["ids"] else [],
        "documents": results["documents"][0] if results["documents"] else [],
        "metadatas": results["metadatas"][0] if results["metadatas"] else [],
        "distances": results["distances"][0] if results["distances"] else [],
    }


def get_all_documents() -> dict:
    """
    Retrieve all documents from the collection.
    Used for building the BM25 index.
    """
    collection = _get_collection()
    count = collection.count()

    if count == 0:
        return {"ids": [], "documents": [], "metadatas": []}

    results = collection.get(
        include=["documents", "metadatas"],
    )
    return {
        "ids": results["ids"],
        "documents": results["documents"],
        "metadatas": results["metadatas"],
    }


def get_collection_stats() -> dict:
    """Return collection statistics for health checks."""
    collection = _get_collection()
    return {
        "collection_name": COLLECTION_NAME,
        "document_count": collection.count(),
        "distance_metric": DISTANCE_METRIC,
    }


def reset_collection():
    """Delete and recreate the collection. Use for testing only."""
    global _collection
    client = _get_collection()  # ensure client is initialized
    if _client is not None:
        _client.delete_collection(COLLECTION_NAME)
        _collection = None
        _get_collection()  # recreate

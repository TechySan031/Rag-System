"""
Pydantic models for API request/response schemas.
Typed contracts between frontend and backend.
"""
from pydantic import BaseModel, Field
from typing import Optional


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="The user's question")


class ChunkInfo(BaseModel):
    """A single retrieved chunk with its metadata and scores."""
    chunk_id: str
    text: str
    source: str
    page: Optional[int] = None
    similarity_score: Optional[float] = None
    bm25_score: Optional[float] = None
    rrf_score: Optional[float] = None
    rerank_score: Optional[float] = None
    match_type: Optional[str] = None     # "semantic", "keyword", or "hybrid"
    selection_reason: Optional[str] = None  # human-readable why this chunk was selected


class DebugInfo(BaseModel):
    """Full pipeline debug information for observability."""
    retrieved_chunks: list[ChunkInfo] = []
    reranked_chunks: list[ChunkInfo] = []
    selected_context: str = ""
    final_prompt: str = ""
    generation_output: str = ""
    latency_ms: dict[str, float] = {}
    query_variations: list[str] = []     # multi-query expansions
    failure_class: str = "success"       # success | retrieval_miss | rerank_failure | generation_issue
    failure_reason: str = ""             # human-readable failure explanation
    cache_hit: bool = False              # whether query cache was used


class QueryResponse(BaseModel):
    answer: str
    sources: list[ChunkInfo] = []
    debug: DebugInfo
    total_latency_ms: float = 0.0
    confidence_score: float = 0.0        # 0.0 to 1.0
    confidence_label: str = "low"        # "high", "medium", "low"


class UploadResponse(BaseModel):
    filename: str
    original_filename: str
    chunk_count: int
    status: str = "success"
    message: str = ""


class HealthResponse(BaseModel):
    status: str = "healthy"
    document_count: int = 0
    collection_name: str = ""
    models_loaded: dict[str, object] = {}
    persistence: dict[str, object] = {}

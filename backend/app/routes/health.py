"""
Health endpoint — system status and readiness check.
"""
from fastapi import APIRouter
from app.models import HealthResponse
from app.core.vectorstore import get_collection_stats
from app.core.generator import get_active_provider

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health():
    """
    Returns system health status, document count, and model availability.
    """
    try:
        stats = get_collection_stats()
        llm_provider = get_active_provider()
        return HealthResponse(
            status="healthy",
            document_count=stats["document_count"],
            collection_name=stats["collection_name"],
            models_loaded={
                "embedder": True,
                "reranker": True,
                "llm": llm_provider != "none",
                "llm_provider": llm_provider,
            },
        )
    except Exception as e:
        return HealthResponse(
            status=f"degraded: {str(e)}",
            document_count=0,
            collection_name="",
            models_loaded={},
        )

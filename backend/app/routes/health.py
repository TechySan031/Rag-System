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
    Returns system health status, document count, model availability,
    and HF Hub persistence sync status.
    """
    try:
        stats = get_collection_stats()
        llm_provider = get_active_provider()

        # Persistence status
        persistence_info = {}
        from app.config import HF_PERSISTENCE_ENABLED
        if HF_PERSISTENCE_ENABLED:
            try:
                from app.core.hf_persistence import get_sync_status
                persistence_info = get_sync_status()
            except Exception:
                persistence_info = {"enabled": True, "status": "import_error"}
        else:
            persistence_info = {"enabled": False}

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
            persistence=persistence_info,
        )
    except Exception as e:
        return HealthResponse(
            status=f"degraded: {str(e)}",
            document_count=0,
            collection_name="",
            models_loaded={},
            persistence={},
        )

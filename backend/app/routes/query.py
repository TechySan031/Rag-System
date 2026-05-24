"""
Query endpoint — handles RAG query requests.
"""
from fastapi import APIRouter, HTTPException
from app.models import QueryRequest, QueryResponse
from app.core.pipeline import run_query

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Execute a RAG query against the ingested documents.
    Returns the generated answer, source chunks, and full debug info.
    """
    try:
        result = run_query(request.query)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

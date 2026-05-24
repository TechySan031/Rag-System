"""
Upload endpoint — handles document ingestion via file upload.
"""
import uuid
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.models import UploadResponse
from app.config import UPLOAD_DIR, MAX_FILE_SIZE_MB, ALLOWED_EXTENSIONS
from app.core.pipeline import ingest_file

router = APIRouter()


@router.post("/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...)):
    """
    Upload a PDF, Markdown, or text file for ingestion into the RAG system.
    The file is parsed, chunked, embedded, and stored in ChromaDB.
    """
    # Validate extension
    original_name = file.filename or "unknown"
    suffix = Path(original_name).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {suffix}. Allowed: {ALLOWED_EXTENSIONS}",
        )

    # Read and validate size
    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large: {size_mb:.1f}MB. Max: {MAX_FILE_SIZE_MB}MB",
        )

    # Save with sanitized filename
    safe_name = f"{uuid.uuid4().hex[:8]}_{original_name}"
    file_path = UPLOAD_DIR / safe_name
    file_path.write_bytes(content)

    # Run ingestion pipeline
    try:
        result = ingest_file(str(file_path))
    except Exception as e:
        # Clean up on failure
        file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

    return UploadResponse(
        filename=safe_name,
        original_filename=original_name,
        chunk_count=result["chunk_count"],
        status=result["status"],
        message=f"Ingested {result['chunk_count']} chunks in {result.get('duration_ms', 0):.0f}ms",
    )

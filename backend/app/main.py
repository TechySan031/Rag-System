"""
FastAPI application entry point.
Configures CORS, registers routes, and handles startup/shutdown.

In production (HF Spaces / Docker), also serves the built React frontend
from /static so the entire app runs as a single container.
"""
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from app.config import CORS_ORIGINS
from app.routes import query, upload, health
from app.core.retriever import rebuild_bm25_index

# Path to built React frontend (populated in Docker build)
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup/shutdown lifecycle.
    - On startup: restore data from HF Hub (if enabled), then rebuild BM25 index
    - On shutdown: final sync to HF Hub (if enabled)
    """
    # Startup: restore persisted data from HF Hub (before BM25 rebuild)
    from app.config import HF_PERSISTENCE_ENABLED
    if HF_PERSISTENCE_ENABLED:
        from app.core.hf_persistence import restore_from_hub
        print("[startup] Restoring data from HF Hub...")
        result = restore_from_hub()
        print(f"[startup] Restore result: {result.get('status', 'unknown')}")

    # Rebuild BM25 index from (possibly restored) ChromaDB data
    print("[startup] Rebuilding BM25 index from ChromaDB...")
    rebuild_bm25_index()
    print("[startup] BM25 index ready.")

    if STATIC_DIR.exists():
        print(f"[startup] Serving frontend from {STATIC_DIR}")
    else:
        print("[startup] No static frontend found (dev mode — use Vite dev server)")

    yield
    # Shutdown: final sync to ensure latest state is persisted
    if HF_PERSISTENCE_ENABLED:
        from app.core.hf_persistence import sync_to_hub
        print("[shutdown] Final sync to HF Hub...")
        sync_to_hub()
    print("[shutdown] RAG system shutting down.")


app = FastAPI(
    title="RAG System API",
    description="Production-grade Retrieval-Augmented Generation system with hybrid search, reranking, and observability.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API route modules
app.include_router(query.router, tags=["Query"])
app.include_router(upload.router, tags=["Upload"])
app.include_router(health.router, tags=["Health"])

# --- Serve built React frontend (production only) ---
if STATIC_DIR.exists():
    # Serve static assets (JS, CSS, images)
    app.mount("/assets", StaticFiles(directory=str(STATIC_DIR / "assets")), name="assets")

    # SPA fallback: any non-API route serves index.html
    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        """Serve React SPA — all non-API routes return index.html."""
        file_path = STATIC_DIR / full_path
        if file_path.is_file():
            return FileResponse(str(file_path))
        return FileResponse(str(STATIC_DIR / "index.html"))


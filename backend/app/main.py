"""
FastAPI application entry point.
Configures CORS, registers routes, and handles startup/shutdown.
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import CORS_ORIGINS
from app.routes import query, upload, health
from app.core.retriever import rebuild_bm25_index


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup/shutdown lifecycle.
    - On startup: rebuild BM25 index from existing ChromaDB data
    - On shutdown: cleanup (if needed)
    """
    # Startup: rebuild BM25 index from persisted ChromaDB documents
    print("[startup] Rebuilding BM25 index from ChromaDB...")
    rebuild_bm25_index()
    print("[startup] BM25 index ready.")
    yield
    # Shutdown
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

# Register route modules
app.include_router(query.router, tags=["Query"])
app.include_router(upload.router, tags=["Upload"])
app.include_router(health.router, tags=["Health"])

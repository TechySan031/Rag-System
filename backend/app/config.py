"""
Configuration constants for the RAG system.
All tuneable parameters centralized here.

Supports environment-based overrides:
  RAG_ENV=prod → stricter thresholds, sampling enabled
  RAG_ENV=dev  → relaxed thresholds, full debug logging (default)

config_hash: deterministic hash of all tunable parameters for reproducibility.
Every trace + evaluation run embeds config_hash so you can reproduce from logs alone.
"""
import os
import hashlib
import json
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from backend directory
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# --- Environment ---
RAG_ENV = os.getenv("RAG_ENV", "dev")  # "dev" or "prod"

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
CHROMA_DIR = BASE_DIR / "chroma_data"
LOG_DIR = BASE_DIR / "logs"
REGRESSION_HISTORY_DIR = LOG_DIR / "regression_history"
METRICS_HISTORY_DIR = LOG_DIR / "metrics_history"
CALIBRATION_HISTORY_DIR = LOG_DIR / "calibration_history"

# Create dirs on import
UPLOAD_DIR.mkdir(exist_ok=True)
CHROMA_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)
REGRESSION_HISTORY_DIR.mkdir(exist_ok=True)
METRICS_HISTORY_DIR.mkdir(exist_ok=True)
CALIBRATION_HISTORY_DIR.mkdir(exist_ok=True)

# --- Chunking ---
CHUNK_SIZE = 512          # characters per chunk
CHUNK_OVERLAP = 64        # overlap between consecutive chunks
STRUCTURE_AWARE_CHUNKING = True   # use heading-based splitting when possible

# --- Embedding ---
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
QUERY_PREFIX = ""
DOCUMENT_PREFIX = ""

# --- Vector Store ---
COLLECTION_NAME = "rag_documents"
DISTANCE_METRIC = "cosine"   # ChromaDB distance function

# --- Retrieval ---
VECTOR_TOP_K = 10          # candidates from vector search
BM25_TOP_K = 10            # candidates from BM25 search
HYBRID_TOP_K = 15          # union candidates for reranking
RRF_K = 60                 # RRF constant
BM25_WEIGHT = 0.4          # weight for BM25 in weighted fusion
VECTOR_WEIGHT = 0.6        # weight for vector in weighted fusion
FUSION_STRATEGY = "rrf"    # "rrf" or "weighted_sum"

# --- Multi-Query ---
MULTI_QUERY_ENABLED = True
MULTI_QUERY_COUNT = 3      # number of query variations to generate
MULTI_QUERY_WEIGHT_ORIGINAL = 2.0   # weight multiplier for original query results
MULTI_QUERY_WEIGHT_EXPANSION = 1.0  # weight for expanded queries

# --- Reranker ---
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_TOP_N = 5           # final chunks after reranking

# --- Confidence ---
CONFIDENCE_THRESHOLDS = {
    "high": 0.7,       # confidence >= 0.7 → "high"
    "medium": 0.4,     # confidence >= 0.4 → "medium"
    # below 0.4 → "low"
}
RERANK_SCORE_THRESHOLD = -8.0   # below this → retrieval_miss
RERANK_SPREAD_THRESHOLD = 2.0   # min spread between top-1 and top-N for confidence

# --- Cache ---
CACHE_MAX_SIZE = 256        # max entries in each cache
CACHE_TTL_SECONDS = 300     # query cache TTL (5 minutes)

# --- Generation ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0.1      # low temp for grounded answers
LLM_MAX_TOKENS = 1024
LLM_TIMEOUT_SECONDS = int(os.getenv("LLM_TIMEOUT_SECONDS", "120"))
LLM_CIRCUIT_BREAKER_THRESHOLD = int(os.getenv("LLM_CIRCUIT_BREAKER_THRESHOLD", "3"))
LLM_CIRCUIT_BREAKER_COOLDOWN = int(os.getenv("LLM_CIRCUIT_BREAKER_COOLDOWN", "60"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "2"))
LLM_MAX_CONCURRENCY = int(os.getenv("LLM_MAX_CONCURRENCY", "4"))

SYSTEM_PROMPT = """You are a precise question-answering assistant. Follow these rules strictly:

1. Answer ONLY based on the provided context below. Do not use prior knowledge.
2. Cite your sources using the format [Source: <filename>, Page: <page>] after each claim.
3. If the context does not contain enough information to answer the question, respond with:
   "I don't have enough information in the provided documents to answer this question."
4. Be concise and factual. Do not speculate or add information not present in the context.
5. If multiple sources support a claim, cite all of them."""

# --- Prompt Versioning ---
PROMPT_VERSION = "v1"     # active prompt version (see core/prompts.py registry)

# --- Regression Gating ---
REGRESSION_THRESHOLDS_HARD = {
    "faithfulness": 0.5,
    "answer_relevancy": 0.5,
    "context_precision": 0.4,
    "context_recall": 0.4,
}
REGRESSION_THRESHOLDS_SOFT = {
    "faithfulness": 0.65,
    "answer_relevancy": 0.65,
    "context_precision": 0.55,
    "context_recall": 0.55,
}
REGRESSION_MIN_SAMPLES = 10   # min samples before gating can produce PASS/FAIL

# --- Evaluation Paths ---
EVAL_DATASET_PATH = Path(__file__).parent / "evaluation" / "dataset.json"
BASELINE_PATH = BASE_DIR / "evaluation" / "baseline.json"

# --- Logging ---
LOG_DEBUG_MODE = os.getenv("RAG_LOG_DEBUG", "true" if RAG_ENV == "dev" else "false").lower() == "true"
LOG_DEBUG_SAMPLE_RATE = float(os.getenv("LOG_DEBUG_SAMPLE_RATE", "1.0" if RAG_ENV == "dev" else "0.1"))
LOG_MAX_BYTES = int(os.getenv("LOG_MAX_BYTES", str(50 * 1024 * 1024)))  # 50MB default
LOG_BACKUP_COUNT = int(os.getenv("LOG_BACKUP_COUNT", "5"))
LOG_SCHEMA_VERSION = "2.0"   # bump when log schema changes — for backward compat

# --- Drift Detection ---
DRIFT_CONFIDENCE_DROP_THRESHOLD = 0.15   # alert if avg confidence drops by this much vs prev window
DRIFT_FAILURE_SPIKE_THRESHOLD = 0.10     # alert if failure rate increases by this much vs prev window

# --- API ---
_cors_env = os.getenv("CORS_ORIGINS", "")
CORS_ORIGINS = _cors_env.split(",") if _cors_env else ["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"]
MAX_FILE_SIZE_MB = 50
ALLOWED_EXTENSIONS = {".pdf", ".md", ".markdown", ".txt"}

# --- OpenTelemetry ---
OTEL_SERVICE_NAME = os.getenv("OTEL_SERVICE_NAME", "rag-pipeline")
OTEL_SERVICE_VERSION = os.getenv("OTEL_SERVICE_VERSION", "3.2")


# ---------------------------------------------------------------------------
# Config hash — deterministic fingerprint of all tunable parameters.
# Embedded in every trace + regression run for full reproducibility.
# ---------------------------------------------------------------------------
def _compute_config_hash() -> str:
    """Compute a deterministic hash of all tunable config values."""
    config_snapshot = {
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "structure_aware": STRUCTURE_AWARE_CHUNKING,
        "embedding_model": EMBEDDING_MODEL,
        "vector_top_k": VECTOR_TOP_K,
        "bm25_top_k": BM25_TOP_K,
        "hybrid_top_k": HYBRID_TOP_K,
        "rrf_k": RRF_K,
        "fusion_strategy": FUSION_STRATEGY,
        "multi_query_enabled": MULTI_QUERY_ENABLED,
        "multi_query_count": MULTI_QUERY_COUNT,
        "reranker_model": RERANKER_MODEL,
        "rerank_top_n": RERANK_TOP_N,
        "rerank_score_threshold": RERANK_SCORE_THRESHOLD,
        "confidence_thresholds": CONFIDENCE_THRESHOLDS,
        "llm_model": LLM_MODEL,
        "llm_temperature": LLM_TEMPERATURE,
        "llm_max_tokens": LLM_MAX_TOKENS,
        "prompt_version": PROMPT_VERSION,
    }
    raw = json.dumps(config_snapshot, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


CONFIG_HASH = _compute_config_hash()

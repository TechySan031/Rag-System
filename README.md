# Production-Grade RAG System

A complete Retrieval-Augmented Generation system with multi-query retrieval, hybrid search, cross-encoder reranking, confidence scoring, failure classification, grounded generation, evaluation, and full observability — built for AI engineering interviews.

## Architecture

```
Query → Multi-Query Expansion → [BM25 + Vector Search] × N → Weighted Fusion → Dedup
     → Cross-Encoder Rerank → Grounded LLM Generation → Confidence Scoring
     → Failure Classification → Structured JSONL Logging → Cache
```

**Stack:** FastAPI · ChromaDB · sentence-transformers · cross-encoder · Ollama/Gemini/OpenAI · React · Vite

## Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+
- Ollama (free, local) — or Gemini/OpenAI API key

### Backend
```bash
cd backend
pip install -r requirements.txt

# Option A: Local LLM (recommended, free)
ollama pull qwen2.5:3b

# Option B: API keys in .env
echo "GEMINI_API_KEY=your-key" > .env
# echo "OPENAI_API_KEY=your-key" >> .env

uvicorn app.main:app --host 127.0.0.1 --port 8000
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173 — upload a PDF/MD file, ask questions, and inspect the debug panel.

## API Endpoints

| Method | Endpoint   | Description |
|--------|-----------|-------------|
| POST   | `/query`  | Run RAG query with full debug info |
| POST   | `/upload` | Upload and ingest a document |
| GET    | `/health` | System status, doc count, LLM provider |

### Query Response Schema
```json
{
  "answer": "The exam assessment...",
  "confidence_score": 0.49,
  "confidence_label": "medium",
  "sources": [
    {
      "source": "syllabus.pdf",
      "page": 11,
      "match_type": "semantic",
      "rerank_score": -2.16,
      "selection_reason": "Found by semantic embedding similarity; high vector similarity (0.604)"
    }
  ],
  "debug": {
    "query_variations": ["What types of questions...", "How do examiners..."],
    "failure_class": "success",
    "failure_reason": "",
    "latency_ms": { "retrieval": 8200, "reranking": 450, "generation": 28000 }
  }
}
```

## Pipeline Stages

### 1. Ingestion
- Parse PDF (PyMuPDF), Markdown, or plain text
- Extract metadata: source filename, page number, file type

### 2. Chunking (Structure-Aware)
- **Structure-aware mode** (default): splits on markdown headers and section boundaries first, then applies fixed-size within sections. Preserves heading context in chunk metadata.
- **Fixed-size fallback**: 512 chars with 64-char overlap and sentence boundary detection
- Deterministic chunk IDs for deduplication
- Backward-compatible: `structure_aware=False` reverts to original behavior

### 3. Embeddings (with Cache)
- `all-MiniLM-L6-v2` via sentence-transformers
- **Embedding cache**: LRU cache avoids re-computing embeddings for repeated queries
- Separate `embed_query()` and `embed_documents()` paths
- Supports asymmetric prefixes (e.g., E5 models)

### 4. Vector Store
- ChromaDB with persistent storage
- Cosine distance metric
- Upsert for safe re-ingestion

### 5. Multi-Query Retrieval
- **Query expansion**: LLM generates 3 alternative query phrasings
- **Semantic filtering**: removes expansions with <0.3 cosine similarity to original
- **Inter-query deduplication**: removes near-duplicate expansions (>0.85 similarity)
- **Parallel retrieval**: `ThreadPoolExecutor` runs all queries concurrently
- **Weighted merge**: original query gets 2x weight, results deduplicated by chunk_id
- Falls back to single-query if LLM unavailable

### 6. Hybrid Retrieval (Configurable Fusion)
- **Vector search**: top-10 cosine similarity from ChromaDB
- **BM25 search**: top-10 lexical match via rank_bm25
- **Fusion strategies** (configurable via `FUSION_STRATEGY`):
  - `rrf` (default): Reciprocal Rank Fusion — rank-based, no score normalization needed
  - `weighted_sum`: Normalized scores combined with configurable weights (0.6 vector / 0.4 BM25)
- **Match-type annotation**: each candidate tagged as `semantic`, `keyword`, or `hybrid`

### 7. Reranking
- Cross-encoder: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Reranks top-15 hybrid candidates → selects top-5
- Joint query-document attention for higher precision

### 8. Generation (Multi-Provider)
- **Provider priority**: Ollama (local/free) → Gemini → OpenAI
- Enforces: citations, context-only answers, "I don't know" fallback
- Low temperature (0.1) for factual responses
- Graceful fallback with retrieval results if all providers fail

### 9. Confidence Scoring
Multi-signal confidence score (0.0–1.0) computed from:

| Signal | Weight | Description |
|--------|--------|-------------|
| Score magnitude | 30% | Sigmoid-normalized top rerank score |
| Score spread | 20% | Gap between best and worst reranked chunk |
| Retrieval agreement | 20% | Fraction of chunks found by both BM25 and vector |
| Cross-query consistency | 15% | Overlap ratio across multi-query expansions |
| Support ratio | 15% | Fraction of chunks above rerank threshold |

**Calibration tracking**: rolling window (last 100 queries) tracks mean, min, max, and p50 for threshold tuning.

### 10. Failure Classification
Each query is classified based on measurable pipeline signals:

| Class | Trigger | Example Reason |
|-------|---------|----------------|
| `success` | Normal operation | — |
| `retrieval_miss` | All rerank scores below threshold | "Best rerank score (-9.2) below threshold (-8.0). Cross-query overlap was 5%." |
| `rerank_failure` | High RRF but low rerank scores | "Retrieval found 15 candidates (avg RRF: 0.03) but reranker scored top at -9.5." |
| `generation_issue` | LLM failure or fallback answer | "LLM refused despite 5 reranked chunks (top rerank: -2.1)." |

### 11. Caching
- **Embedding cache**: LRU (256 entries), no TTL (deterministic for same model)
- **Query cache**: LRU (256 entries), 5-min TTL, **config-aware keys** (includes fusion strategy, top_k, multi-query flag hash to prevent stale results)
- Cache invalidated on document upload

### 12. Evaluation
```bash
cd backend
python -m app.evaluation.evaluate
```
Runs **baseline vs improved** comparison across 4 RAGAS metrics:
- Faithfulness
- Answer relevancy
- Context precision *(new)*
- Context recall *(new)*

Reports metric deltas with visual indicators (✅/❌).

### 13. Observability
Every query logs to `backend/logs/rag_queries.jsonl` with:
- Retrieved/reranked chunks with all scores and `match_type`
- Confidence score, label, signal breakdown, and calibration stats
- Failure class, detailed reason, and supporting metrics
- Multi-query: variations, count, `selected_query_variant`
- `cache_hit` flag
- Per-stage latency breakdown
- Final prompt and generated answer

## Frontend Features
- **Query Input** with drag-and-drop file upload
- **Answer Display** with:
  - Citation highlighting
  - Confidence badge (color-coded: green/yellow/red)
  - Failure class indicator
  - Latency badge
- **Sources Panel** with:
  - Match-type badges (🧠 Semantic / 🔤 Keyword / 🔗 Hybrid)
  - Score badges (rerank, similarity, RRF)
  - **"💡 Why selected?"** toggle — human-readable selection explanation
  - Expandable chunk text
- **Debug Panel** with tabbed views:
  - ⏱ Latency breakdown (visual bars)
  - 🔄 **Before vs After** reranking comparison (rank change indicators ↑↓)
  - 📥 Retrieved chunks table (with match_type column)
  - 🎯 Reranked results
  - 📄 Assembled context block
  - 💬 Full prompt sent to LLM
  - 🔀 Query expansion chips at top of debug panel

## Project Structure

```
├── backend/
│   ├── app/
│   │   ├── main.py                 # FastAPI entry point
│   │   ├── config.py               # All configuration constants
│   │   ├── models.py               # Pydantic schemas
│   │   ├── core/
│   │   │   ├── parser.py           # PDF + Markdown parsing
│   │   │   ├── chunker.py          # Structure-aware + fixed chunking
│   │   │   ├── embedder.py         # sentence-transformers + cache
│   │   │   ├── vectorstore.py      # ChromaDB operations
│   │   │   ├── retriever.py        # Hybrid BM25 + vector + RRF/weighted
│   │   │   ├── reranker.py         # Cross-encoder reranking
│   │   │   ├── generator.py        # Multi-provider LLM generation
│   │   │   ├── pipeline.py         # Full V2 orchestration
│   │   │   ├── multi_query.py      # Query expansion + parallel retrieval
│   │   │   ├── confidence.py       # Multi-signal confidence scoring
│   │   │   ├── failure_classifier.py  # Structured failure classification
│   │   │   └── cache.py            # Embedding + query cache (LRU + TTL)
│   │   ├── evaluation/
│   │   │   └── evaluate.py         # RAGAS metrics (baseline vs improved)
│   │   ├── observability/
│   │   │   └── logger.py           # Structured JSONL logging
│   │   └── routes/
│   │       ├── query.py            # Query endpoint
│   │       ├── upload.py           # Upload endpoint
│   │       └── health.py           # Health check
│   ├── .env                        # API keys (not committed)
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.jsx + App.css
│   │   └── components/
│   │       ├── QueryInput.jsx
│   │       ├── AnswerDisplay.jsx   # Confidence badge + failure label
│   │       ├── SourcesPanel.jsx    # Match-type + "Why selected?"
│   │       └── DebugPanel.jsx      # Before/After + query variations
│   └── package.json
└── README.md
```

## Configuration

All parameters in `backend/app/config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CHUNK_SIZE` | 512 | Characters per chunk |
| `STRUCTURE_AWARE_CHUNKING` | True | Enable heading-based splitting |
| `FUSION_STRATEGY` | "rrf" | "rrf" or "weighted_sum" |
| `MULTI_QUERY_ENABLED` | True | Enable LLM query expansion |
| `MULTI_QUERY_COUNT` | 3 | Number of query variations |
| `RERANK_TOP_N` | 5 | Final chunks after reranking |
| `CACHE_TTL_SECONDS` | 300 | Query cache expiry (5 min) |
| `CONFIDENCE_THRESHOLDS` | {high: 0.7, medium: 0.4} | Label boundaries |

## Design Decisions

1. **Multi-query with filtering**: LLM-generated expansions are filtered by semantic similarity (>0.3) and deduplicated (>0.85 similarity removed) to ensure each query adds unique retrieval signal.

2. **Parallel retrieval**: ThreadPoolExecutor runs all query variants concurrently. On a 4-query setup, this adds ~0ms over single-query retrieval latency.

3. **Config-aware cache keys**: Query cache includes a hash of retrieval configuration (fusion strategy, top-k values, multi-query flag) so changing config parameters automatically invalidates stale entries.

4. **Confidence calibration**: Rolling window of last 100 confidence scores tracks distribution (mean, p50) for tuning thresholds based on actual workload.

5. **Structured failure reasons**: Every failure class includes specific measured values (rerank scores, overlap ratios, match type distributions) rather than generic messages — enabling systematic debugging.

6. **Selection explanations**: Each source includes a human-readable `selection_reason` explaining *why* it was chosen (semantic match, keyword overlap, reranker confirmation) — critical for debugging and trust.

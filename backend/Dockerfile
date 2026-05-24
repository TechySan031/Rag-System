# ===========================================================
#  Backend Dockerfile — Multi-stage build
#  Stage 1: Install Python deps (cached layer)
#  Stage 2: Copy app code (fast rebuild on code changes)
# ===========================================================
FROM python:3.13-slim AS base

# System deps for PyMuPDF, torch, chromadb
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- Stage: Dependencies (cached unless requirements.txt changes) ---
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# --- Stage: App code ---
COPY . .

# Create persistent dirs
RUN mkdir -p /app/logs /app/uploads /app/chroma_data \
    /app/logs/regression_history /app/logs/metrics_history /app/logs/calibration_history

# Production defaults
ENV RAG_ENV=prod
ENV LOG_DEBUG_SAMPLE_RATE=0.1
ENV LLM_TIMEOUT_SECONDS=60
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# Gunicorn with uvicorn workers for production
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2", "--timeout-keep-alive", "120"]

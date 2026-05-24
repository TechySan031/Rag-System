# ===========================================================
#  Hugging Face Spaces — Combined Dockerfile
#  Stage 1: Build React frontend (Vite)
#  Stage 2: Python backend + serve built frontend
#
#  Deploys the ENTIRE app as a single container.
#  One URL — no CORS issues — free forever.
# ===========================================================

# --- Stage 1: Build React frontend ---
FROM node:20-alpine AS frontend-build

WORKDIR /frontend

COPY frontend/package*.json ./
RUN npm ci

COPY frontend/ .

# In single-container mode, frontend calls API on same origin
ENV VITE_API_URL=""
RUN npm run build

# --- Stage 2: Python backend + frontend assets ---
FROM python:3.13-slim

# System deps for PyMuPDF, torch
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps (cached layer)
COPY backend/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY backend/ .

# Copy built frontend → /app/static (FastAPI serves this)
COPY --from=frontend-build /frontend/dist /app/static

# Create persistent dirs
RUN mkdir -p /app/logs /app/uploads /app/chroma_data \
    /app/logs/regression_history /app/logs/metrics_history /app/logs/calibration_history

# Production defaults
ENV RAG_ENV=prod
ENV LOG_DEBUG_SAMPLE_RATE=0.1
ENV LLM_TIMEOUT_SECONDS=120
ENV PYTHONUNBUFFERED=1

# HF Spaces requires port 7860
EXPOSE 7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]

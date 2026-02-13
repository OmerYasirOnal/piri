# =============================================================================
# Piri v3 — AKIS Platform RAG Engine
# Multi-stage Docker build optimized for OCI Free Tier (ARM64, low RAM)
# =============================================================================

# Stage 1: Builder — install dependencies
FROM python:3.11-slim AS builder

WORKDIR /build

# System deps for building native extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install CPU-only torch first (saves ~1.5GB vs full torch)
RUN pip install --no-cache-dir --target=/build/deps \
    torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
RUN pip install --no-cache-dir --target=/build/deps -r requirements.txt

# Stage 2: Runtime — minimal image
FROM python:3.11-slim AS runtime

WORKDIR /app

# Runtime system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -r -s /bin/false piri

# Copy installed packages from builder
COPY --from=builder /build/deps /usr/local/lib/python3.11/site-packages

# Copy application source
COPY main.py .
COPY piri/ ./piri/
COPY static/ ./static/
COPY knowledge_base/ ./knowledge_base/

# Create directories for runtime data
RUN mkdir -p /app/vector_store /app/data \
    && chown -R piri:piri /app

USER piri

# Piri defaults
ENV PYTHONUNBUFFERED=1 \
    PIRI_HOST=0.0.0.0 \
    PIRI_PORT=8000

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -sf http://localhost:8000/health || exit 1

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

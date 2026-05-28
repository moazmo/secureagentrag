# ==============================================================================
# Stage 1: Builder — install dependencies with uv
# ==============================================================================
FROM python:3.11-slim AS builder

WORKDIR /app

# Install uv package manager
RUN pip install --no-cache-dir uv

# Copy dependency manifest
COPY pyproject.toml ./

# Optional extras appended to the editable install, e.g. "[api,metrics]".
# Defaults to empty so the base image stays lean; the observability overlay
# passes INSTALL_EXTRAS=[api,metrics] to get the uvicorn API + /metrics.
ARG INSTALL_EXTRAS=""

# Create virtual environment and install dependencies
# Using uv pip install instead of uv sync to avoid uv.lock requirement
RUN uv venv /app/.venv && \
    uv pip install --python /app/.venv/bin/python -e ".${INSTALL_EXTRAS}"

# ==============================================================================
# Stage 2: Runtime — lean production image
# ==============================================================================
FROM python:3.11-slim AS runtime

WORKDIR /app

# Install runtime system dependencies (for PaddleOCR, PDF processing)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgl1-mesa-glx \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /app/.venv /app/.venv

# Ensure virtualenv is on PATH
ENV PATH="/app/.venv/bin:$PATH"

# Copy application source code
COPY . .

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]

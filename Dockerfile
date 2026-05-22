# ---------------------------------------------------------------------------
# Stage 1: Builder – install dependencies into a virtual environment
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS builder

WORKDIR /app

# Install uv (fast Python package manager)
RUN pip install --no-cache-dir uv

# Copy only the files needed to resolve & install dependencies.
# This layer is cached as long as pyproject.toml and uv.lock don't change.
COPY pyproject.toml uv.lock ./

# Install production dependencies (no dev extras) into /app/.venv
RUN uv sync --frozen --no-dev

# ---------------------------------------------------------------------------
# Stage 2: Runtime – lean image with only what the app needs
# ---------------------------------------------------------------------------
FROM python:3.12-slim

WORKDIR /app

# Copy virtual environment from the builder stage
COPY --from=builder /app/.venv /app/.venv

# Copy application source code
COPY app/ ./app/
COPY main.py ./

# Copy trained model
COPY data/ ./data/

# Make the venv the default Python environment
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

# Expose the application port
EXPOSE 8000

# Health check – hits the /health endpoint every 30 seconds
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health').read()"

# Start the Litestar API server
CMD ["litestar", "--app", "main:app", "run", "--host", "0.0.0.0", "--port", "8000"]

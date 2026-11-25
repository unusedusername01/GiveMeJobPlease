FROM python:3.12-slim

# Copy uv binary from official Astral image for fast deterministic installs
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Minimal OS dependencies (CA certs for HTTPS + libgomp1 for ONNX Runtime)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root runtime user
RUN useradd --create-home --shell /bin/bash --uid 10001 appuser

# Install dependencies first for Docker layer caching
COPY pyproject.toml README.md ./
COPY src ./src
COPY alembic.ini ./
COPY migrations ./migrations
COPY entrypoint.sh ./

RUN uv sync --no-dev && \
    chmod +x /app/entrypoint.sh && \
    mkdir -p /app/data/logs /app/data/backups /app/data/model_cache && \
    chown -R appuser:appuser /app

USER appuser

VOLUME ["/app/data"]

ENTRYPOINT ["/app/entrypoint.sh"]

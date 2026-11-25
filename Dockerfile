FROM python:3.12-slim@sha256:191083cdf6c7bda8c76158983163ead5d5cf7b77cf3033e805bb0f5296f51b4b

# Copy uv binary from official Astral image for fast deterministic installs.
# Pinned by version AND digest: a moved tag or registry compromise cannot swap
# the toolchain that installs every dependency.
COPY --from=ghcr.io/astral-sh/uv:0.13.10@sha256:a3ea1357342c1a4bf9a0c53a3deb1b55379aefa037338d8113d5771efa01774f /uv /uvx /bin/

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

# Install dependencies first for Docker layer caching.
# --locked verifies that uv.lock actually matches pyproject.toml and refuses to
# re-resolve; the build fails loudly instead of silently installing a different
# (possibly compromised) artifact set than the one reviewed in the lockfile.
COPY pyproject.toml README.md uv.lock ./
RUN uv sync --locked --no-dev --no-install-project

COPY src ./src
COPY alembic.ini ./
COPY migrations ./migrations
COPY entrypoint.sh ./

RUN uv sync --locked --no-dev && \
    chmod +x /app/entrypoint.sh && \
    mkdir -p /app/data/logs /app/data/backups /app/data/model_cache && \
    chown -R appuser:appuser /app

USER appuser

VOLUME ["/app/data"]

ENTRYPOINT ["/app/entrypoint.sh"]

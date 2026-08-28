# Multi-stage build for narrative-engine
FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.12.7 /uv /uvx /bin/

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
# Install the locked server environment. Linux resolves torch from the
# CPU-only index, avoiding the otherwise-unused CUDA runtime.
RUN uv sync --frozen --no-install-project --extra llm --extra web
COPY src/ ./src/
RUN uv sync --frozen --no-editable --extra llm --extra web

# Tests need a larger environment, but those tools do not belong in the
# long-running server image.
FROM builder AS test-builder
RUN uv sync --frozen --no-editable --extra dev --extra llm --extra web

FROM python:3.12-slim AS runtime-base

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 appuser
WORKDIR /app
ENV PYTHONPATH=/app/src
ENV PATH=/app/.venv/bin:$PATH

COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser pyproject.toml ./
COPY --chown=appuser:appuser alembic.ini ./
COPY --chown=appuser:appuser alembic/ ./alembic/

FROM runtime-base AS runtime
COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv
USER appuser
CMD ["uvicorn", "narrative_engine.api.app:app", "--host", "0.0.0.0", "--port", "8000"]

FROM runtime-base AS test
COPY --from=test-builder --chown=appuser:appuser /app/.venv /app/.venv
COPY --chown=appuser:appuser tests/ ./tests/
RUN chown appuser:appuser /app
USER appuser
CMD ["python", "-m", "pytest", "-v", "--tb=short"]

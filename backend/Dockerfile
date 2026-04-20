FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app/backend

COPY backend/ ./
RUN uv sync --frozen --no-dev

EXPOSE 8000

CMD ["/app/backend/.venv/bin/python", "-m", "backend.app"]

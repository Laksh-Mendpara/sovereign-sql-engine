"""Production-grade FastAPI backend for the Text2SQL pipeline.

Package layout
──────────────
backend/
├── __init__.py
├── app.py               — FastAPI factory + lifespan + uvicorn entry
├── config.py            — Pydantic Settings (reads pipeline_test/.env)
├── dependencies.py      — FastAPI Depends() helpers for service injection
├── logging_config.py    — Structured JSON logging + contextvars
├── metrics.py           — In-process latency histograms + counters
├── middleware.py        — RequestContext + Timing ASGI middleware
├── models.py            — Pydantic request / response schemas
├── pipeline_executor.py — Async parallel orchestrator
└── routes/
    ├── __init__.py
    ├── health.py        — GET /health, /ready, /metrics
    └── pipeline.py      — POST /v1/pipeline/query
"""

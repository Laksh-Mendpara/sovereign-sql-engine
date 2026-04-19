"""FastAPI dependency injection for pipeline services.

All heavy service objects are created once at startup inside ``lifespan``
and stored on ``app.state``.  These ``Depends()`` functions pull them out
in route handlers without any global state.
"""

from __future__ import annotations

from fastapi import Request

from .services.classifier_service import ClassifierService
from .services.guard_service import GuardService
from .services.metadata_service import MetadataService
from .services.neo4j_service import Neo4jService
from .services.pinecone_service import PineconeService
from .services.runpod_service import RunpodService

from .config import BackendSettings


def get_settings(request: Request) -> BackendSettings:
    return request.app.state.settings  # type: ignore[return-value]


def get_guard_service(request: Request) -> GuardService:
    return request.app.state.guard_service  # type: ignore[return-value]


def get_classifier_service(request: Request) -> ClassifierService:
    return request.app.state.classifier_service  # type: ignore[return-value]


def get_pinecone_service(request: Request) -> PineconeService:
    return request.app.state.pinecone_service  # type: ignore[return-value]


def get_neo4j_service(request: Request) -> Neo4jService:
    return request.app.state.neo4j_service  # type: ignore[return-value]


def get_metadata_service(request: Request) -> MetadataService:
    return request.app.state.metadata_service  # type: ignore[return-value]


def get_runpod_service(request: Request) -> RunpodService:
    return request.app.state.runpod_service  # type: ignore[return-value]

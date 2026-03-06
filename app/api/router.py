"""API router configuration."""

from fastapi import APIRouter

from app.api.endpoints import analysis, health

api_router = APIRouter()

api_router.include_router(health.router, tags=["health"])
api_router.include_router(analysis.router, prefix="/api/v1", tags=["analysis"])

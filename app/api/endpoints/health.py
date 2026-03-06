"""Health check endpoint."""

from fastapi import APIRouter

from app.config import get_settings

router = APIRouter()


@router.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint for monitoring and load balancers."""
    settings = get_settings()
    return {
        "status": "healthy",
        "service": settings.app_name,
        "version": settings.app_version,
    }


@router.get("/ready")
async def readiness_check() -> dict[str, str | bool]:
    """Readiness check - verifies the service can process requests."""
    settings = get_settings()
    checks: dict[str, bool] = {
        "supabase_configured": bool(settings.supabase_url and settings.supabase_service_role_key),
    }
    all_ready = all(checks.values())
    return {
        "status": "ready" if all_ready else "not_ready",
        **checks,
    }

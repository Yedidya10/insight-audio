"""Supabase client for database operations."""

import logging

from supabase import Client, create_client

from app.config import get_settings

logger = logging.getLogger(__name__)

_client: Client | None = None


def get_supabase_client() -> Client:
    """Get or create a Supabase client instance."""
    global _client  # noqa: PLW0603

    if _client is None:
        settings = get_settings()
        if not settings.supabase_url or not settings.supabase_service_role_key:
            raise RuntimeError("Supabase URL and service role key must be configured")

        _client = create_client(
            settings.supabase_url,
            settings.supabase_service_role_key,
        )
        logger.info("Supabase client initialized")

    return _client

"""Tests for configuration."""

from app.config import Settings, get_settings


def test_default_settings():
    settings = Settings()
    assert settings.app_name == "insight-audio"
    assert settings.port == 8000
    assert settings.worker_concurrency == 2
    assert settings.max_retry_attempts == 3


def test_get_settings_cached():
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2

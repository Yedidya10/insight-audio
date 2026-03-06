"""Application configuration loaded from environment variables."""

from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings from environment variables."""

    # App
    app_name: str = "insight-audio"
    app_version: str = "0.1.0"
    debug: bool = False
    log_level: str = "INFO"

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1

    # Supabase
    supabase_url: str = ""
    supabase_service_role_key: str = ""

    # CORS
    cors_origins: list[str] = ["http://localhost:3000"]

    # Audio processing
    temp_audio_dir: str = "/tmp/insight-audio"
    max_file_size_mb: int = 100
    audio_sample_rate: int = 22050
    processing_timeout_seconds: int = 300

    # Queue worker
    worker_concurrency: int = 2
    max_retry_attempts: int = 3
    poll_interval_seconds: int = 10

    # Webhook
    webhook_url: str = ""
    webhook_secret: str = ""

    # Models
    models_dir: str = "./models"
    panns_model_name: str = "Cnn14_mAP=0.431.pth"
    clap_model_name: str = "laion/larger_clap_music_and_speech"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "case_sensitive": False}


@lru_cache
def get_settings() -> Settings:
    """Get cached application settings."""
    return Settings()

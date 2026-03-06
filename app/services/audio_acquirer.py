"""Audio acquisition service - downloads audio from YouTube via yt-dlp."""

import logging
import subprocess
import tempfile
from pathlib import Path

from app.config import get_settings

logger = logging.getLogger(__name__)


class AudioAcquirer:
    """Downloads audio from YouTube for analysis."""

    def __init__(self) -> None:
        self.settings = get_settings()

    async def download(self, artist: str, track: str, track_id: str) -> Path | None:
        """Download audio for a track from YouTube.

        Returns the path to the downloaded WAV file, or None if download failed.
        Audio is downloaded to a temp directory and must be cleaned up after analysis.
        """
        search_query = f"{artist} - {track} official audio"
        output_dir = Path(self.settings.temp_audio_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / f"{track_id}.wav"

        try:
            result = subprocess.run(  # noqa: S603
                [
                    "yt-dlp",
                    "--extract-audio",
                    "--audio-format", "wav",
                    "--audio-quality", "0",
                    "--no-playlist",
                    "--default-search", "ytsearch1",
                    "--output", str(output_dir / f"{track_id}.%(ext)s"),
                    "--quiet",
                    "--no-warnings",
                    "--max-filesize", f"{self.settings.max_file_size_mb}M",
                    search_query,
                ],
                capture_output=True,
                text=True,
                timeout=self.settings.processing_timeout_seconds,
                check=False,
            )

            if result.returncode != 0:
                logger.error("yt-dlp failed for '%s': %s", search_query, result.stderr)
                return None

            if output_path.exists():
                logger.info("Downloaded audio for '%s' (%d bytes)", search_query, output_path.stat().st_size)
                return output_path

            logger.error("Output file not found after download: %s", output_path)
            return None

        except subprocess.TimeoutExpired:
            logger.error("Download timed out for '%s'", search_query)
            return None
        except FileNotFoundError:
            logger.error("yt-dlp not found. Install it with: pip install yt-dlp")
            return None

    @staticmethod
    def cleanup(file_path: Path) -> None:
        """Delete a downloaded audio file."""
        try:
            if file_path.exists():
                file_path.unlink()
                logger.debug("Cleaned up audio file: %s", file_path)
        except OSError:
            logger.warning("Failed to clean up: %s", file_path)

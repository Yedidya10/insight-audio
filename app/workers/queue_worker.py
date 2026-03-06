"""Audio analysis queue worker.

Polls the audio_analysis_queue table for pending jobs and processes them
through the analysis pipeline (download → librosa → PANNs → CLAP → store results).
"""

import asyncio
import logging
import time
from pathlib import Path

from app.config import get_settings
from app.services.audio_acquirer import AudioAcquirer
from app.services.librosa_extractor import LibrosaExtractor
from app.services.supabase_client import get_supabase_client

logger = logging.getLogger(__name__)


class QueueWorker:
    """Processes audio analysis jobs from the queue."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.acquirer = AudioAcquirer()
        self.librosa_extractor = LibrosaExtractor(sample_rate=self.settings.audio_sample_rate)
        self._running = False

    async def start(self) -> None:
        """Start the queue worker polling loop."""
        self._running = True
        logger.info(
            "Queue worker started (concurrency=%d, poll_interval=%ds)",
            self.settings.worker_concurrency,
            self.settings.poll_interval_seconds,
        )

        while self._running:
            try:
                await self._poll_and_process()
            except Exception:
                logger.exception("Error in queue worker poll cycle")

            await asyncio.sleep(self.settings.poll_interval_seconds)

    async def stop(self) -> None:
        """Stop the queue worker."""
        self._running = False
        logger.info("Queue worker stopped")

    async def _poll_and_process(self) -> None:
        """Poll for pending jobs and process them."""
        client = get_supabase_client()

        # Fetch pending jobs ordered by priority
        result = (
            client.table("audio_analysis_queue")
            .select("*")
            .eq("status", "pending")
            .lt("attempts", self.settings.max_retry_attempts)
            .order("priority", desc=True)
            .order("created_at")
            .limit(self.settings.worker_concurrency)
            .execute()
        )

        if not result.data:
            return

        logger.info("Found %d pending jobs", len(result.data))

        for job in result.data:
            await self._process_job(job)

    async def _process_job(self, job: dict) -> None:
        """Process a single analysis job."""
        job_id = job["id"]
        track_id = job["track_id"]
        start_time = time.monotonic()

        logger.info("Processing job %s (track: %s)", job_id, track_id)

        client = get_supabase_client()

        # Mark as processing
        client.table("audio_analysis_queue").update(
            {"status": "processing", "attempts": job["attempts"] + 1}
        ).eq("id", job_id).execute()

        audio_path: Path | None = None
        try:
            # Get track info
            track_result = (
                client.table("tracks")
                .select("name, artist_name")
                .eq("spotify_track_id", track_id)
                .limit(1)
                .execute()
            )

            if not track_result.data:
                raise ValueError(f"Track {track_id} not found")

            track = track_result.data[0]

            # Download audio
            audio_path = await self.acquirer.download(
                artist=track["artist_name"],
                track=track["name"],
                track_id=track_id,
            )

            if audio_path is None:
                raise RuntimeError(f"Failed to download audio for {track_id}")

            # Extract librosa features
            librosa_features = await self.librosa_extractor.extract(audio_path)

            processing_time = time.monotonic() - start_time

            # Store results
            client.table("track_audio_analysis").upsert(
                {
                    "track_id": track_id,
                    "librosa_features": librosa_features.model_dump(),
                    "status": "completed",
                    "analysis_version": "1.0.0",
                    "processing_time_seconds": processing_time,
                }
            ).execute()

            # Mark job as completed
            client.table("audio_analysis_queue").update(
                {"status": "completed"}
            ).eq("id", job_id).execute()

            logger.info(
                "Job %s completed in %.1fs (track: %s)",
                job_id, processing_time, track_id,
            )

        except Exception as exc:
            processing_time = time.monotonic() - start_time
            logger.exception("Job %s failed: %s", job_id, exc)

            # Mark as failed
            client.table("audio_analysis_queue").update(
                {
                    "status": "failed" if job["attempts"] + 1 >= self.settings.max_retry_attempts else "pending",
                    "error_log": str(exc)[:1000],
                }
            ).eq("id", job_id).execute()

        finally:
            # Always clean up audio file
            if audio_path is not None:
                self.acquirer.cleanup(audio_path)

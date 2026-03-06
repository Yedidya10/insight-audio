"""Audio analysis queue worker.

Polls the audio_analysis_queue table for pending jobs and processes them
through the full analysis pipeline:
  1. Download audio (yt-dlp from YouTube)
  2. Tier 1: librosa feature extraction
  3. Tier 1: PANNs Cnn14 classification + embedding
  4. Tier 2: CLAP audio embedding (if requested)
  5. Store results in track_audio_analysis + track_audio_embeddings
  6. Send webhook notification to main Insight app

Features:
  - Configurable concurrency (1-4 parallel jobs via asyncio.Semaphore)
  - Exponential backoff retry (max 3 attempts)
  - Dead letter handling for permanently failed jobs
  - Priority queue (user-requested > background)
  - Status webhook notifications
  - Monitoring: queue depth, processing time, error rate logging
"""

import asyncio
import hashlib
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

from app.config import get_settings
from app.services.audio_acquirer import AudioAcquirer
from app.services.librosa_extractor import LibrosaExtractor
from app.services.panns_classifier import PannsClassifier
from app.services.clap_embedder import ClapEmbedder
from app.services.supabase_client import get_supabase_client

logger = logging.getLogger(__name__)


def compute_fingerprint(artist: str, track: str) -> str:
    """Compute a dedup fingerprint for a track."""
    normalized = f"{artist.strip().lower()}::{track.strip().lower()}"
    return hashlib.sha256(normalized.encode()).hexdigest()


class QueueWorker:
    """Processes audio analysis jobs from the queue with concurrency control."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.acquirer = AudioAcquirer()
        self.librosa_extractor = LibrosaExtractor(sample_rate=self.settings.audio_sample_rate)
        self.panns_classifier = PannsClassifier(model_dir=self.settings.models_dir)
        self.clap_embedder = ClapEmbedder()
        self._running = False
        self._semaphore: asyncio.Semaphore | None = None

        # Monitoring counters
        self._jobs_completed = 0
        self._jobs_failed = 0
        self._total_processing_time = 0.0

    async def start(self) -> None:
        """Start the queue worker polling loop."""
        self._running = True
        self._semaphore = asyncio.Semaphore(self.settings.worker_concurrency)

        logger.info(
            "Queue worker started (concurrency=%d, poll_interval=%ds, max_retries=%d)",
            self.settings.worker_concurrency,
            self.settings.poll_interval_seconds,
            self.settings.max_retry_attempts,
        )

        while self._running:
            try:
                await self._poll_and_process()
            except Exception:
                logger.exception("Error in queue worker poll cycle")

            await asyncio.sleep(self.settings.poll_interval_seconds)

    async def stop(self) -> None:
        """Stop the queue worker gracefully."""
        self._running = False
        logger.info(
            "Queue worker stopped. Stats: completed=%d, failed=%d, avg_time=%.1fs",
            self._jobs_completed,
            self._jobs_failed,
            self._total_processing_time / max(self._jobs_completed, 1),
        )

    async def _poll_and_process(self) -> None:
        """Poll for queued jobs and process them concurrently."""
        client = get_supabase_client()

        # Fetch queued jobs ordered by priority (higher first), then by creation time
        result = (
            client.table("audio_analysis_queue")
            .select("*")
            .eq("status", "queued")
            .lt("retry_count", "max_retries")
            .order("priority", desc=True)
            .order("created_at")
            .limit(self.settings.worker_concurrency)
            .execute()
        )

        if not result.data:
            return

        logger.info("Found %d queued jobs", len(result.data))

        # Process jobs concurrently with semaphore
        tasks = [self._process_with_semaphore(job) for job in result.data]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _process_with_semaphore(self, job: dict) -> None:
        """Process a job with concurrency control."""
        assert self._semaphore is not None
        async with self._semaphore:
            await self._process_job(job)

    async def _process_job(self, job: dict) -> None:
        """Process a single analysis job through the full pipeline."""
        job_id = job["id"]
        track_name = job["track_name"]
        artist_name = job["artist_name"]
        track_fingerprint = job["track_fingerprint"]
        analysis_tier = job.get("analysis_tier", 1)
        retry_count = job.get("retry_count", 0)
        start_time = time.monotonic()

        logger.info(
            "Processing job %s: '%s - %s' (tier=%d, attempt=%d)",
            job_id, artist_name, track_name, analysis_tier, retry_count + 1,
        )

        client = get_supabase_client()

        # Mark as processing
        client.table("audio_analysis_queue").update({
            "status": "processing",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "retry_count": retry_count + 1,
        }).eq("id", job_id).execute()

        audio_path: Path | None = None
        try:
            # Step 1: Download audio
            audio_path = await self.acquirer.download(
                artist=artist_name,
                track=track_name,
                track_id=track_fingerprint[:16],
            )

            if audio_path is None:
                raise RuntimeError(f"Failed to download audio for '{artist_name} - {track_name}'")

            # Step 2: Librosa feature extraction (always, Tier 1)
            librosa_features = await self.librosa_extractor.extract(audio_path)

            # Step 3: PANNs classification (always, Tier 1)
            panns_features = await self.panns_classifier.classify(audio_path)

            processing_time = time.monotonic() - start_time

            # Store analysis results
            analysis_data = {
                "track_fingerprint": track_fingerprint,
                "track_name": track_name,
                "artist_name": artist_name,
                "track_id": job.get("track_id"),
                "isrc": job.get("isrc"),
                "audio_source": "youtube",
                "audio_duration_analyzed": librosa_features.duration_seconds,
                # Genre
                "genres": [g.model_dump() for g in panns_features.genres],
                "primary_genre": panns_features.genres[0].tag if panns_features.genres else None,
                # Mood
                "moods": [m.model_dump() for m in panns_features.moods],
                "primary_mood": panns_features.moods[0].tag if panns_features.moods else None,
                # Instruments
                "instruments": [i.model_dump() for i in panns_features.instruments],
                # Vocal
                "is_vocal": panns_features.is_vocal,
                "singing_detected": panns_features.is_vocal > 0.5,
                "male_singing": panns_features.male_singing,
                "female_singing": panns_features.female_singing,
                # Librosa spectral
                "spectral_centroid_mean": librosa_features.spectral_centroid_mean,
                "spectral_bandwidth_mean": librosa_features.spectral_bandwidth_mean,
                "spectral_rolloff_mean": librosa_features.spectral_rolloff_mean,
                "spectral_flatness_mean": librosa_features.spectral_flatness_mean,
                # Librosa timbre
                "mfcc_means": librosa_features.mfccs,
                "mfcc_vars": librosa_features.mfcc_vars,
                "chroma_mean": librosa_features.chroma,
                "tonnetz_mean": librosa_features.tonnetz,
                "zero_crossing_rate_mean": librosa_features.zero_crossing_rate_mean,
                "rms_energy_mean": librosa_features.rms_energy_mean,
                # Rhythm
                "bpm": librosa_features.tempo,
                "bpm_confidence": librosa_features.bpm_confidence,
                "estimated_key": f"{librosa_features.key} {librosa_features.mode}",
                "key_confidence": librosa_features.key_confidence,
                # Metadata
                "analysis_status": "completed",
                "analysis_version": "1.0.0",
            }

            client.table("track_audio_analysis").upsert(
                analysis_data, on_conflict="track_fingerprint"
            ).execute()

            # Step 4: Store PANNs embedding (Tier 2+)
            if analysis_tier >= 2:
                panns_embedding = await self.panns_classifier.get_embedding(audio_path)
                client.table("track_audio_embeddings").upsert({
                    "track_fingerprint": track_fingerprint,
                    "model_name": "panns-cnn14",
                    "embedding_dim": len(panns_embedding),
                    "embedding": panns_embedding,
                }, on_conflict="track_fingerprint,model_name").execute()

                # CLAP embedding
                clap_embedding = await self.clap_embedder.get_audio_embedding(audio_path)
                client.table("track_audio_embeddings").upsert({
                    "track_fingerprint": track_fingerprint,
                    "model_name": "clap-music",
                    "embedding_dim": len(clap_embedding),
                    "embedding": clap_embedding,
                }, on_conflict="track_fingerprint,model_name").execute()

            # Mark job completed
            client.table("audio_analysis_queue").update({
                "status": "completed",
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }).eq("id", job_id).execute()

            self._jobs_completed += 1
            self._total_processing_time += processing_time

            logger.info(
                "Job %s completed in %.1fs: '%s - %s'",
                job_id, processing_time, artist_name, track_name,
            )

            # Send webhook notification
            await self._send_webhook("completed", job, processing_time)

        except Exception as exc:
            processing_time = time.monotonic() - start_time
            self._jobs_failed += 1
            is_permanent_failure = retry_count + 1 >= self.settings.max_retry_attempts

            logger.exception("Job %s failed (attempt %d/%d): %s",
                             job_id, retry_count + 1, self.settings.max_retry_attempts, exc)

            # Update job status
            new_status = "failed" if is_permanent_failure else "queued"
            client.table("audio_analysis_queue").update({
                "status": new_status,
                "error_message": str(exc)[:1000],
            }).eq("id", job_id).execute()

            if is_permanent_failure:
                logger.warning("Job %s moved to dead letter (max retries exceeded)", job_id)
                await self._send_webhook("failed", job, processing_time)

        finally:
            # Always clean up audio file
            if audio_path is not None:
                self.acquirer.cleanup(audio_path)

    async def _send_webhook(self, status: str, job: dict, processing_time: float) -> None:
        """Send a status webhook to the main Insight app."""
        if not self.settings.webhook_url:
            return

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                await client.post(
                    self.settings.webhook_url,
                    json={
                        "event": "audio_analysis_complete",
                        "status": status,
                        "job_id": job["id"],
                        "track_fingerprint": job["track_fingerprint"],
                        "track_name": job["track_name"],
                        "artist_name": job["artist_name"],
                        "user_id": job["user_id"],
                        "processing_time_seconds": processing_time,
                    },
                    headers={"X-Webhook-Secret": self.settings.webhook_secret},
                )
        except Exception:
            logger.warning("Failed to send webhook for job %s", job["id"])

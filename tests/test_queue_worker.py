"""Tests for the audio analysis queue worker."""

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.workers.queue_worker import QueueWorker, compute_fingerprint


# ---------------------------------------------------------------------------
# compute_fingerprint
# ---------------------------------------------------------------------------

def test_compute_fingerprint_deterministic():
    fp1 = compute_fingerprint("Artist", "Track")
    fp2 = compute_fingerprint("Artist", "Track")
    assert fp1 == fp2


def test_compute_fingerprint_case_insensitive():
    fp1 = compute_fingerprint("Artist", "Track")
    fp2 = compute_fingerprint("ARTIST", "TRACK")
    assert fp1 == fp2


def test_compute_fingerprint_strips_whitespace():
    fp1 = compute_fingerprint("  Artist ", " Track  ")
    fp2 = compute_fingerprint("Artist", "Track")
    assert fp1 == fp2


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

def _make_job(
    *,
    job_id: str = "job-1",
    status: str = "queued",
    retry_count: int = 0,
    priority: int = 0,
    analysis_tier: int = 1,
) -> dict:
    return {
        "id": job_id,
        "track_name": "Bohemian Rhapsody",
        "artist_name": "Queen",
        "track_fingerprint": "abc123" * 8,
        "track_id": "spotify:track:123",
        "isrc": "GBUM71029604",
        "user_id": "user-1",
        "status": status,
        "retry_count": retry_count,
        "max_retries": 3,
        "priority": priority,
        "analysis_tier": analysis_tier,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


class _FakeFeatures:
    """Mimics LibrosaFeatures model for testing."""

    def __init__(self):
        self.duration_seconds = 180.0
        self.spectral_centroid_mean = 1500.0
        self.spectral_bandwidth_mean = 2000.0
        self.spectral_rolloff_mean = 4000.0
        self.spectral_flatness_mean = 0.01
        self.mfccs = [0.0] * 13
        self.mfcc_vars = [0.0] * 13
        self.chroma = [0.0] * 12
        self.tonnetz = [0.0] * 6
        self.zero_crossing_rate_mean = 0.05
        self.rms_energy_mean = 0.1
        self.tempo = 120.0
        self.bpm_confidence = 0.9
        self.key = "C"
        self.mode = "major"
        self.key_confidence = 0.8
        self.onset_strength_mean = 3.5


class _FakeTag:
    def __init__(self, tag: str, confidence: float):
        self.tag = tag
        self.confidence = confidence

    def model_dump(self):
        return {"tag": self.tag, "confidence": self.confidence}


class _FakePanns:
    def __init__(self):
        self.genres = [_FakeTag("Rock", 0.9)]
        self.moods = [_FakeTag("Energetic", 0.8)]
        self.instruments = [_FakeTag("Piano", 0.7)]
        self.is_vocal = 0.9
        self.male_singing = 0.8
        self.female_singing = 0.1


# ---------------------------------------------------------------------------
# QueueWorker tests
# ---------------------------------------------------------------------------

@pytest.fixture
def worker():
    """Create a QueueWorker with mocked dependencies."""
    with (
        patch("app.workers.queue_worker.AudioAcquirer") as MockAcquirer,
        patch("app.workers.queue_worker.LibrosaExtractor") as MockLibrosa,
        patch("app.workers.queue_worker.PannsClassifier") as MockPanns,
        patch("app.workers.queue_worker.ClapEmbedder") as MockClap,
        patch("app.workers.queue_worker.get_settings") as mock_settings,
    ):
        settings = MagicMock()
        settings.audio_sample_rate = 22050
        settings.models_dir = "./models"
        settings.worker_concurrency = 2
        settings.max_retry_attempts = 3
        settings.poll_interval_seconds = 1
        settings.webhook_url = ""
        settings.webhook_secret = ""
        mock_settings.return_value = settings

        w = QueueWorker()
        w.acquirer = MockAcquirer.return_value
        w.librosa_extractor = MockLibrosa.return_value
        w.panns_classifier = MockPanns.return_value
        w.clap_embedder = MockClap.return_value

        yield w


@pytest.mark.asyncio
async def test_process_job_success(worker):
    """Test successful job processing with Tier 1 analysis."""
    job = _make_job()
    audio_path = Path("/tmp/test.mp3")

    worker.acquirer.download = AsyncMock(return_value=audio_path)
    worker.acquirer.cleanup = MagicMock()
    worker.librosa_extractor.extract = AsyncMock(return_value=_FakeFeatures())
    worker.panns_classifier.classify = AsyncMock(return_value=_FakePanns())

    mock_table = MagicMock()
    mock_table.update.return_value.eq.return_value.execute.return_value = None
    mock_table.upsert.return_value.execute.return_value = None

    with patch("app.workers.queue_worker.get_supabase_client") as mock_client:
        mock_client.return_value.table.return_value = mock_table
        await worker._process_job(job)

    assert worker._jobs_completed == 1
    assert worker._jobs_failed == 0
    worker.acquirer.cleanup.assert_called_once_with(audio_path)


@pytest.mark.asyncio
async def test_process_job_download_failure(worker):
    """Test job failure when audio download returns None."""
    job = _make_job(retry_count=2)  # last attempt

    worker.acquirer.download = AsyncMock(return_value=None)
    worker.acquirer.cleanup = MagicMock()

    mock_table = MagicMock()
    mock_table.update.return_value.eq.return_value.execute.return_value = None

    with patch("app.workers.queue_worker.get_supabase_client") as mock_client:
        mock_client.return_value.table.return_value = mock_table
        await worker._process_job(job)

    assert worker._jobs_failed == 1
    assert worker._jobs_completed == 0


@pytest.mark.asyncio
async def test_process_job_retryable_failure(worker):
    """Test that a job is re-queued when retries remain."""
    job = _make_job(retry_count=0)  # first attempt, can still retry

    worker.acquirer.download = AsyncMock(return_value=None)
    worker.acquirer.cleanup = MagicMock()

    update_calls = []
    mock_table = MagicMock()

    def capture_update(data):
        update_calls.append(data)
        mock_eq = MagicMock()
        mock_eq.execute.return_value = None
        return MagicMock(eq=MagicMock(return_value=mock_eq))

    mock_table.update = capture_update

    with patch("app.workers.queue_worker.get_supabase_client") as mock_client:
        mock_client.return_value.table.return_value = mock_table
        await worker._process_job(job)

    # Last update should set status back to queued (not failed)
    last_update = update_calls[-1]
    assert last_update["status"] == "queued"


@pytest.mark.asyncio
async def test_process_job_permanent_failure(worker):
    """Test that a job is marked failed when max retries exceeded."""
    job = _make_job(retry_count=2)  # attempt 3 of 3

    worker.acquirer.download = AsyncMock(side_effect=RuntimeError("network error"))
    worker.acquirer.cleanup = MagicMock()

    update_calls = []
    mock_table = MagicMock()

    def capture_update(data):
        update_calls.append(data)
        mock_eq = MagicMock()
        mock_eq.execute.return_value = None
        return MagicMock(eq=MagicMock(return_value=mock_eq))

    mock_table.update = capture_update

    with patch("app.workers.queue_worker.get_supabase_client") as mock_client:
        mock_client.return_value.table.return_value = mock_table
        await worker._process_job(job)

    last_update = update_calls[-1]
    assert last_update["status"] == "failed"
    assert "network error" in last_update["error_message"]


@pytest.mark.asyncio
async def test_process_job_tier2_stores_embeddings(worker):
    """Test that Tier 2 analysis stores PANNs and CLAP embeddings."""
    job = _make_job(analysis_tier=2)
    audio_path = Path("/tmp/test.mp3")

    worker.acquirer.download = AsyncMock(return_value=audio_path)
    worker.acquirer.cleanup = MagicMock()
    worker.librosa_extractor.extract = AsyncMock(return_value=_FakeFeatures())
    worker.panns_classifier.classify = AsyncMock(return_value=_FakePanns())
    worker.panns_classifier.get_embedding = AsyncMock(return_value=[0.1] * 2048)
    worker.clap_embedder.get_audio_embedding = AsyncMock(return_value=[0.2] * 512)

    upsert_calls = []
    mock_table = MagicMock()
    mock_table.update.return_value.eq.return_value.execute.return_value = None

    def capture_upsert(data, **kwargs):
        upsert_calls.append((data, kwargs))
        mock_exec = MagicMock()
        mock_exec.execute.return_value = None
        return mock_exec

    mock_table.upsert = capture_upsert

    with patch("app.workers.queue_worker.get_supabase_client") as mock_client:
        mock_client.return_value.table.return_value = mock_table
        await worker._process_job(job)

    # Should have 3 upserts: track_audio_analysis + 2 embeddings
    assert len(upsert_calls) == 3
    embedding_upserts = [c for c in upsert_calls if "model_name" in c[0]]
    model_names = {c[0]["model_name"] for c in embedding_upserts}
    assert model_names == {"panns-cnn14", "clap-music"}


@pytest.mark.asyncio
async def test_poll_and_process_no_jobs(worker):
    """Test that poll_and_process handles empty queue gracefully."""
    worker._semaphore = asyncio.Semaphore(2)

    mock_result = MagicMock()
    mock_result.data = []

    with patch("app.workers.queue_worker.get_supabase_client") as mock_client:
        (
            mock_client.return_value.table.return_value
            .select.return_value
            .eq.return_value
            .lt.return_value
            .order.return_value
            .order.return_value
            .limit.return_value
            .execute.return_value
        ) = mock_result
        await worker._poll_and_process()

    assert worker._jobs_completed == 0


@pytest.mark.asyncio
async def test_webhook_not_sent_when_url_empty(worker):
    """Test that no webhook is sent when webhook_url is empty."""
    with patch("app.workers.queue_worker.httpx.AsyncClient") as mock_httpx:
        await worker._send_webhook("completed", _make_job(), 5.0)
        mock_httpx.assert_not_called()


@pytest.mark.asyncio
async def test_webhook_sent_when_url_configured(worker):
    """Test that webhook is sent when webhook_url is configured."""
    worker.settings.webhook_url = "https://example.com/webhook"
    worker.settings.webhook_secret = "secret123"

    mock_response = MagicMock()
    mock_client_instance = AsyncMock()
    mock_client_instance.post.return_value = mock_response

    with patch("app.workers.queue_worker.httpx.AsyncClient") as MockClient:
        MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_client_instance)
        MockClient.return_value.__aexit__ = AsyncMock(return_value=False)
        await worker._send_webhook("completed", _make_job(), 5.0)

    mock_client_instance.post.assert_called_once()
    call_kwargs = mock_client_instance.post.call_args
    assert call_kwargs[0][0] == "https://example.com/webhook"
    assert call_kwargs[1]["headers"]["X-Webhook-Secret"] == "secret123"
    assert call_kwargs[1]["json"]["status"] == "completed"


def test_worker_initial_state(worker):
    """Test initial state of the worker."""
    assert worker._jobs_completed == 0
    assert worker._jobs_failed == 0
    assert worker._total_processing_time == 0.0
    assert worker._running is False


@pytest.mark.asyncio
async def test_cleanup_always_called_on_exception(worker):
    """Test that audio cleanup happens even on unexpected exceptions."""
    job = _make_job()
    audio_path = Path("/tmp/test.mp3")

    worker.acquirer.download = AsyncMock(return_value=audio_path)
    worker.acquirer.cleanup = MagicMock()
    worker.librosa_extractor.extract = AsyncMock(side_effect=RuntimeError("extraction error"))

    mock_table = MagicMock()
    mock_table.update.return_value.eq.return_value.execute.return_value = None

    with patch("app.workers.queue_worker.get_supabase_client") as mock_client:
        mock_client.return_value.table.return_value = mock_table
        await worker._process_job(job)

    worker.acquirer.cleanup.assert_called_once_with(audio_path)

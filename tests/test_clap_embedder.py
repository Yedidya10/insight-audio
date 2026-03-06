"""Tests for the CLAP semantic audio embeddings service."""

import numpy as np
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.clap_embedder import ClapEmbedder, EMBEDDING_DIM, MODEL_NAME


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def embedder():
    """Create a ClapEmbedder with mocked model and settings."""
    with patch("app.services.clap_embedder.get_settings") as mock_settings:
        settings = MagicMock()
        settings.clap_model_name = "laion/larger_clap_music_and_speech"
        mock_settings.return_value = settings

        emb = ClapEmbedder()
        yield emb


def _make_torch_tensor(vectors: list[list[float]]):
    """Create a mock torch tensor from a list of float vectors."""
    import torch
    return torch.tensor(vectors, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Core embedding tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_audio_embedding(embedder):
    """Test generating a single audio embedding."""
    fake_embedding = [[0.1] * EMBEDDING_DIM]
    mock_model = MagicMock()
    mock_model.get_audio_embeddings.return_value = _make_torch_tensor(fake_embedding)
    embedder._model = mock_model

    result = await embedder.get_audio_embedding(Path("/tmp/test.mp3"))

    assert len(result) == EMBEDDING_DIM
    mock_model.get_audio_embeddings.assert_called_once_with(["/tmp/test.mp3"])


@pytest.mark.asyncio
async def test_get_text_embedding(embedder):
    """Test generating a text embedding."""
    fake_embedding = [[0.2] * EMBEDDING_DIM]
    mock_model = MagicMock()
    mock_model.get_text_embeddings.return_value = _make_torch_tensor(fake_embedding)
    embedder._model = mock_model

    result = await embedder.get_text_embedding("a rainy day")

    assert len(result) == EMBEDDING_DIM
    mock_model.get_text_embeddings.assert_called_once_with(["a rainy day"])


@pytest.mark.asyncio
async def test_get_audio_embeddings_batch(embedder):
    """Test batch audio embedding generation."""
    batch_size = 3
    fake_embeddings = [[float(i)] * EMBEDDING_DIM for i in range(batch_size)]
    mock_model = MagicMock()
    mock_model.get_audio_embeddings.return_value = _make_torch_tensor(fake_embeddings)
    embedder._model = mock_model

    paths = [Path(f"/tmp/track_{i}.mp3") for i in range(batch_size)]
    results = await embedder.get_audio_embeddings_batch(paths)

    assert len(results) == batch_size
    assert all(len(emb) == EMBEDDING_DIM for emb in results)


@pytest.mark.asyncio
async def test_get_text_embeddings_batch(embedder):
    """Test batch text embedding generation."""
    texts = ["rainy day", "sunny morning", "rock concert"]
    fake_embeddings = [[float(i)] * EMBEDDING_DIM for i in range(len(texts))]
    mock_model = MagicMock()
    mock_model.get_text_embeddings.return_value = _make_torch_tensor(fake_embeddings)
    embedder._model = mock_model

    results = await embedder.get_text_embeddings_batch(texts)

    assert len(results) == 3
    assert all(len(emb) == EMBEDDING_DIM for emb in results)


# ---------------------------------------------------------------------------
# classify_audio
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_classify_audio(embedder):
    """Test zero-shot audio classification."""
    # Audio embedding: [1, 0, 0...]
    audio_emb = [[1.0] + [0.0] * (EMBEDDING_DIM - 1)]
    # Label embeddings: first label closest to audio
    label_embs = [
        [1.0] + [0.0] * (EMBEDDING_DIM - 1),   # "rock" - should be highest
        [0.0, 1.0] + [0.0] * (EMBEDDING_DIM - 2),  # "jazz" - orthogonal
    ]
    mock_model = MagicMock()
    mock_model.get_audio_embeddings.return_value = _make_torch_tensor(audio_emb)
    mock_model.get_text_embeddings.return_value = _make_torch_tensor(label_embs)
    embedder._model = mock_model

    results = await embedder.classify_audio(Path("/tmp/test.mp3"), ["rock", "jazz"])

    assert results[0][0] == "rock"
    assert results[0][1] > results[1][1]


# ---------------------------------------------------------------------------
# cosine_similarity
# ---------------------------------------------------------------------------

def test_cosine_similarity_identical():
    """Identical vectors should have similarity 1.0."""
    vec = [1.0, 2.0, 3.0]
    sim = ClapEmbedder.cosine_similarity(vec, vec)
    assert abs(sim - 1.0) < 1e-5


def test_cosine_similarity_orthogonal():
    """Orthogonal vectors should have similarity 0.0."""
    a = [1.0, 0.0, 0.0]
    b = [0.0, 1.0, 0.0]
    sim = ClapEmbedder.cosine_similarity(a, b)
    assert abs(sim) < 1e-5


def test_cosine_similarity_opposite():
    """Opposite vectors should have similarity -1.0."""
    a = [1.0, 0.0]
    b = [-1.0, 0.0]
    sim = ClapEmbedder.cosine_similarity(a, b)
    assert abs(sim - (-1.0)) < 1e-5


def test_cosine_similarity_zero_vector():
    """Zero vector should return 0.0."""
    a = [0.0, 0.0, 0.0]
    b = [1.0, 2.0, 3.0]
    sim = ClapEmbedder.cosine_similarity(a, b)
    assert sim == 0.0


# ---------------------------------------------------------------------------
# Supabase integration
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_store_embedding(embedder):
    """Test storing an embedding to Supabase."""
    embedding = [0.1] * EMBEDDING_DIM

    with patch("app.services.clap_embedder.get_supabase_client") as mock_client:
        mock_table = MagicMock()
        mock_table.upsert.return_value.execute.return_value = None
        mock_client.return_value.table.return_value = mock_table

        await embedder.store_embedding("fp123", embedding)

    mock_table.upsert.assert_called_once()
    call_args = mock_table.upsert.call_args
    assert call_args[0][0]["track_fingerprint"] == "fp123"
    assert call_args[0][0]["model_name"] == MODEL_NAME
    assert call_args[0][0]["embedding_dim"] == EMBEDDING_DIM


@pytest.mark.asyncio
async def test_find_similar_tracks(embedder):
    """Test finding similar tracks via pgvector."""
    embedding = [0.1] * EMBEDDING_DIM

    with patch("app.services.clap_embedder.get_supabase_client") as mock_client:
        mock_rpc_result = MagicMock()
        mock_rpc_result.data = [
            {"track_fingerprint": "fp1", "similarity": 0.95},
            {"track_fingerprint": "fp2", "similarity": 0.80},
        ]
        mock_client.return_value.rpc.return_value.execute.return_value = mock_rpc_result

        results = await embedder.find_similar_tracks(embedding, limit=5)

    assert len(results) == 2
    assert results[0]["track_fingerprint"] == "fp1"


@pytest.mark.asyncio
async def test_find_similar_tracks_with_threshold(embedder):
    """Test threshold filtering."""
    embedding = [0.1] * EMBEDDING_DIM

    with patch("app.services.clap_embedder.get_supabase_client") as mock_client:
        mock_rpc_result = MagicMock()
        mock_rpc_result.data = [
            {"track_fingerprint": "fp1", "similarity": 0.95},
            {"track_fingerprint": "fp2", "similarity": 0.50},
        ]
        mock_client.return_value.rpc.return_value.execute.return_value = mock_rpc_result

        results = await embedder.find_similar_tracks(
            embedding, limit=5, threshold=0.8
        )

    assert len(results) == 1
    assert results[0]["track_fingerprint"] == "fp1"


@pytest.mark.asyncio
async def test_search_by_text(embedder):
    """Test text-to-audio search end-to-end."""
    fake_text_emb = [[0.3] * EMBEDDING_DIM]
    mock_model = MagicMock()
    mock_model.get_text_embeddings.return_value = _make_torch_tensor(fake_text_emb)
    embedder._model = mock_model

    with patch("app.services.clap_embedder.get_supabase_client") as mock_client:
        mock_rpc_result = MagicMock()
        mock_rpc_result.data = [
            {"track_fingerprint": "fp1", "similarity": 0.90},
        ]
        mock_client.return_value.rpc.return_value.execute.return_value = mock_rpc_result

        results = await embedder.search_by_text("rainy day jazz")

    assert len(results) == 1


@pytest.mark.asyncio
async def test_search_by_audio(embedder):
    """Test audio-to-audio similarity search."""
    fake_audio_emb = [[0.5] * EMBEDDING_DIM]
    mock_model = MagicMock()
    mock_model.get_audio_embeddings.return_value = _make_torch_tensor(fake_audio_emb)
    embedder._model = mock_model

    with patch("app.services.clap_embedder.get_supabase_client") as mock_client:
        mock_rpc_result = MagicMock()
        mock_rpc_result.data = [
            {"track_fingerprint": "fp2", "similarity": 0.85},
        ]
        mock_client.return_value.rpc.return_value.execute.return_value = mock_rpc_result

        results = await embedder.search_by_audio(Path("/tmp/test.mp3"))

    assert len(results) == 1


# ---------------------------------------------------------------------------
# Backfill
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_backfill_embeddings(embedder):
    """Test batch backfill of CLAP embeddings."""
    fake_embeddings = [[float(i)] * EMBEDDING_DIM for i in range(2)]
    mock_model = MagicMock()
    mock_model.get_audio_embeddings.return_value = _make_torch_tensor(fake_embeddings)
    embedder._model = mock_model

    with patch("app.services.clap_embedder.get_supabase_client") as mock_client:
        mock_table = MagicMock()
        mock_table.upsert.return_value.execute.return_value = None
        mock_client.return_value.table.return_value = mock_table

        result = await embedder.backfill_embeddings({
            "fp1": Path("/tmp/t1.mp3"),
            "fp2": Path("/tmp/t2.mp3"),
        })

    assert result["processed"] == 2
    assert result["succeeded"] == 2
    assert result["failed"] == 0


def test_constants():
    """Verify module-level constants."""
    assert EMBEDDING_DIM == 512
    assert MODEL_NAME == "clap-music"

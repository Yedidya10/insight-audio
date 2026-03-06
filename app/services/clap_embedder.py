"""CLAP semantic audio embeddings service.

CLAP (Contrastive Language-Audio Pretraining) maps audio and text into a shared
512-dimensional embedding space. This enables:
  - Audio-to-audio similarity search (find tracks that sound alike)
  - Text-to-audio semantic search (e.g. "find tracks that sound like a rainy day")
  - Zero-shot audio classification against arbitrary text labels
  - Batch embedding generation for existing track libraries
"""

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from app.config import get_settings
from app.services.supabase_client import get_supabase_client

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 512
MODEL_NAME = "clap-music"
BATCH_SIZE = 16


class ClapEmbedder:
    """CLAP contrastive language-audio embedding generator.

    Lazy-loads the CLAP model on first use and provides methods for
    audio embedding, text embedding, similarity search, and batch processing.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self._model: Any = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load_model(self) -> None:
        """Load CLAP model (lazy loading on first call)."""
        if self._model is not None:
            return

        try:
            from msclap import CLAP

            start = time.monotonic()
            self._model = CLAP(
                version=self.settings.clap_model_name,
                use_cuda=torch.cuda.is_available(),
            )
            elapsed = time.monotonic() - start
            logger.info("CLAP model loaded on %s in %.1fs", self._device, elapsed)
        except Exception:
            logger.exception("Failed to load CLAP model")
            raise

    # ------------------------------------------------------------------
    # Core embedding methods
    # ------------------------------------------------------------------

    async def get_audio_embedding(self, audio_path: Path) -> list[float]:
        """Generate a 512-dim audio embedding for a single track."""
        self._load_model()
        start = time.monotonic()

        audio_embeddings = self._model.get_audio_embeddings([str(audio_path)])
        embedding = audio_embeddings[0].cpu().numpy().tolist()

        logger.info(
            "CLAP audio embedding generated in %.2fs (%d dims) for %s",
            time.monotonic() - start, len(embedding), audio_path.name,
        )
        return embedding

    async def get_audio_embeddings_batch(
        self, audio_paths: list[Path]
    ) -> list[list[float]]:
        """Generate 512-dim audio embeddings for a batch of tracks."""
        self._load_model()
        start = time.monotonic()

        all_embeddings: list[list[float]] = []
        for i in range(0, len(audio_paths), BATCH_SIZE):
            batch = audio_paths[i : i + BATCH_SIZE]
            batch_paths = [str(p) for p in batch]
            batch_embs = self._model.get_audio_embeddings(batch_paths)
            all_embeddings.extend(
                batch_embs[j].cpu().numpy().tolist() for j in range(len(batch))
            )

        logger.info(
            "CLAP batch: %d embeddings in %.2fs",
            len(all_embeddings), time.monotonic() - start,
        )
        return all_embeddings

    async def get_text_embedding(self, text: str) -> list[float]:
        """Generate a 512-dim text embedding for semantic search."""
        self._load_model()

        text_embeddings = self._model.get_text_embeddings([text])
        embedding = text_embeddings[0].cpu().numpy().tolist()
        return embedding

    async def get_text_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate 512-dim text embeddings for multiple queries."""
        self._load_model()

        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            batch_embs = self._model.get_text_embeddings(batch)
            all_embeddings.extend(
                batch_embs[j].cpu().numpy().tolist() for j in range(len(batch))
            )
        return all_embeddings

    # ------------------------------------------------------------------
    # Similarity and search
    # ------------------------------------------------------------------

    async def classify_audio(
        self, audio_path: Path, labels: list[str]
    ) -> list[tuple[str, float]]:
        """Zero-shot classify audio against a set of text labels.

        Returns (label, similarity_score) pairs sorted by descending similarity.
        """
        self._load_model()
        logger.info("CLAP zero-shot classification with %d labels", len(labels))

        audio_emb = self._model.get_audio_embeddings([str(audio_path)])
        text_embs = self._model.get_text_embeddings(labels)

        # Cosine similarity
        audio_norm = audio_emb / audio_emb.norm(dim=-1, keepdim=True)
        text_norm = text_embs / text_embs.norm(dim=-1, keepdim=True)
        similarities = (audio_norm @ text_norm.T).squeeze(0)

        scores = similarities.cpu().numpy()
        return sorted(zip(labels, scores.tolist()), key=lambda x: -x[1])

    @staticmethod
    def cosine_similarity(a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two embedding vectors."""
        va = np.array(a, dtype=np.float32)
        vb = np.array(b, dtype=np.float32)
        dot = np.dot(va, vb)
        norm = np.linalg.norm(va) * np.linalg.norm(vb)
        return float(dot / norm) if norm > 0 else 0.0

    # ------------------------------------------------------------------
    # Supabase integration: store & search embeddings
    # ------------------------------------------------------------------

    async def store_embedding(
        self, track_fingerprint: str, embedding: list[float]
    ) -> None:
        """Store a CLAP embedding in track_audio_embeddings (pgvector)."""
        client = get_supabase_client()
        client.table("track_audio_embeddings").upsert(
            {
                "track_fingerprint": track_fingerprint,
                "model_name": MODEL_NAME,
                "embedding_dim": EMBEDDING_DIM,
                "embedding": embedding,
            },
            on_conflict="track_fingerprint,model_name",
        ).execute()

    async def find_similar_tracks(
        self,
        embedding: list[float],
        *,
        limit: int = 10,
        threshold: float = 0.0,
    ) -> list[dict]:
        """Find tracks with similar CLAP embeddings via pgvector cosine distance.

        Uses the Supabase RPC `find_similar_tracks` which performs:
          1 - (embedding <=> query_embedding) AS similarity
        """
        client = get_supabase_client()
        result = client.rpc(
            "find_similar_tracks",
            {
                "query_embedding": embedding,
                "match_count": limit,
                "model": MODEL_NAME,
            },
        ).execute()

        matches = result.data or []
        if threshold > 0:
            matches = [m for m in matches if m.get("similarity", 0) >= threshold]
        return matches

    async def search_by_text(
        self,
        query: str,
        *,
        limit: int = 10,
        threshold: float = 0.0,
    ) -> list[dict]:
        """Find tracks matching a natural language description.

        Converts the text query into a CLAP text embedding and searches
        the track_audio_embeddings table via pgvector cosine similarity.
        """
        text_embedding = await self.get_text_embedding(query)
        return await self.find_similar_tracks(
            text_embedding, limit=limit, threshold=threshold
        )

    async def search_by_audio(
        self,
        audio_path: Path,
        *,
        limit: int = 10,
        threshold: float = 0.0,
    ) -> list[dict]:
        """Find tracks similar to a given audio file."""
        audio_embedding = await self.get_audio_embedding(audio_path)
        return await self.find_similar_tracks(
            audio_embedding, limit=limit, threshold=threshold
        )

    # ------------------------------------------------------------------
    # Batch processing for existing tracks
    # ------------------------------------------------------------------

    async def backfill_embeddings(
        self,
        audio_paths_by_fingerprint: dict[str, Path],
    ) -> dict[str, int]:
        """Generate and store CLAP embeddings for a batch of existing tracks.

        Args:
            audio_paths_by_fingerprint: mapping of track_fingerprint → audio file path

        Returns:
            Summary dict with keys: processed, succeeded, failed
        """
        fingerprints = list(audio_paths_by_fingerprint.keys())
        paths = [audio_paths_by_fingerprint[fp] for fp in fingerprints]
        succeeded = 0
        failed = 0

        logger.info("Starting CLAP backfill for %d tracks", len(paths))

        for i in range(0, len(paths), BATCH_SIZE):
            batch_fps = fingerprints[i : i + BATCH_SIZE]
            batch_paths = paths[i : i + BATCH_SIZE]

            try:
                embeddings = await self.get_audio_embeddings_batch(batch_paths)
                for fp, emb in zip(batch_fps, embeddings):
                    await self.store_embedding(fp, emb)
                succeeded += len(batch_fps)
            except Exception:
                logger.exception(
                    "Batch %d-%d failed", i, i + len(batch_paths)
                )
                failed += len(batch_fps)

        logger.info(
            "CLAP backfill complete: %d/%d succeeded, %d failed",
            succeeded, len(paths), failed,
        )

        return {"processed": len(paths), "succeeded": succeeded, "failed": failed}

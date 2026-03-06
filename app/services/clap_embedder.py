"""CLAP semantic audio embeddings service."""

import logging
from pathlib import Path

import librosa
import numpy as np
import torch

from app.config import get_settings

logger = logging.getLogger(__name__)


class ClapEmbedder:
    """CLAP contrastive language-audio embedding generator."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self._model: object | None = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load_model(self) -> None:
        """Load CLAP model (lazy loading)."""
        if self._model is not None:
            return

        try:
            from msclap import CLAP

            self._model = CLAP(
                version=self.settings.clap_model_name,
                use_cuda=torch.cuda.is_available(),
            )
            logger.info("CLAP model loaded on %s", self._device)
        except Exception:
            logger.exception("Failed to load CLAP model")
            raise

    async def get_audio_embedding(self, audio_path: Path) -> list[float]:
        """Generate a 512-dim audio embedding for a track."""
        self._load_model()

        logger.info("Generating CLAP embedding for %s", audio_path)

        audio_embeddings = self._model.get_audio_embeddings([str(audio_path)])  # type: ignore[union-attr]
        embedding = audio_embeddings[0].cpu().numpy().tolist()

        return embedding

    async def get_text_embedding(self, text: str) -> list[float]:
        """Generate a 512-dim text embedding for similarity search."""
        self._load_model()

        text_embeddings = self._model.get_text_embeddings([text])  # type: ignore[union-attr]
        embedding = text_embeddings[0].cpu().numpy().tolist()

        return embedding

    async def classify_audio(
        self, audio_path: Path, labels: list[str]
    ) -> list[tuple[str, float]]:
        """Zero-shot classify audio against a set of text labels."""
        self._load_model()

        logger.info("CLAP zero-shot classification with %d labels", len(labels))

        audio_emb = self._model.get_audio_embeddings([str(audio_path)])  # type: ignore[union-attr]
        text_embs = self._model.get_text_embeddings(labels)  # type: ignore[union-attr]

        # Cosine similarity
        audio_emb_norm = audio_emb / audio_emb.norm(dim=-1, keepdim=True)
        text_embs_norm = text_embs / text_embs.norm(dim=-1, keepdim=True)
        similarities = (audio_emb_norm @ text_embs_norm.T).squeeze(0)

        scores = similarities.cpu().numpy()
        results = sorted(zip(labels, scores.tolist()), key=lambda x: -x[1])

        return results

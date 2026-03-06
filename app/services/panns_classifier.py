"""PANNs Cnn14 audio classification service.

Tier 2 of the audio analysis pipeline — AudioSet-based audio tagging
that classifies tracks into 527 classes covering genres, moods, instruments,
and sound events. Also extracts 2048-dim embeddings for similarity search.
"""

import logging
import time
from pathlib import Path

import librosa
import numpy as np
import torch

from app.models.schemas import PannsFeatures, PannsResult

logger = logging.getLogger(__name__)

# AudioSet music-relevant class indices and names
GENRE_TAGS = {
    137: "Music", 138: "Musical instrument", 288: "Pop music", 289: "Hip hop music",
    290: "Rock music", 296: "Rhythm and blues", 297: "Soul music", 298: "Reggae",
    299: "Country", 300: "Funk", 301: "Folk music", 302: "Middle Eastern music",
    303: "Jazz", 304: "Disco", 305: "Classical music", 306: "Electronic music",
    308: "House music", 310: "Techno", 318: "Punk rock", 319: "Drum and bass",
    325: "Heavy metal", 330: "Latin music", 340: "Blues",
}

MOOD_TAGS = {
    341: "Happy music", 342: "Funny music", 343: "Sad music",
    344: "Tender music", 345: "Exciting music", 346: "Angry music", 347: "Scary music",
}

INSTRUMENT_TAGS = {
    139: "Plucked string instrument", 140: "Guitar", 141: "Electric guitar",
    142: "Bass guitar", 143: "Acoustic guitar", 149: "Piano",
    151: "Keyboard (musical)", 153: "Drum kit", 154: "Drum machine",
    159: "Drum", 160: "Snare drum", 161: "Bass drum", 163: "Hi-hat",
    164: "Cymbal", 170: "Violin, fiddle", 171: "Cello",
    180: "Trumpet", 181: "Trombone", 184: "Saxophone", 186: "Flute",
    191: "Synthesizer", 192: "Singing",
}

VOCAL_TAGS = {
    192: "Singing",
    193: "Male singing",
    194: "Female singing",
    0: "Speech",
    132: "Vocal music",
    133: "A capella",
}

# All tags combined for reverse lookup
ALL_TAGS = {**GENRE_TAGS, **MOOD_TAGS, **INSTRUMENT_TAGS, **VOCAL_TAGS}

# PANNs sample rate
PANNS_SAMPLE_RATE = 32000


class PannsClassifier:
    """PANNs Cnn14 audio classification with GPU support and embedding extraction."""

    def __init__(self, model_dir: str = "./models") -> None:
        self.model_dir = Path(model_dir)
        self._model: object | None = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load_model(self) -> None:
        """Load PANNs model (lazy loading with warm-up)."""
        if self._model is not None:
            return

        try:
            from panns_inference import AudioTagging

            self._model = AudioTagging(
                checkpoint_path=None,  # Downloads ~300MB checkpoint automatically
                device=str(self._device),
            )
            logger.info("PANNs Cnn14 loaded on %s", self._device)

            # Warm up model with a short dummy input
            dummy = np.zeros((1, PANNS_SAMPLE_RATE), dtype=np.float32)
            self._model.inference(dummy)
            logger.info("PANNs model warmed up")

        except Exception:
            logger.exception("Failed to load PANNs model")
            raise

    async def classify(self, audio_path: Path) -> PannsFeatures:
        """Classify audio using PANNs Cnn14.

        Extracts 527 AudioSet tag probabilities, then maps to genres,
        moods, instruments, and vocal presence.
        """
        self._load_model()

        logger.info("Classifying audio with PANNs: %s", audio_path)
        start_time = time.monotonic()

        # Load and resample audio for PANNs (32kHz mono)
        y, _ = librosa.load(str(audio_path), sr=PANNS_SAMPLE_RATE, mono=True)
        audio_input = y[np.newaxis, :]

        # Run inference - returns (clipwise_output, embedding)
        clipwise_output, embedding = self._model.inference(audio_input)  # type: ignore[union-attr]
        probs = clipwise_output[0]

        # Extract music-relevant tags
        top_tags = self._extract_top_tags(probs, n=20)
        genres = self._extract_category(probs, GENRE_TAGS, n=5)
        moods = self._extract_category(probs, MOOD_TAGS, n=3)
        instruments = self._extract_category(probs, INSTRUMENT_TAGS, n=10)

        # Vocal detection
        singing_prob = float(probs[192]) if len(probs) > 192 else 0.0
        male_singing = float(probs[193]) if len(probs) > 193 else 0.0
        female_singing = float(probs[194]) if len(probs) > 194 else 0.0

        elapsed = time.monotonic() - start_time
        logger.info("PANNs classification completed in %.1fs on %s", elapsed, self._device)

        return PannsFeatures(
            top_tags=top_tags,
            genres=genres,
            moods=moods,
            instruments=instruments,
            is_vocal=singing_prob,
            male_singing=male_singing,
            female_singing=female_singing,
        )

    async def get_embedding(self, audio_path: Path) -> list[float]:
        """Extract 2048-dim embedding from PANNs Cnn14 penultimate layer.

        These embeddings capture the general audio characteristics and can
        be used for track-to-track similarity search.
        """
        self._load_model()

        y, _ = librosa.load(str(audio_path), sr=PANNS_SAMPLE_RATE, mono=True)
        audio_input = y[np.newaxis, :]

        _, embedding = self._model.inference(audio_input)  # type: ignore[union-attr]
        return embedding[0].tolist()

    def _extract_top_tags(self, probs: np.ndarray, n: int = 20) -> list[PannsResult]:
        """Extract top N tags across all 527 AudioSet classes."""
        top_indices = np.argsort(probs)[::-1][:n]
        return [
            PannsResult(
                tag=ALL_TAGS.get(int(idx), f"class_{idx}"),
                probability=float(probs[idx]),
            )
            for idx in top_indices
        ]

    def _extract_category(
        self, probs: np.ndarray, tag_map: dict[int, str], n: int = 5
    ) -> list[PannsResult]:
        """Extract top tags from a specific category."""
        results = [
            PannsResult(tag=name, probability=float(probs[idx]))
            for idx, name in tag_map.items()
        ]
        results.sort(key=lambda r: r.probability, reverse=True)
        return results[:n]

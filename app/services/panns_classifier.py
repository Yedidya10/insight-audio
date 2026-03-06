"""PANNs Cnn14 audio classification service."""

import logging
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


class PannsClassifier:
    """PANNs Cnn14 audio classification."""

    def __init__(self, model_dir: str = "./models") -> None:
        self.model_dir = Path(model_dir)
        self._model: torch.nn.Module | None = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load_model(self) -> None:
        """Load PANNs model (lazy loading)."""
        if self._model is not None:
            return

        try:
            from panns_inference import AudioTagging

            self._model = AudioTagging(
                checkpoint_path=None,  # Will download automatically
                device=str(self._device),
            )
            logger.info("PANNs model loaded on %s", self._device)
        except Exception:
            logger.exception("Failed to load PANNs model")
            raise

    async def classify(self, audio_path: Path, sample_rate: int = 32000) -> PannsFeatures:
        """Classify audio using PANNs Cnn14."""
        self._load_model()

        logger.info("Classifying audio with PANNs: %s", audio_path)

        # Load and resample audio for PANNs (32kHz mono)
        y, _ = librosa.load(str(audio_path), sr=sample_rate, mono=True)
        audio_input = y[np.newaxis, :]

        # Run inference
        clipwise_output, _ = self._model.inference(audio_input)  # type: ignore[union-attr]
        probs = clipwise_output[0]

        # Extract music-relevant tags
        top_tags = self._extract_top_tags(probs, n=20)
        genres = self._extract_category(probs, GENRE_TAGS, n=5)
        moods = self._extract_category(probs, MOOD_TAGS, n=3)
        instruments = self._extract_category(probs, INSTRUMENT_TAGS, n=10)

        return PannsFeatures(
            top_tags=top_tags,
            genres=genres,
            moods=moods,
            instruments=instruments,
        )

    def _extract_top_tags(self, probs: np.ndarray, n: int = 20) -> list[PannsResult]:
        """Extract top N tags across all AudioSet classes."""
        top_indices = np.argsort(probs)[::-1][:n]
        results = []
        for idx in top_indices:
            # Map index to AudioSet label name (simplified)
            all_tags = {**GENRE_TAGS, **MOOD_TAGS, **INSTRUMENT_TAGS}
            tag_name = all_tags.get(int(idx), f"class_{idx}")
            results.append(PannsResult(tag=tag_name, probability=float(probs[idx])))
        return results

    def _extract_category(
        self, probs: np.ndarray, tag_map: dict[int, str], n: int = 5
    ) -> list[PannsResult]:
        """Extract top tags from a specific category."""
        results = []
        for idx, name in tag_map.items():
            results.append(PannsResult(tag=name, probability=float(probs[idx])))
        results.sort(key=lambda r: r.probability, reverse=True)
        return results[:n]

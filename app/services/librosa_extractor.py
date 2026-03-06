"""Librosa audio feature extraction service."""

import logging
from pathlib import Path

import librosa
import numpy as np

from app.models.schemas import LibrosaFeatures

logger = logging.getLogger(__name__)

# Key names for pitch class mapping
KEY_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Krumhansl-Schmuckler key profiles
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


class LibrosaExtractor:
    """Extracts audio features using librosa."""

    def __init__(self, sample_rate: int = 22050) -> None:
        self.sample_rate = sample_rate

    async def extract(self, audio_path: Path) -> LibrosaFeatures:
        """Extract all librosa features from an audio file."""
        logger.info("Extracting librosa features from %s", audio_path)

        y, sr = librosa.load(str(audio_path), sr=self.sample_rate, mono=True)
        duration = librosa.get_duration(y=y, sr=sr)

        # Tempo and beats
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        if isinstance(tempo, np.ndarray):
            tempo = float(tempo[0])

        # Key detection
        key, key_confidence, mode = self._detect_key(y, sr)

        # Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]

        # MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_means = np.mean(mfccs, axis=1)

        # Chroma
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_means = np.mean(chroma, axis=1)

        # Energy
        rms = librosa.feature.rms(y=y)[0]

        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(y=y)[0]

        # Onset strength
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)

        return LibrosaFeatures(
            tempo=float(tempo),
            beat_count=len(beats),
            key=key,
            key_confidence=float(key_confidence),
            mode=mode,
            spectral_centroid_mean=float(np.mean(spectral_centroid)),
            spectral_centroid_std=float(np.std(spectral_centroid)),
            spectral_bandwidth_mean=float(np.mean(spectral_bandwidth)),
            spectral_rolloff_mean=float(np.mean(spectral_rolloff)),
            mfccs=[float(x) for x in mfcc_means],
            chroma=[float(x) for x in chroma_means],
            rms_energy_mean=float(np.mean(rms)),
            rms_energy_std=float(np.std(rms)),
            zero_crossing_rate_mean=float(np.mean(zcr)),
            onset_strength_mean=float(np.mean(onset_env)),
            duration_seconds=float(duration),
        )

    def _detect_key(self, y: np.ndarray, sr: int) -> tuple[str, float, str]:
        """Detect musical key using Krumhansl-Schmuckler algorithm."""
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)

        best_corr = -1.0
        best_key = 0
        best_mode = "major"

        for i in range(12):
            major_corr = float(np.corrcoef(np.roll(MAJOR_PROFILE, i), chroma_mean)[0, 1])
            minor_corr = float(np.corrcoef(np.roll(MINOR_PROFILE, i), chroma_mean)[0, 1])

            if major_corr > best_corr:
                best_corr = major_corr
                best_key = i
                best_mode = "major"

            if minor_corr > best_corr:
                best_corr = minor_corr
                best_key = i
                best_mode = "minor"

        key_name = KEY_NAMES[best_key]
        return key_name, best_corr, best_mode

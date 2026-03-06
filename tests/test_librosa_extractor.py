"""Tests for librosa feature extraction."""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from app.services.librosa_extractor import LibrosaExtractor, KEY_NAMES, MAJOR_PROFILE, MINOR_PROFILE


def test_key_names_count():
    assert len(KEY_NAMES) == 12


def test_key_profiles_shape():
    assert len(MAJOR_PROFILE) == 12
    assert len(MINOR_PROFILE) == 12


def test_extractor_init():
    extractor = LibrosaExtractor(sample_rate=22050)
    assert extractor.sample_rate == 22050


def test_extractor_custom_sample_rate():
    extractor = LibrosaExtractor(sample_rate=16000)
    assert extractor.sample_rate == 16000


def test_key_detection():
    """Test key detection with a synthetic C major signal."""
    extractor = LibrosaExtractor()
    sr = 22050
    duration = 3.0
    # Generate a C4 note (261.63 Hz)
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    y = np.sin(2 * np.pi * 261.63 * t).astype(np.float32)

    key, confidence, mode = extractor._detect_key(y, sr)
    assert key in KEY_NAMES
    assert -1.0 <= confidence <= 1.0
    assert mode in ("major", "minor")


@pytest.mark.asyncio
async def test_extract_synthetic_audio():
    """Test full extraction pipeline with a synthetic audio file."""
    extractor = LibrosaExtractor(sample_rate=22050)
    sr = 22050
    duration = 2.0

    # Generate a simple sine wave
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    y = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)

    # Write to a temp file
    import soundfile as sf

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, y, sr)
        audio_path = Path(f.name)

    try:
        features = await extractor.extract(audio_path)

        assert features.tempo > 0
        assert features.duration_seconds > 0
        assert features.key in KEY_NAMES
        assert features.mode in ("major", "minor")
        assert len(features.mfccs) == 13
        assert len(features.mfcc_vars) == 13
        assert len(features.chroma) == 12
        assert len(features.tonnetz) == 6
        assert features.rms_energy_mean > 0
        assert features.spectral_centroid_mean > 0
        assert features.bpm_confidence >= 0
    finally:
        audio_path.unlink(missing_ok=True)

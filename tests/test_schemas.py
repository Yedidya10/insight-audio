"""Tests for Pydantic schemas."""

from app.models.schemas import (
    AnalysisRequest,
    AnalysisResponse,
    LibrosaFeatures,
    PannsFeatures,
    PannsResult,
    TrackAnalysisResult,
)


def test_analysis_request():
    req = AnalysisRequest(
        track_id="abc123",
        track_name="Test Song",
        artist_name="Test Artist",
    )
    assert req.track_id == "abc123"
    assert req.priority == 0
    assert req.user_id is None


def test_analysis_response():
    resp = AnalysisResponse(
        track_id="abc123",
        status="queued",
        message="Analysis queued",
    )
    assert resp.status == "queued"


def test_librosa_features():
    features = LibrosaFeatures(
        tempo=120.0,
        beat_count=100,
        bpm_confidence=0.75,
        key="C",
        key_confidence=0.85,
        mode="major",
        spectral_centroid_mean=2000.0,
        spectral_centroid_std=500.0,
        spectral_bandwidth_mean=3000.0,
        spectral_rolloff_mean=5000.0,
        spectral_flatness_mean=0.01,
        mfccs=[0.0] * 13,
        mfcc_vars=[0.0] * 13,
        chroma=[0.0] * 12,
        tonnetz=[0.0] * 6,
        rms_energy_mean=0.1,
        rms_energy_std=0.05,
        zero_crossing_rate_mean=0.1,
        onset_strength_mean=5.0,
        duration_seconds=240.0,
    )
    assert features.tempo == 120.0
    assert features.key == "C"
    assert len(features.mfccs) == 13
    assert len(features.mfcc_vars) == 13
    assert len(features.tonnetz) == 6
    assert features.bpm_confidence == 0.75


def test_panns_features():
    features = PannsFeatures(
        top_tags=[PannsResult(tag="Music", probability=0.95)],
        genres=[PannsResult(tag="Rock music", probability=0.8)],
        moods=[PannsResult(tag="Happy music", probability=0.6)],
        instruments=[PannsResult(tag="Guitar", probability=0.7)],
    )
    assert features.genres[0].tag == "Rock music"


def test_track_analysis_result():
    result = TrackAnalysisResult(track_id="abc123")
    assert result.analysis_version == "1.0.0"
    assert result.librosa_features is None

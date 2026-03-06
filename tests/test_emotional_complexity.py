"""Tests for the multimodal emotional complexity engine."""

from unittest.mock import MagicMock, patch

import pytest

from app.services.emotional_complexity import (
    AudioFeatures,
    TextAnalysis,
    DissonanceResult,
    DEFAULT_ALPHA,
    DEFAULT_BETA,
    COMPLEXITY_THRESHOLD,
    EMA_DECAY,
    normalize_sentiment,
    derive_valence,
    derive_energy,
    calculate_dissonance,
    update_eci,
    update_openness,
    store_interaction,
    update_user_profile,
    process_track_emotion,
)


# ------------------------------------------------------------------
# normalize_sentiment
# ------------------------------------------------------------------

class TestNormalizeSentiment:
    def test_negative_one(self):
        assert normalize_sentiment(-1.0) == 0.0

    def test_zero(self):
        assert normalize_sentiment(0.0) == 0.5

    def test_positive_one(self):
        assert normalize_sentiment(1.0) == 1.0

    def test_mid_positive(self):
        assert normalize_sentiment(0.5) == 0.75

    def test_mid_negative(self):
        assert normalize_sentiment(-0.5) == 0.25

    def test_clamps_below_range(self):
        assert normalize_sentiment(-2.0) == 0.0

    def test_clamps_above_range(self):
        assert normalize_sentiment(2.0) == 1.0


# ------------------------------------------------------------------
# derive_valence
# ------------------------------------------------------------------

class TestDeriveValence:
    def test_all_happy(self):
        v = derive_valence(mood_happy=1.0)
        assert 0.7 < v <= 1.0  # should be high

    def test_all_sad(self):
        v = derive_valence(mood_sad=1.0)
        assert 0.0 <= v < 0.3  # should be low

    def test_neutral(self):
        v = derive_valence()
        assert v == 0.5  # offset of 0.5

    def test_clamped_to_01(self):
        v = derive_valence(mood_happy=1.0, mood_funny=1.0, mood_tender=1.0)
        assert v <= 1.0


# ------------------------------------------------------------------
# derive_energy
# ------------------------------------------------------------------

class TestDeriveEnergy:
    def test_high_exciting_high_energy(self):
        e = derive_energy(mood_exciting=1.0, rms_energy_mean=0.2, bpm=180.0)
        assert e > 0.6

    def test_calm_low_energy(self):
        e = derive_energy(mood_exciting=0.0, rms_energy_mean=0.01, bpm=70.0)
        assert e < 0.2

    def test_clamped_to_01(self):
        e = derive_energy(mood_exciting=1.0, mood_angry=1.0,
                          rms_energy_mean=1.0, spectral_centroid_mean=10000.0, bpm=200.0)
        assert e <= 1.0
        assert e >= 0.0


# ------------------------------------------------------------------
# calculate_dissonance
# ------------------------------------------------------------------

class TestCalculateDissonance:
    def test_audio_only_mode(self):
        """No text → audio-only intra-modal dissonance."""
        audio = AudioFeatures(valence=0.2, energy=0.9)
        result = calculate_dissonance(audio)
        assert result.mode == "audio_only"
        assert abs(result.score - 0.7) < 1e-5  # |0.9 - 0.2|
        assert result.cross_modal_gap == 0.0
        assert abs(result.intra_modal_gap - 0.7) < 1e-5

    def test_audio_only_with_none_text(self):
        audio = AudioFeatures(valence=0.5, energy=0.5)
        text = TextAnalysis(source="none")
        result = calculate_dissonance(audio, text)
        assert result.mode == "audio_only"
        assert result.score == 0.0

    def test_full_mode_with_text(self):
        """Full formula with text sentiment."""
        audio = AudioFeatures(valence=0.2, energy=0.3)
        text = TextAnalysis(sentiment_score=0.8, source="lyrics")
        result = calculate_dissonance(audio, text)

        s_norm = normalize_sentiment(0.8)  # 0.9
        expected_cross = abs(0.2 - s_norm)  # |0.2 - 0.9| = 0.7
        expected_intra = abs(0.3 - 0.2)     # 0.1
        expected_d = DEFAULT_ALPHA * expected_cross + DEFAULT_BETA * expected_intra

        assert result.mode == "full"
        assert abs(result.score - expected_d) < 1e-5

    def test_happy_song_happy_lyrics_low_dissonance(self):
        """Example from the doc: happy song + happy lyrics → low D."""
        audio = AudioFeatures(valence=0.8, energy=0.7)
        text = TextAnalysis(sentiment_score=0.8, source="lyrics")
        result = calculate_dissonance(audio, text)
        assert result.score < 0.2
        assert not result.is_complex

    def test_sad_melody_uplifting_lyrics(self):
        """Sad melody + positive lyrics → moderate dissonance."""
        audio = AudioFeatures(valence=0.2, energy=0.3)
        text = TextAnalysis(sentiment_score=0.8, source="lyrics")
        result = calculate_dissonance(audio, text)
        assert result.score > 0.3

    def test_aggressive_positive_lyrics_complex(self):
        """Aggressive bass + positive lyrics → high dissonance."""
        audio = AudioFeatures(valence=0.15, energy=0.95)
        text = TextAnalysis(sentiment_score=0.9, source="lyrics")
        result = calculate_dissonance(audio, text)
        # Cross: |0.15 - 0.95| = 0.80 * 0.6 = 0.48
        # Intra: |0.95 - 0.15| = 0.80 * 0.4 = 0.32
        # D = 0.80
        assert result.score > 0.6
        assert result.is_complex

    def test_custom_weights(self):
        audio = AudioFeatures(valence=0.5, energy=0.5)
        text = TextAnalysis(sentiment_score=0.0, source="lyrics")
        # S_norm = 0.5 → cross=0, intra=0
        result = calculate_dissonance(audio, text, alpha=0.9, beta=0.1)
        assert result.score == 0.0

    def test_score_bounded_zero_one(self):
        audio = AudioFeatures(valence=0.0, energy=1.0)
        text = TextAnalysis(sentiment_score=1.0, source="lyrics")
        result = calculate_dissonance(audio, text)
        assert 0.0 <= result.score <= 1.0

    def test_complexity_threshold(self):
        audio = AudioFeatures(valence=0.3, energy=0.3)
        # Audio-only: D = 0.0
        result = calculate_dissonance(audio, threshold=0.5)
        assert not result.is_complex


# ------------------------------------------------------------------
# update_eci
# ------------------------------------------------------------------

class TestUpdateECI:
    def test_first_interaction(self):
        assert update_eci(0.0, 0.8, total_interactions=0) == 0.8

    def test_subsequent_interaction_ema(self):
        eci = update_eci(0.5, 0.8, total_interactions=5)
        expected = 0.5 * EMA_DECAY + 0.8 * (1 - EMA_DECAY)
        assert abs(eci - expected) < 1e-10

    def test_convergence(self):
        """After many interactions at 0.7, ECI should converge to ~0.7."""
        eci = 0.0
        for i in range(100):
            eci = update_eci(eci, 0.7, total_interactions=i)
        assert abs(eci - 0.7) < 0.01


# ------------------------------------------------------------------
# update_openness
# ------------------------------------------------------------------

class TestUpdateOpenness:
    def test_no_change_when_not_complex(self):
        assert update_openness(0.5, is_complex=False, eci=0.8) == 0.5

    def test_no_change_when_eci_low(self):
        assert update_openness(0.5, is_complex=True, eci=0.3) == 0.5

    def test_increments_when_complex_and_high_eci(self):
        result = update_openness(0.5, is_complex=True, eci=0.8)
        assert abs(result - 0.52) < 1e-10

    def test_capped_at_one(self):
        result = update_openness(0.99, is_complex=True, eci=0.9)
        assert result == 1.0


# ------------------------------------------------------------------
# Supabase integration (mocked)
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_store_interaction():
    audio = AudioFeatures(valence=0.5, energy=0.6, bpm=120.0)
    result = DissonanceResult(
        score=0.1, is_complex=False, cross_modal_gap=0.0,
        intra_modal_gap=0.1, mode="audio_only",
    )

    with patch("app.services.emotional_complexity.get_supabase_client") as mock_client:
        mock_table = MagicMock()
        mock_resp = MagicMock()
        mock_resp.data = [{"id": "rec-1"}]
        mock_table.insert.return_value.execute.return_value = mock_resp
        mock_client.return_value.table.return_value = mock_table

        record_id = await store_interaction(
            user_id="user-1",
            audio=audio,
            result=result,
            track_fingerprint="fp-1",
            track_name="Test Track",
            artist_name="Test Artist",
        )

    assert record_id == "rec-1"
    insert_data = mock_table.insert.call_args[0][0]
    assert insert_data["user_id"] == "user-1"
    assert insert_data["audio_valence"] == 0.5
    assert insert_data["multimodal_dissonance_score"] == 0.1
    assert insert_data["is_complex"] is False


@pytest.mark.asyncio
async def test_update_user_profile_first_time():
    """First interaction creates a new profile."""
    with patch("app.services.emotional_complexity.get_supabase_client") as mock_client:
        mock_table = MagicMock()
        # No existing profile
        mock_select = MagicMock()
        mock_select.data = []
        mock_table.select.return_value.eq.return_value.limit.return_value.execute.return_value = mock_select
        mock_table.insert.return_value.execute.return_value = MagicMock()
        mock_client.return_value.table.return_value = mock_table

        await update_user_profile("user-1", 0.7, is_complex=True)

    insert_data = mock_table.insert.call_args[0][0]
    assert insert_data["emotional_complexity_index"] == 0.7
    assert insert_data["total_analyzed_interactions"] == 1


@pytest.mark.asyncio
async def test_update_user_profile_subsequent():
    """Subsequent interaction updates ECI via EMA."""
    with patch("app.services.emotional_complexity.get_supabase_client") as mock_client:
        mock_table = MagicMock()
        mock_select = MagicMock()
        mock_select.data = [{
            "emotional_complexity_index": 0.5,
            "total_analyzed_interactions": 10,
            "openness_score": 0.6,
            "max_dissonance": 0.8,
            "avg_dissonance": 0.4,
            "high_dissonance_ratio": 0.3,
        }]
        mock_table.select.return_value.eq.return_value.limit.return_value.execute.return_value = mock_select
        mock_table.update.return_value.eq.return_value.execute.return_value = MagicMock()
        mock_client.return_value.table.return_value = mock_table

        await update_user_profile("user-1", 0.6, is_complex=False)

    update_data = mock_table.update.call_args[0][0]
    expected_eci = 0.5 * 0.9 + 0.6 * 0.1
    assert abs(update_data["emotional_complexity_index"] - expected_eci) < 1e-10
    assert update_data["total_analyzed_interactions"] == 11


@pytest.mark.asyncio
async def test_process_track_emotion_full_pipeline():
    """End-to-end: compute + store + update."""
    audio = AudioFeatures(valence=0.3, energy=0.8)
    text = TextAnalysis(sentiment_score=0.9, source="lyrics")

    with patch("app.services.emotional_complexity.get_supabase_client") as mock_client:
        mock_table = MagicMock()
        # store_interaction
        mock_insert_resp = MagicMock()
        mock_insert_resp.data = [{"id": "rec-2"}]
        mock_table.insert.return_value.execute.return_value = mock_insert_resp
        # update_user_profile - no existing profile
        mock_select = MagicMock()
        mock_select.data = []
        mock_table.select.return_value.eq.return_value.limit.return_value.execute.return_value = mock_select
        mock_client.return_value.table.return_value = mock_table

        result = await process_track_emotion(
            user_id="user-1",
            track_fingerprint="fp-1",
            track_name="Test",
            artist_name="Artist",
            audio=audio,
            text=text,
        )

    assert result.mode == "full"
    assert result.score > 0  # non-trivial dissonance
    # Should have called insert for interaction + insert for new profile
    assert mock_table.insert.call_count == 2

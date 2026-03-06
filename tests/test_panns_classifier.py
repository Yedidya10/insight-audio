"""Tests for PANNs Cnn14 classifier."""

import numpy as np

from app.services.panns_classifier import (
    ALL_TAGS,
    GENRE_TAGS,
    INSTRUMENT_TAGS,
    MOOD_TAGS,
    PANNS_SAMPLE_RATE,
    VOCAL_TAGS,
    PannsClassifier,
)


def test_tag_maps_have_entries():
    assert len(GENRE_TAGS) > 15
    assert len(MOOD_TAGS) == 7
    assert len(INSTRUMENT_TAGS) > 15
    assert len(VOCAL_TAGS) > 0


def test_all_tags_combines_maps():
    expected_len = len(set(GENRE_TAGS) | set(MOOD_TAGS) | set(INSTRUMENT_TAGS) | set(VOCAL_TAGS))
    assert len(ALL_TAGS) == expected_len


def test_panns_sample_rate():
    assert PANNS_SAMPLE_RATE == 32000


def test_classifier_init():
    classifier = PannsClassifier(model_dir="./test-models")
    assert classifier.model_dir.name == "test-models"
    assert classifier._model is None


def test_extract_category():
    classifier = PannsClassifier()
    # Simulate a probability array
    probs = np.zeros(527)
    probs[303] = 0.85  # Jazz
    probs[288] = 0.65  # Pop
    probs[290] = 0.50  # Rock

    results = classifier._extract_category(probs, GENRE_TAGS, n=3)
    assert len(results) == 3
    assert results[0].tag == "Jazz"
    assert results[0].probability == 0.85


def test_extract_top_tags():
    classifier = PannsClassifier()
    probs = np.zeros(527)
    probs[303] = 0.9  # Jazz
    probs[341] = 0.7  # Happy music
    probs[149] = 0.8  # Piano

    results = classifier._extract_top_tags(probs, n=3)
    assert len(results) == 3
    assert results[0].probability == 0.9  # Highest prob first
    assert results[0].tag == "Jazz"


def test_mood_tags_complete():
    """Ensure all 7 AudioSet mood classes are mapped."""
    mood_names = set(MOOD_TAGS.values())
    expected = {"Happy music", "Funny music", "Sad music", "Tender music",
                "Exciting music", "Angry music", "Scary music"}
    assert mood_names == expected

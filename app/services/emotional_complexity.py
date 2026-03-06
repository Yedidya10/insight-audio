"""Multimodal Emotional Complexity Engine.

Computes emotional dissonance scores from audio features and optional text
sentiment, tracking cross-modal (lyrics vs melody) and intra-modal (energy vs
melody) tension.

Formula:
  Full:       D = alpha * |V_a - S_norm| + beta * |E_a - V_a|
  Audio-only: D = |E_a - V_a|

Where:
  V_a    = audio valence  (0-1, from PANNs mood scores + librosa)
  E_a    = audio energy   (0-1, from PANNs + librosa RMS/BPM)
  S_norm = (S_t + 1) / 2  (text sentiment normalized from [-1,+1] to [0,1])

Ref: docs/MULTIMODAL_EMOTIONAL_COMPLEXITY.md Section 4
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

from app.services.supabase_client import get_supabase_client

logger = logging.getLogger(__name__)

# Default formula weights
DEFAULT_ALPHA = 0.6  # cross-modal weight (text vs melody)
DEFAULT_BETA = 0.4   # intra-modal weight (energy vs melody)

# Complexity threshold
COMPLEXITY_THRESHOLD = 0.6

# EMA decay factor for emotional complexity index
EMA_DECAY = 0.9

# Openness score increment when complex + high ECI
OPENNESS_INCREMENT = 0.02
OPENNESS_ECI_THRESHOLD = 0.65

# PANNs mood tags → valence/energy weights
VALENCE_WEIGHTS = {
    "mood_happy": 0.35,
    "mood_funny": 0.10,
    "mood_tender": 0.10,
    "mood_sad": -0.30,
    "mood_angry": -0.10,
    "mood_scary": -0.05,
}

ENERGY_WEIGHTS = {
    "mood_exciting": 0.30,
    "mood_angry": 0.15,
    # rms, spectral_centroid, bpm handled via normalize_energy()
}


@dataclass
class AudioFeatures:
    """Audio-derived emotional features from track_audio_analysis."""

    valence: float  # 0-1
    energy: float   # 0-1

    # Optional PANNs mood breakdown
    mood_happy: float | None = None
    mood_sad: float | None = None
    mood_tender: float | None = None
    mood_exciting: float | None = None
    mood_angry: float | None = None

    # Sonic context
    bpm: float | None = None
    estimated_key: str | None = None
    primary_genre: str | None = None
    primary_mood: str | None = None


@dataclass
class TextAnalysis:
    """Text-derived emotional features (lyrics or chat)."""

    sentiment_score: float = 0.0   # -1 to +1
    dominant_emotions: list[str] = field(default_factory=list)
    source: str = "none"           # "lyrics", "chat", "none"


@dataclass
class DissonanceResult:
    """Result of a dissonance calculation."""

    score: float                # 0-1
    is_complex: bool            # D > threshold
    cross_modal_gap: float      # |V_a - S_norm|
    intra_modal_gap: float      # |E_a - V_a|
    mode: str                   # "full" or "audio_only"


# ------------------------------------------------------------------
# Core algorithmic functions
# ------------------------------------------------------------------

def normalize_sentiment(sentiment: float) -> float:
    """Normalize text sentiment from [-1, +1] to [0, 1]."""
    return max(0.0, min(1.0, (sentiment + 1.0) / 2.0))


def derive_valence(
    mood_happy: float = 0.0,
    mood_funny: float = 0.0,
    mood_tender: float = 0.0,
    mood_sad: float = 0.0,
    mood_angry: float = 0.0,
    mood_scary: float = 0.0,
) -> float:
    """Derive valence (0-1) from PANNs mood scores using weighted sum."""
    raw = (
        VALENCE_WEIGHTS["mood_happy"] * mood_happy
        + VALENCE_WEIGHTS["mood_funny"] * mood_funny
        + VALENCE_WEIGHTS["mood_tender"] * mood_tender
        + VALENCE_WEIGHTS["mood_sad"] * mood_sad
        + VALENCE_WEIGHTS["mood_angry"] * mood_angry
        + VALENCE_WEIGHTS["mood_scary"] * mood_scary
    )
    # Normalize to 0-1 (raw range is roughly -0.45 to +0.55)
    return max(0.0, min(1.0, raw + 0.5))


def derive_energy(
    mood_exciting: float = 0.0,
    mood_angry: float = 0.0,
    rms_energy_mean: float = 0.0,
    spectral_centroid_mean: float = 0.0,
    bpm: float = 120.0,
) -> float:
    """Derive energy/arousal (0-1) from PANNs mood + librosa features."""
    # Normalize librosa features to 0-1
    rms_norm = max(0.0, min(1.0, rms_energy_mean * 5.0))  # ~0-0.2 typical range
    sc_norm = max(0.0, min(1.0, spectral_centroid_mean / 8000.0))
    bpm_norm = max(0.0, min(1.0, (bpm - 60.0) / 140.0))

    raw = (
        0.30 * mood_exciting
        + 0.15 * mood_angry
        + 0.25 * rms_norm
        + 0.15 * sc_norm
        + 0.15 * bpm_norm
    )
    return max(0.0, min(1.0, raw))


def calculate_dissonance(
    audio: AudioFeatures,
    text: TextAnalysis | None = None,
    *,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
    threshold: float = COMPLEXITY_THRESHOLD,
) -> DissonanceResult:
    """Calculate the multimodal dissonance score.

    When text is available (source != "none"):
      D = alpha * |V_a - S_norm| + beta * |E_a - V_a|

    When text is not available (audio-only mode):
      D = |E_a - V_a|

    Returns a DissonanceResult with score, is_complex flag, and gap breakdowns.
    """
    intra_modal_gap = abs(audio.energy - audio.valence)

    if text is None or text.source == "none":
        # Audio-only fallback
        score = intra_modal_gap
        cross_modal_gap = 0.0
        mode = "audio_only"
    else:
        s_norm = normalize_sentiment(text.sentiment_score)
        cross_modal_gap = abs(audio.valence - s_norm)
        score = alpha * cross_modal_gap + beta * intra_modal_gap
        mode = "full"

    # Ensure bounded
    score = max(0.0, min(1.0, score))

    return DissonanceResult(
        score=score,
        is_complex=score > threshold,
        cross_modal_gap=cross_modal_gap,
        intra_modal_gap=intra_modal_gap,
        mode=mode,
    )


def update_eci(
    current_eci: float,
    new_dissonance: float,
    total_interactions: int,
) -> float:
    """Update the Emotional Complexity Index using exponential moving average.

    First interaction:  ECI = D
    Subsequent:         ECI = ECI * 0.9 + D * 0.1
    """
    if total_interactions == 0:
        return new_dissonance
    return current_eci * EMA_DECAY + new_dissonance * (1.0 - EMA_DECAY)


def update_openness(
    current_openness: float,
    is_complex: bool,
    eci: float,
) -> float:
    """Increment openness score when complex + high ECI."""
    if is_complex and eci > OPENNESS_ECI_THRESHOLD:
        return min(current_openness + OPENNESS_INCREMENT, 1.0)
    return current_openness


# ------------------------------------------------------------------
# Supabase integration
# ------------------------------------------------------------------

async def store_interaction(
    *,
    user_id: str,
    audio: AudioFeatures,
    result: DissonanceResult,
    text: TextAnalysis | None = None,
    track_fingerprint: str | None = None,
    track_name: str | None = None,
    artist_name: str | None = None,
    interaction_type: str = "track_listen",
) -> str:
    """Store an emotional interaction record in Supabase.

    Returns the record ID.
    """
    client = get_supabase_client()

    record = {
        "user_id": user_id,
        "interaction_type": interaction_type,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "track_fingerprint": track_fingerprint,
        "track_name": track_name,
        "artist_name": artist_name,
        "audio_valence": audio.valence,
        "audio_energy": audio.energy,
        "text_sentiment": text.sentiment_score if text and text.source != "none" else None,
        "dominant_emotions": text.dominant_emotions if text else [],
        "multimodal_dissonance_score": result.score,
        "is_complex": result.is_complex,
        "mood_happy": audio.mood_happy,
        "mood_sad": audio.mood_sad,
        "mood_tender": audio.mood_tender,
        "mood_exciting": audio.mood_exciting,
        "mood_angry": audio.mood_angry,
        "bpm": audio.bpm,
        "estimated_key": audio.estimated_key,
        "primary_genre": audio.primary_genre,
        "primary_mood": audio.primary_mood,
    }

    resp = client.table("emotional_interaction_records").insert(record).execute()
    record_id = resp.data[0]["id"] if resp.data else ""

    logger.info(
        "Stored emotional interaction %s for user %s: D=%.3f (%s, %s)",
        record_id, user_id, result.score, result.mode,
        "complex" if result.is_complex else "congruent",
    )
    return record_id


async def update_user_profile(
    user_id: str,
    new_dissonance: float,
    is_complex: bool,
) -> None:
    """Update the user's emotional profile after a new interaction.

    Performs an EMA update on the emotional_complexity_index and
    optionally increments the openness_score.
    """
    client = get_supabase_client()

    # Fetch current profile
    result = (
        client.table("user_emotional_profile")
        .select("*")
        .eq("user_id", user_id)
        .limit(1)
        .execute()
    )

    now = datetime.now(timezone.utc).isoformat()

    if not result.data:
        # First time — create profile
        client.table("user_emotional_profile").insert({
            "user_id": user_id,
            "emotional_complexity_index": new_dissonance,
            "total_analyzed_interactions": 1,
            "avg_dissonance": new_dissonance,
            "max_dissonance": new_dissonance,
            "high_dissonance_ratio": 1.0 if is_complex else 0.0,
            "openness_score": 0.5,
            "last_computed_at": now,
        }).execute()
        logger.info("Created emotional profile for user %s (ECI=%.3f)", user_id, new_dissonance)
        return

    profile = result.data[0]
    total = profile["total_analyzed_interactions"]
    current_eci = profile["emotional_complexity_index"]
    current_openness = profile["openness_score"]
    current_max = profile.get("max_dissonance", 0.0)
    current_avg = profile.get("avg_dissonance", 0.0)
    current_hdr = profile.get("high_dissonance_ratio", 0.0)

    # Update ECI
    new_eci = update_eci(current_eci, new_dissonance, total)
    new_total = total + 1

    # Update running average
    new_avg = (current_avg * total + new_dissonance) / new_total

    # Update max
    new_max = max(current_max, new_dissonance)

    # Update high-dissonance ratio
    complex_count = round(current_hdr * total) + (1 if is_complex else 0)
    new_hdr = complex_count / new_total

    # Update openness
    new_openness = update_openness(current_openness, is_complex, new_eci)

    client.table("user_emotional_profile").update({
        "emotional_complexity_index": new_eci,
        "total_analyzed_interactions": new_total,
        "avg_dissonance": new_avg,
        "max_dissonance": new_max,
        "high_dissonance_ratio": new_hdr,
        "openness_score": new_openness,
        "last_computed_at": now,
    }).eq("user_id", user_id).execute()

    logger.info(
        "Updated emotional profile for user %s: ECI=%.3f→%.3f, openness=%.3f, interactions=%d",
        user_id, current_eci, new_eci, new_openness, new_total,
    )


async def process_track_emotion(
    *,
    user_id: str,
    track_fingerprint: str,
    track_name: str | None = None,
    artist_name: str | None = None,
    audio: AudioFeatures,
    text: TextAnalysis | None = None,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
) -> DissonanceResult:
    """Full pipeline: compute dissonance, store interaction, update profile.

    This is the main entry point called after audio analysis completes.
    """
    result = calculate_dissonance(audio, text, alpha=alpha, beta=beta)

    await store_interaction(
        user_id=user_id,
        audio=audio,
        result=result,
        text=text,
        track_fingerprint=track_fingerprint,
        track_name=track_name,
        artist_name=artist_name,
    )

    await update_user_profile(user_id, result.score, result.is_complex)

    return result

"""Audio analysis endpoints."""

from fastapi import APIRouter, HTTPException

from app.models.schemas import (
    AnalysisRequest,
    AnalysisResponse,
    AnalysisStatusResponse,
    SearchResponse,
    SimilarTrackRequest,
    SimilarTrackResult,
    TextSearchRequest,
)
from app.services.clap_embedder import ClapEmbedder
from app.services.supabase_client import get_supabase_client

router = APIRouter()


@router.post("/analyze", response_model=AnalysisResponse)
async def request_analysis(request: AnalysisRequest) -> AnalysisResponse:
    """Queue a track for audio analysis."""
    if not request.track_id or not request.track_name or not request.artist_name:
        raise HTTPException(status_code=400, detail="track_id, track_name, and artist_name are required")

    # TODO: Queue the analysis job via Supabase
    return AnalysisResponse(
        track_id=request.track_id,
        status="queued",
        message=f"Analysis queued for '{request.artist_name} - {request.track_name}'",
    )


@router.get("/status/{track_id}", response_model=AnalysisStatusResponse)
async def get_analysis_status(track_id: str) -> AnalysisStatusResponse:
    """Get the analysis status for a track."""
    # TODO: Fetch status from Supabase
    return AnalysisStatusResponse(
        track_id=track_id,
        status="pending",
    )


@router.post("/search/text", response_model=SearchResponse)
async def search_by_text(request: TextSearchRequest) -> SearchResponse:
    """Find tracks matching a natural-language description.

    Converts the text query into a CLAP embedding and searches the
    track_audio_embeddings table via pgvector cosine similarity.

    Example: "find tracks that sound like a rainy day"
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")

    embedder = ClapEmbedder()
    matches = await embedder.search_by_text(
        request.query, limit=request.limit, threshold=request.threshold
    )

    results = [
        SimilarTrackResult(
            track_fingerprint=m["track_fingerprint"],
            similarity=m.get("similarity", 0),
            track_name=m.get("track_name"),
            artist_name=m.get("artist_name"),
        )
        for m in matches
    ]

    return SearchResponse(results=results, query_type="text", count=len(results))


@router.post("/search/similar", response_model=SearchResponse)
async def search_similar_tracks(request: SimilarTrackRequest) -> SearchResponse:
    """Find tracks similar to a given track by CLAP embedding.

    Looks up the track's existing CLAP embedding and finds the closest
    neighbours in the track_audio_embeddings table.
    """
    client = get_supabase_client()
    result = (
        client.table("track_audio_embeddings")
        .select("embedding")
        .eq("track_fingerprint", request.track_fingerprint)
        .eq("model_name", "clap-music")
        .limit(1)
        .execute()
    )

    if not result.data:
        raise HTTPException(
            status_code=404,
            detail=f"No CLAP embedding found for track '{request.track_fingerprint}'",
        )

    embedding = result.data[0]["embedding"]

    embedder = ClapEmbedder()
    matches = await embedder.find_similar_tracks(
        embedding, limit=request.limit + 1, threshold=request.threshold
    )

    # Exclude the query track from results
    matches = [m for m in matches if m["track_fingerprint"] != request.track_fingerprint]
    matches = matches[: request.limit]

    results = [
        SimilarTrackResult(
            track_fingerprint=m["track_fingerprint"],
            similarity=m.get("similarity", 0),
            track_name=m.get("track_name"),
            artist_name=m.get("artist_name"),
        )
        for m in matches
    ]

    return SearchResponse(results=results, query_type="similar", count=len(results))

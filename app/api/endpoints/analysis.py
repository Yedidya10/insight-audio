"""Audio analysis endpoints."""

from fastapi import APIRouter, HTTPException

from app.models.schemas import AnalysisRequest, AnalysisResponse, AnalysisStatusResponse

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

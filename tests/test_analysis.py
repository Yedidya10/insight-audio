"""Tests for API analysis endpoints."""

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.mark.anyio
async def test_request_analysis(client: AsyncClient):
    response = await client.post(
        "/api/v1/analyze",
        json={
            "track_id": "test-track-123",
            "track_name": "Bohemian Rhapsody",
            "artist_name": "Queen",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["track_id"] == "test-track-123"
    assert data["status"] == "queued"


@pytest.mark.anyio
async def test_request_analysis_missing_fields(client: AsyncClient):
    response = await client.post(
        "/api/v1/analyze",
        json={"track_id": "", "track_name": "", "artist_name": ""},
    )
    assert response.status_code == 400


@pytest.mark.anyio
async def test_get_analysis_status(client: AsyncClient):
    response = await client.get("/api/v1/status/test-track-123")
    assert response.status_code == 200
    data = response.json()
    assert data["track_id"] == "test-track-123"

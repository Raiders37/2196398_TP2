# tests/test_smoke.py
import pytest
from httpx import AsyncClient
from app.main import app

@pytest.mark.anyio
async def test_smoke():
    async with AsyncClient(app=app, base_url="http://test") as client:
        resp = await client.get("/")
    assert resp.status_code == 200
    assert resp.json() == {"message": "API is up and running!"}
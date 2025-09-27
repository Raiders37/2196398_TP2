# tests/test_api.py
import pytest
from httpx import AsyncClient
from app.main import app

@pytest.mark.anyio
async def test_predict_success():
    async with AsyncClient(app=app, base_url="http://test") as client:
        resp = await client.post("/predict", json={
        "features": [3.5, 1.2, 4.9]
    })
    assert resp.status_code == 200
    assert {"predictions": [7.0, 2.4, 9.8]} == resp.json()

@pytest.mark.anyio
async def test_predict_unprocessable_entity():
    async with AsyncClient(app=app, base_url="http://test") as client:
        resp = await client.post("/predict", json={
        "feature1": 3.5,
        "feature2": 1.2,
        "feature3": 4.9
    })
    assert resp.status_code == 422

# 1) Un test qui valide une prédiction correcte avec les valeurs [1.0, 2.0, 3.0].

@pytest.mark.anyio
async def test_predict_success_val():
    async with AsyncClient(app=app, base_url="http://test") as client:
        resp = await client.post("/predict", json={
        "features": [1.0, 2.0, 3.0]
    })
    assert resp.status_code == 200
    assert resp.json()["predictions"] == [2.0, 4.0, 6.0]

#############################################################################

# 2) Un test qui valide une prédiction incorrecte (ex. : comparer avec un résultat attendu volontairement faux).

@pytest.mark.anyio
async def test_predict_wrong_val():
    async with AsyncClient(app=app, base_url="http://test") as client:
        resp = await client.post("/predict", json={
        "feature1": 1.0,
        "feature2": 2.0,
        "feature3": 3.0
    })
    assert resp.status_code == 422


##############################################################################

# 3) Un test qui envoie un JSON incorrect (exemple : champ features manquant, {[3.5, 1.2, 4.9]}).

@pytest.mark.anyio
async def test_predict_missing_features():
    async with AsyncClient(app=app, base_url="http://test") as client:
        resp = await client.post("/predict", json={
            "champ_feature_manquant": [3.5, 1.2, 4.9]
        })
    assert resp.status_code == 422
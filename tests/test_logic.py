import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient

from src.api import app

client = TestClient(app)


def test_health_endpoint():
    # check health endpoint
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "operational"


def test_prediction_logic():
    # seinfg the prediction endpoint with sample data
    payload = {
        "distance_km": 10.0,
        "weight_g": 500,
        "purchase_month": 1,
        "order_hour": 12,
        "vehicle_type": "car",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    # checing response structure
    assert "estimated_delivery_minutes" in response.json()
    assert response.json()["estimated_delivery_minutes"] > 0
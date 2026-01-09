"""
Unit tests for FastAPI endpoints and prediction logic.
"""
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api import app, PredictionEngine

client = TestClient(app)


@pytest.fixture(autouse=True)
def mock_prediction_engine():
    """Mock the PredictionEngine to avoid Kaggle downloads in tests."""
    with patch.object(PredictionEngine, '__init__', return_value=None):
        with patch.object(PredictionEngine, 'is_ready', return_value=True):
            with patch.object(PredictionEngine, 'get_metrics', return_value={
                "records": 100000,
                "r2_score": 0.85,
                "mae": 3.5
            }):
                with patch.object(PredictionEngine, 'predict', return_value={
                    "predicted_days": 7.5,
                    "r2_score": 0.85,
                    "mae": 3.5,
                    "warnings": []
                }):
                    yield


def test_root_endpoint():
    """Test root endpoint returns API information."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "Delivery Time Estimation API"
    assert "endpoints" in data


def test_health_endpoint():
    """Test health endpoint returns system status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert "records" in data
    assert "r2_score" in data
    assert "mae" in data


def test_prediction_endpoint_valid_payload():
    """Test prediction with valid data matching API schema."""
    payload = {
        "product_weight_g": 1200.0,
        "product_vol_cm3": 4500.0,
        "distance_km": 800.0,
        "customer_lat": -23.55,
        "customer_lng": -46.63,
        "seller_lat": -23.95,
        "seller_lng": -46.33,
        "payment_lag_days": 2.0,
        "is_weekend_order": False,
        "freight_value": 29.9,
        "purchase_month": 11
    }
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    assert "predicted_days" in data
    assert isinstance(data["predicted_days"], float)
    assert data["predicted_days"] > 0


def test_prediction_endpoint_invalid_payload():
    """Test prediction with missing required fields."""
    payload = {
        "product_weight_g": 1200.0,
        # Missing other required fields
    }
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Unprocessable Entity


def test_prediction_endpoint_invalid_values():
    """Test prediction with out-of-range values."""
    payload = {
        "product_weight_g": -100.0,  # Negative weight (invalid)
        "product_vol_cm3": 4500.0,
        "distance_km": 800.0,
        "customer_lat": -23.55,
        "customer_lng": -46.63,
        "seller_lat": -23.95,
        "seller_lng": -46.33,
        "payment_lag_days": 2.0,
        "is_weekend_order": False,
        "freight_value": 29.9,
        "purchase_month": 11
    }
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_prediction_endpoint_invalid_coordinates():
    """Test prediction with invalid latitude/longitude."""
    payload = {
        "product_weight_g": 1200.0,
        "product_vol_cm3": 4500.0,
        "distance_km": 800.0,
        "customer_lat": 95.0,  # Invalid latitude (> 90)
        "customer_lng": -46.63,
        "seller_lat": -23.95,
        "seller_lng": -46.33,
        "payment_lag_days": 2.0,
        "is_weekend_order": False,
        "freight_value": 29.9,
        "purchase_month": 11
    }
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
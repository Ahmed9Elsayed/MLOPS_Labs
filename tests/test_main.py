"""
Tests for the Churn Prediction API.
Run with:
    uv run pytest tests/ -v
    uv run pytest tests/ -v --cov=app --cov=main --cov-report=term-missing
"""
import pytest
from litestar.testing import TestClient
from app.model_utils import predict_churn
from main import app
# ---------------------------------------------------------------------------
# Shared valid payload
# ---------------------------------------------------------------------------
VALID_PAYLOAD = {
    "credit_score": 619.0,
    "geography": "France",
    "gender": "Female",
    "age": 42.0,
    "tenure": 2.0,
    "balance": 0.0,
    "num_of_products": 1,
    "has_cr_card": 1,
    "is_active_member": 1,
    "estimated_salary": 101_348.88,
}
# ---------------------------------------------------------------------------
# Function Tests  (  1)
# ---------------------------------------------------------------------------
def test_predict_churn_returns_valid_prediction():
    """ predict_churn returns 0 or 1 with valid input."""
    result = predict_churn(
        credit_score=619.0,
        geography="France",
        gender="Female",
        age=42.0,
        tenure=2.0,
        balance=0.0,
        num_of_products=1,
        has_cr_card=1,
        is_active_member=1,
        estimated_salary=101_348.88,
    )
    assert result in (0, 1), "Prediction must be 0 or 1"
    assert isinstance(result, int), "Prediction must be an integer"
def test_predict_churn_with_high_balance_customer():
    """– edge case: high-balance, engaged customer."""
    result = predict_churn(
        credit_score=850.0,
        geography="Germany",
        gender="Male",
        age=35.0,
        tenure=8.0,
        balance=150_000.0,
        num_of_products=4,
        has_cr_card=1,
        is_active_member=1,
        estimated_salary=200_000.0,
    )
    assert result in (0, 1), "Prediction must be 0 or 1"
def test_predict_churn_with_low_credit_score():
    """(bonus) – edge case: low credit-score customer."""
    result = predict_churn(
        credit_score=300.0,
        geography="Spain",
        gender="Female",
        age=65.0,
        tenure=1.0,
        balance=0.0,
        num_of_products=1,
        has_cr_card=0,
        is_active_member=0,
        estimated_salary=50_000.0,
    )
    assert result in (0, 1), "Prediction must be 0 or 1"
def test_predict_churn_with_all_geographies():
    """(bonus) – verify all three geographies produce valid predictions."""
    for geo in ("France", "Spain", "Germany"):
        result = predict_churn(
            credit_score=600.0,
            geography=geo,
            gender="Male",
            age=40.0,
            tenure=5.0,
            balance=50_000.0,
            num_of_products=2,
            has_cr_card=1,
            is_active_member=1,
            estimated_salary=100_000.0,
        )
        assert result in (0, 1), f"Unexpected prediction for geography={geo}"
def test_predict_churn_with_extreme_ages():
    """(bonus) – edge case: youngest and oldest plausible customers."""
    for age in (18.0, 92.0):
        result = predict_churn(
            credit_score=600.0,
            geography="France",
            gender="Female",
            age=age,
            tenure=0.0,
            balance=10_000.0,
            num_of_products=1,
            has_cr_card=1,
            is_active_member=1,
            estimated_salary=50_000.0,
        )
        assert result in (0, 1), f"Unexpected prediction for age={age}"
# ---------------------------------------------------------------------------
# Endpoint Tests  ( 3, 4, 5)
# ---------------------------------------------------------------------------
def test_get_home_endpoint():
    """  5 – GET / returns 200 with expected keys."""
    with TestClient(app=app) as client:
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "endpoints" in data
        assert data["message"] == "Welcome to Churn Prediction API"
def test_get_health_endpoint():
    """  4 – GET /health returns 200 and {"status": "healthy"}."""
    with TestClient(app=app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
def test_post_predict_endpoint_valid_input():
    """  3 – POST /predict with valid payload returns 201 and a prediction."""
    with TestClient(app=app) as client:
        response = client.post("/predict", json=VALID_PAYLOAD)
        assert response.status_code == 201, (
            f"Expected 201 but got {response.status_code}: {response.text}"
        )
        data = response.json()
        assert "prediction" in data
        assert data["prediction"] in (0, 1)
        assert "label" in data
        assert data["label"] in ("churn", "no_churn")
def test_post_predict_endpoint_with_germany_customer():
    """  3 – POST /predict works for a German male customer."""
    with TestClient(app=app) as client:
        payload = {
            "credit_score": 750.0,
            "geography": "Germany",
            "gender": "Male",
            "age": 30.0,
            "tenure": 6.0,
            "balance": 100_000.0,
            "num_of_products": 3,
            "has_cr_card": 1,
            "is_active_member": 1,
            "estimated_salary": 150_000.0,
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 201
        data = response.json()
        assert "prediction" in data
        assert data["prediction"] in (0, 1)
def test_post_predict_with_edge_case_minimum_values():
    """POST /predict – edge case: minimum credit score and new customer."""
    with TestClient(app=app) as client:
        payload = {
            "credit_score": 300.0,
            "geography": "Spain",
            "gender": "Female",
            "age": 18.0,
            "tenure": 0.0,
            "balance": 0.0,
            "num_of_products": 1,
            "has_cr_card": 0,
            "is_active_member": 0,
            "estimated_salary": 20_000.0,
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 201
        data = response.json()
        assert data["prediction"] in (0, 1)
def test_post_predict_with_high_values():
    """POST /predict – edge case: maximum-value customer profile."""
    with TestClient(app=app) as client:
        payload = {
            "credit_score": 850.0,
            "geography": "France",
            "gender": "Male",
            "age": 75.0,
            "tenure": 10.0,
            "balance": 500_000.0,
            "num_of_products": 4,
            "has_cr_card": 1,
            "is_active_member": 1,
            "estimated_salary": 300_000.0,
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 201
        data = response.json()
        assert data["prediction"] in (0, 1)
# ---------------------------------------------------------------------------
# Bonus – Invalid input returns 400  (  6)
# ---------------------------------------------------------------------------
def test_post_predict_missing_required_field_returns_400():
    """  6 (bonus) – missing required field should return HTTP 400."""
    with TestClient(app=app) as client:
        # Send only a subset of required fields
        payload = {
            "credit_score": 619.0,
            "geography": "France",
            "gender": "Female",
            "age": 42.0,
            # tenure, balance, num_of_products, has_cr_card,
            # is_active_member, estimated_salary are all missing
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 400, (
            f"Expected 400 but got {response.status_code}: {response.text}"
        )
def test_post_predict_wrong_type_returns_400():
    """  6 (bonus) – sending a string for a numeric field returns HTTP 400."""
    with TestClient(app=app) as client:
        payload = {**VALID_PAYLOAD, "credit_score": "not_a_number"}
        response = client.post("/predict", json=payload)
        assert response.status_code == 400, (
            f"Expected 400 but got {response.status_code}: {response.text}"
        )

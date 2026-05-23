"""
Churn Prediction API
====================
A Litestar-powered REST API that serves the trained bank-customer churn
prediction model with full Axiom observability.

Endpoints
---------
GET  /           – Welcome message and available endpoints
GET  /health     – Health-check (returns {"status": "healthy"})
POST /predict    – Churn prediction for a single customer

Axiom Monitoring
----------------
Every request is logged to Axiom with:
  - Model metrics  : prediction class, churn probability
  - Data metrics   : input features, data-quality flags
  - Server metrics : latency (ms), status code, endpoint

Run with:
    uv run litestar --app main:app run --reload
Swagger UI:
    http://localhost:8000/schema/swagger
"""

from typing import Annotated

from dotenv import load_dotenv
from litestar import Litestar, get, post
from litestar.openapi import OpenAPIConfig
from litestar.params import Body
from msgspec import Struct

from app.axiom_logger import Timer, log_error_event, log_prediction_event, log_request_event
from app.logger_setup import setup_logging
from app.model_utils import predict_churn

# Load .env file (AXIOM_TOKEN, AXIOM_DATASET)
load_dotenv()

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logger = setup_logging()

# ---------------------------------------------------------------------------
# Request / Response Structs
# ---------------------------------------------------------------------------


class ChurnRequest(Struct):
    """Request body for the POST /predict endpoint.
    All fields map directly to the features expected by the trained
    RandomForestClassifier model.
    """

    credit_score: float
    """Customer credit score (300–850)."""
    geography: str
    """Country of residence: ``France``, ``Spain``, or ``Germany``."""
    gender: str
    """Gender: ``Male`` or ``Female``."""
    age: float
    """Age in years."""
    tenure: float
    """Years as a bank customer."""
    balance: float
    """Current account balance."""
    num_of_products: int
    """Number of bank products the customer uses."""
    has_cr_card: int
    """Has a credit card? 1 = yes, 0 = no."""
    is_active_member: int
    """Is an active member? 1 = yes, 0 = no."""
    estimated_salary: float
    """Estimated annual salary."""


# ---------------------------------------------------------------------------
# GET /  – Home endpoint
# ---------------------------------------------------------------------------


@get("/", tags=["General"])
async def home() -> dict:
    """Welcome endpoint.
    Returns a welcome message and a list of available API endpoints.
    """
    with Timer() as t:
        logger.info("Home endpoint accessed.")
        response = {
            "message": "Welcome to Churn Prediction API",
            "description": (
                "A machine-learning API that predicts whether a bank customer "
                "is likely to churn based on their profile."
            ),
            "endpoints": {
                "GET  /": "This welcome message",
                "GET  /health": "Health check",
                "POST /predict": "Churn prediction for a single customer",
                "GET  /schema/swagger": "Interactive Swagger UI",
            },
        }

    log_request_event(endpoint="/", method="GET", status_code=200, latency_ms=t.elapsed_ms)
    return response


# ---------------------------------------------------------------------------
# GET /health – Health-check endpoint
# ---------------------------------------------------------------------------


@get("/health", tags=["General"])
async def health() -> dict:
    """Health-check endpoint.
    Returns ``{"status": "healthy"}`` so that load-balancers and monitoring
    tools can verify that the service is up.
    """
    with Timer() as t:
        logger.info("Health check endpoint accessed – service is healthy.")
        response = {"status": "healthy"}

    log_request_event(
        endpoint="/health", method="GET", status_code=200, latency_ms=t.elapsed_ms
    )
    return response


# ---------------------------------------------------------------------------
# POST /predict – Prediction endpoint
# ---------------------------------------------------------------------------


@post("/predict", tags=["Prediction"])
async def predict(data: Annotated[ChurnRequest, Body(title="Churn Request")]) -> dict:
    """Predict whether a customer will churn.

    Accepts a JSON body with all required customer features, runs the
    trained RandomForestClassifier model, and returns:
    - **prediction**: 0 (no churn) or 1 (churn)
    - **label**: human-readable label
    - **churn_probability**: model confidence score (0.0 – 1.0)

    All request details are logged to Axiom for observability.
    """
    with Timer() as t:
        logger.info(
            "POST /predict | geography=%s gender=%s age=%.1f credit_score=%.1f",
            data.geography,
            data.gender,
            data.age,
            data.credit_score,
        )

        prediction, churn_probability = predict_churn(
            credit_score=data.credit_score,
            geography=data.geography,
            gender=data.gender,
            age=data.age,
            tenure=data.tenure,
            balance=data.balance,
            num_of_products=data.num_of_products,
            has_cr_card=data.has_cr_card,
            is_active_member=data.is_active_member,
            estimated_salary=data.estimated_salary,
        )

        churn_label = "churn" if prediction == 1 else "no_churn"

        logger.info(
            "POST /predict → prediction=%d (%s) confidence=%.1f%%",
            prediction,
            churn_label,
            churn_probability * 100,
        )

    # ── Send structured event to Axiom ──────────────────────────────────────
    log_prediction_event(
        credit_score=data.credit_score,
        geography=data.geography,
        gender=data.gender,
        age=data.age,
        tenure=data.tenure,
        balance=data.balance,
        num_of_products=data.num_of_products,
        has_cr_card=data.has_cr_card,
        is_active_member=data.is_active_member,
        estimated_salary=data.estimated_salary,
        prediction=prediction,
        churn_probability=churn_probability,
        latency_ms=t.elapsed_ms,
        status_code=201,
    )

    return {
        "prediction": prediction,
        "label": churn_label,
        "churn_probability": round(churn_probability, 4),
    }


# ---------------------------------------------------------------------------
# Litestar app
# ---------------------------------------------------------------------------

app = Litestar(
    route_handlers=[home, health, predict],
    openapi_config=OpenAPIConfig(
        title="Churn Prediction API",
        version="1.0.0",
        description=(
            "A Litestar-powered REST API that predicts bank-customer churn "
            "using a trained RandomForestClassifier model. "
            "All requests are monitored via Axiom."
        ),
    ),
)
"""
Churn Prediction API
====================
A Litestar-powered REST API that serves the trained bank-customer churn
prediction model.
Endpoints
---------
GET  /           – Welcome message and available endpoints
GET  /health     – Health-check (returns {"status": "healthy"})
POST /predict    – Churn prediction for a single customer
Run with:
    uv run litestar --app main:app run --reload
Swagger UI:
    http://localhost:8000/schema/swagger
"""
from typing import Annotated
from litestar import Litestar, get, post
from litestar.openapi import OpenAPIConfig
from litestar.params import Body
from msgspec import Struct
from app.logger_setup import setup_logging
from app.model_utils import predict_churn
# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logger = setup_logging()
# ---------------------------------------------------------------------------
#   1: Define ChurnRequest fields
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
#   2: GET /  – Home endpoint
# ---------------------------------------------------------------------------
@get("/", tags=["General"])
async def home() -> dict:
    """Welcome endpoint.
    Returns a welcome message and a list of available API endpoints.
    """
    logger.info("Home endpoint accessed.")
    return {
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
# ---------------------------------------------------------------------------
#   3: GET /health – Health-check endpoint
# ---------------------------------------------------------------------------
@get("/health", tags=["General"])
async def health() -> dict:
    """Health-check endpoint.
    Returns ``{"status": "healthy"}`` so that load-balancers and monitoring
    tools can verify that the service is up.
    """
    logger.info("Health check endpoint accessed – service is healthy.")
    return {"status": "healthy"}
# ---------------------------------------------------------------------------
#   4: POST /predict – Prediction endpoint
# ---------------------------------------------------------------------------
@post("/predict", tags=["Prediction"])
async def predict(data: Annotated[ChurnRequest, Body(title="Churn Request")]) -> dict:
    """Predict whether a customer will churn.
    Accepts a JSON body with all required customer features, runs the
    trained RandomForestClassifier model, and returns the binary prediction.
    - **0** → customer is *not* predicted to churn
    - **1** → customer *is* predicted to churn
    """
    logger.info(
        "POST /predict called | geography=%s gender=%s age=%.1f "
        "credit_score=%.1f tenure=%.1f balance=%.2f num_of_products=%d "
        "has_cr_card=%d is_active_member=%d estimated_salary=%.2f",
        data.geography,
        data.gender,
        data.age,
        data.credit_score,
        data.tenure,
        data.balance,
        data.num_of_products,
        data.has_cr_card,
        data.is_active_member,
        data.estimated_salary,
    )
    prediction = predict_churn(
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
        "POST /predict response | prediction=%d label=%s geography=%s age=%.1f",
        prediction,
        churn_label,
        data.geography,
        data.age,
    )
    return {
        "prediction": prediction,
        "label": churn_label,
    }
# ---------------------------------------------------------------------------
#   5: Register handlers in Litestar(route_handlers=[...])
# ---------------------------------------------------------------------------
app = Litestar(
    route_handlers=[home, health, predict],
    openapi_config=OpenAPIConfig(
        title="Churn Prediction API",
        version="1.0.0",
        description=(
            "A Litestar-powered REST API that predicts bank-customer churn "
            "using a trained RandomForestClassifier model."
        ),
    ),
)
"""
Axiom monitoring module for the Churn Prediction API.

Sends structured JSON events to Axiom via direct HTTP (no SDK needed).
Uses httpx which is already installed as part of litestar[standard].

Axiom Ingest API:
    POST https://api.axiom.co/v1/datasets/{dataset}/ingest
    Authorization: Bearer {AXIOM_TOKEN}
    Content-Type: application/json

Requires env vars (set in .env):
    AXIOM_TOKEN   – your Axiom API ingest token (xaat-...)
    AXIOM_DATASET – your Axiom dataset name (e.g. churn-api-logs)
"""

import logging
import os
import time
from datetime import datetime, timezone
from typing import Any

import httpx

logger = logging.getLogger("churn_prediction_api.axiom_logger")

AXIOM_INGEST_URL = "https://api.axiom.co/v1/datasets/{dataset}/ingest"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_credentials() -> tuple[str, str] | tuple[None, None]:
    """Return (token, dataset) from environment, or (None, None) if unset."""
    token = os.getenv("AXIOM_TOKEN", "").strip()
    dataset = os.getenv("AXIOM_DATASET", "churn-api-logs").strip()
    if not token or token.startswith("xaat-xxx"):
        return None, None
    return token, dataset


def _send(events: list[dict[str, Any]]) -> None:
    """
    POST one or more events to the Axiom ingest API.
    Failures are logged as warnings – never raise so the API keeps serving.
    """
    token, dataset = _get_credentials()
    if token is None:
        logger.debug("Axiom not configured – skipping event send.")
        return

    now = datetime.now(timezone.utc).isoformat()
    for event in events:
        event.setdefault("_time", now)

    url = AXIOM_INGEST_URL.format(dataset=dataset)
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    try:
        response = httpx.post(url, json=events, headers=headers, timeout=5.0)
        if response.status_code not in (200, 204):
            logger.warning(
                "Axiom ingest returned %d: %s", response.status_code, response.text[:200]
            )
        else:
            logger.debug("Axiom: %d event(s) ingested → dataset=%s", len(events), dataset)
    except httpx.TimeoutException:
        logger.warning("Axiom ingest timed out – event dropped.")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Axiom ingest error: %s", exc)


# ---------------------------------------------------------------------------
# Public event helpers – called from main.py
# ---------------------------------------------------------------------------


def log_prediction_event(
    *,
    # Input features
    credit_score: float,
    geography: str,
    gender: str,
    age: float,
    tenure: float,
    balance: float,
    num_of_products: int,
    has_cr_card: int,
    is_active_member: int,
    estimated_salary: float,
    # Model output
    prediction: int,
    churn_probability: float,
    # Server info
    latency_ms: float,
    status_code: int = 201,
) -> None:
    """
    Log a /predict request to Axiom with all 3 metric categories.

    ┌─────────────────────────────────────────────────────┐
    │ Model-related metrics                               │
    │   prediction          – 0 (no churn) or 1 (churn)  │
    │   churn_probability   – confidence score 0.0-1.0   │
    │   predicted_label     – "churn" or "no_churn"      │
    ├─────────────────────────────────────────────────────┤
    │ Data-related metrics                                │
    │   All 10 input features                             │
    │   is_credit_score_valid  (300–850 range)            │
    │   is_age_valid           (18–100 range)             │
    │   is_geography_valid     (France/Spain/Germany)     │
    │   has_data_quality_issue – any flag above is False  │
    ├─────────────────────────────────────────────────────┤
    │ Server-related metrics                              │
    │   latency_ms   – end-to-end request duration        │
    │   status_code  – HTTP response code                 │
    │   endpoint     – "/predict"                         │
    └─────────────────────────────────────────────────────┘
    """
    # ── Data quality flags ────────────────────────────────────────────────
    is_credit_score_valid = 300 <= credit_score <= 850
    is_age_valid = 18 <= age <= 100
    is_geography_valid = geography in {"France", "Spain", "Germany"}

    event = {
        # Model-related
        "prediction": prediction,
        "predicted_label": "churn" if prediction == 1 else "no_churn",
        "churn_probability": round(churn_probability, 4),
        "no_churn_probability": round(1.0 - churn_probability, 4),
        # Data-related – input features
        "credit_score": credit_score,
        "geography": geography,
        "gender": gender,
        "age": age,
        "tenure": tenure,
        "balance": balance,
        "num_of_products": num_of_products,
        "has_cr_card": has_cr_card,
        "is_active_member": is_active_member,
        "estimated_salary": estimated_salary,
        # Data quality
        "is_credit_score_valid": is_credit_score_valid,
        "is_age_valid": is_age_valid,
        "is_geography_valid": is_geography_valid,
        "has_data_quality_issue": not all(
            [is_credit_score_valid, is_age_valid, is_geography_valid]
        ),
        # Server-related
        "endpoint": "/predict",
        "method": "POST",
        "status_code": status_code,
        "latency_ms": round(latency_ms, 2),
    }

    _send([event])
    logger.debug("Axiom event queued | prediction=%d latency=%.2fms", prediction, latency_ms)


def log_request_event(
    *,
    endpoint: str,
    method: str,
    status_code: int,
    latency_ms: float,
) -> None:
    """Log a lightweight event for non-prediction requests (/, /health)."""
    _send([
        {
            "endpoint": endpoint,
            "method": method,
            "status_code": status_code,
            "latency_ms": round(latency_ms, 2),
        }
    ])


def log_error_event(
    *,
    endpoint: str,
    method: str,
    status_code: int,
    error_message: str,
    latency_ms: float,
) -> None:
    """Log a 4xx / 5xx error event to Axiom."""
    _send([
        {
            "endpoint": endpoint,
            "method": method,
            "status_code": status_code,
            "error_message": error_message[:500],
            "latency_ms": round(latency_ms, 2),
            "is_error": True,
        }
    ])
    logger.warning("Error logged to Axiom | %s %s → %d", method, endpoint, status_code)


# ---------------------------------------------------------------------------
# Timer utility
# ---------------------------------------------------------------------------


class Timer:
    """Context manager: measures elapsed time in milliseconds."""

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000

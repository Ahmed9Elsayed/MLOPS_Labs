"""
Locust load test for the Churn Prediction API.

Targets:
    GET  /           - Home endpoint
    GET  /health     - Health check endpoint
    POST /predict    - Churn prediction endpoint (main load target)

Run (against your deployed EC2):
    locust -f locustfile.py --host http://<EC2-IP>:443

Run (headless, no UI):
    locust -f locustfile.py --host http://<EC2-IP>:443 \
           --users 50 --spawn-rate 5 --run-time 60s --headless

Run (locally):
    locust -f locustfile.py --host http://localhost:8000
"""

import random

from locust import HttpUser, between, task


# ---------------------------------------------------------------------------
# Sample customer profiles – drawn from the Churn_Modelling.csv dataset
# These cover a range of customer archetypes to simulate realistic traffic
# ---------------------------------------------------------------------------
SAMPLE_CUSTOMERS = [
    # Row 1 – France, Female, 42  (Exited=1 – churner)
    {
        "credit_score": 619.0,
        "geography": "France",
        "gender": "Female",
        "age": 42.0,
        "tenure": 2.0,
        "balance": 0.0,
        "num_of_products": 1,
        "has_cr_card": 1,
        "is_active_member": 1,
        "estimated_salary": 101348.88,
    },
    # Row 2 – Spain, Female, 41  (Exited=0 – retained)
    {
        "credit_score": 608.0,
        "geography": "Spain",
        "gender": "Female",
        "age": 41.0,
        "tenure": 1.0,
        "balance": 83807.86,
        "num_of_products": 1,
        "has_cr_card": 0,
        "is_active_member": 1,
        "estimated_salary": 112542.58,
    },
    # High-value customer – Germany, Male, 35
    {
        "credit_score": 850.0,
        "geography": "Germany",
        "gender": "Male",
        "age": 35.0,
        "tenure": 8.0,
        "balance": 150000.0,
        "num_of_products": 2,
        "has_cr_card": 1,
        "is_active_member": 1,
        "estimated_salary": 200000.0,
    },
    # At-risk customer – low credit, inactive
    {
        "credit_score": 350.0,
        "geography": "France",
        "gender": "Female",
        "age": 58.0,
        "tenure": 1.0,
        "balance": 0.0,
        "num_of_products": 1,
        "has_cr_card": 0,
        "is_active_member": 0,
        "estimated_salary": 35000.0,
    },
    # Young new customer – Spain, Male, 24
    {
        "credit_score": 720.0,
        "geography": "Spain",
        "gender": "Male",
        "age": 24.0,
        "tenure": 0.0,
        "balance": 0.0,
        "num_of_products": 1,
        "has_cr_card": 1,
        "is_active_member": 1,
        "estimated_salary": 55000.0,
    },
    # Long-tenure Germany customer
    {
        "credit_score": 680.0,
        "geography": "Germany",
        "gender": "Female",
        "age": 45.0,
        "tenure": 10.0,
        "balance": 125000.0,
        "num_of_products": 2,
        "has_cr_card": 1,
        "is_active_member": 0,
        "estimated_salary": 88000.0,
    },
]


# ---------------------------------------------------------------------------
# User behaviour
# ---------------------------------------------------------------------------

class ChurnAPIUser(HttpUser):
    """
    Simulates a typical API consumer that:
      - Occasionally checks the health endpoint (monitoring / k8s probe)
      - Occasionally visits the home page
      - Frequently calls /predict  (the hot path – 70 % of requests)

    wait_time: each simulated user waits 1–3 seconds between tasks,
    mimicking human-paced or lightly-throttled service calls.
    """

    wait_time = between(1, 3)

    # ------------------------------------------------------------------
    # Tasks
    # weight=7 → predict is called ~7x more often than health/home
    # ------------------------------------------------------------------

    @task(7)
    def predict_churn(self):
        """POST /predict – the primary load target."""
        payload = random.choice(SAMPLE_CUSTOMERS)
        with self.client.post(
            "/predict",
            json=payload,
            name="/predict",
            catch_response=True,
        ) as response:
            if response.status_code == 201:
                data = response.json()
                if "prediction" not in data:
                    response.failure("Response missing 'prediction' key")
                else:
                    response.success()
            else:
                response.failure(
                    f"Unexpected status {response.status_code}: {response.text[:200]}"
                )

    @task(2)
    def health_check(self):
        """GET /health – simulates monitoring probes."""
        with self.client.get(
            "/health",
            name="/health",
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                if response.json().get("status") == "healthy":
                    response.success()
                else:
                    response.failure("Health check returned non-healthy status")
            else:
                response.failure(f"Health check failed: {response.status_code}")

    @task(1)
    def home(self):
        """GET / – simulates a client discovering the API."""
        with self.client.get(
            "/",
            name="/",
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Home returned {response.status_code}")


# ---------------------------------------------------------------------------
# Heavy load user – no wait, for stress testing
# ---------------------------------------------------------------------------

class StressUser(HttpUser):
    """
    Hammers /predict with no wait time.
    Use this to find the breaking point of your API.
    Only activate by setting --tags stress in the Locust CLI.
    """

    wait_time = between(0.1, 0.5)

    @task
    def predict_churn_stress(self):
        payload = random.choice(SAMPLE_CUSTOMERS)
        self.client.post("/predict", json=payload, name="/predict [stress]")

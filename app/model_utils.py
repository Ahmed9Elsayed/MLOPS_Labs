"""Model loading and prediction logic.
The model must be loaded ONCE at module level, NOT inside the predict function.
  1: Load your serialized churn model from data/model.pkl
  2: Implement predict_churn()
  3: Fill in sample features (see __main__ block at the bottom)
"""
import logging
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
logger = logging.getLogger("churn_prediction_api.model_utils")
# ---------------------------------------------------------------------------
#   1: Load model at module level (once, not inside the predict function)
# ---------------------------------------------------------------------------
MODEL_PATH = "data/model.pkl"
logger.info("Loading churn model from '%s' …", MODEL_PATH)
model = joblib.load(MODEL_PATH)
logger.info(
    "Model loaded successfully. Type: %s",
    type(model).__name__,
)
# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------
def _create_preprocessor() -> ColumnTransformer:
    """
    Create the same preprocessing transformer used during model training.
    Returns:
        ColumnTransformer: Unfitted preprocessing pipeline.
    """
    cat_cols = ["Geography", "Gender"]
    num_cols = [
        "CreditScore",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "HasCrCard",
        "IsActiveMember",
        "EstimatedSalary",
    ]
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), cat_cols),
        ],
        remainder="passthrough",
    )
    return preprocessor
# Global preprocessor – lazily fitted on first call.
_preprocessor: ColumnTransformer | None = None
def _fit_preprocessor() -> None:
    """
    Fit the preprocessor on a representative reference dataset that covers
    all category values seen during training.
    """
    global _preprocessor
    if _preprocessor is not None:
        return
    logger.info("Fitting preprocessor on reference data …")
    _preprocessor = _create_preprocessor()
    # Representative rows that include all Geography and Gender categories.
    dummy_data = pd.DataFrame(
        {
            "CreditScore": [300, 500, 800],
            "Geography": ["France", "Spain", "Germany"],
            "Gender": ["Male", "Female", "Male"],
            "Age": [25, 50, 75],
            "Tenure": [0, 5, 10],
            "Balance": [0, 125_000, 250_000],
            "NumOfProducts": [1, 2, 4],
            "HasCrCard": [0, 1, 1],
            "IsActiveMember": [0, 1, 1],
            "EstimatedSalary": [0, 100_000, 200_000],
        }
    )
    _preprocessor.fit(dummy_data)
    logger.info("Preprocessor fitted successfully.")
# ---------------------------------------------------------------------------
#   2: Implement predict_churn()
# ---------------------------------------------------------------------------
def predict_churn(
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
) -> int:
    """
    Take individual feature values and return a churn prediction (0 or 1).
    Args:
        credit_score: Customer's credit score (300–850).
        geography: Country of residence – one of ``France``, ``Spain``, ``Germany``.
        gender: Gender – one of ``Male``, ``Female``.
        age: Customer's age in years.
        tenure: Number of years the customer has been with the bank.
        balance: Account balance in the local currency.
        num_of_products: Number of bank products the customer uses.
        has_cr_card: Whether the customer has a credit card (1 = yes, 0 = no).
        is_active_member: Whether the customer is an active member (1 = yes, 0 = no).
        estimated_salary: Customer's estimated annual salary.
    Returns:
        int: ``0`` – customer is **not** predicted to churn;
             ``1`` – customer **is** predicted to churn.
    """
    global _preprocessor
    # Ensure the preprocessor is ready.
    if _preprocessor is None:
        _fit_preprocessor()
    logger.debug(
        "predict_churn called | credit_score=%.1f geography=%s gender=%s age=%.1f "
        "tenure=%.1f balance=%.2f num_of_products=%d has_cr_card=%d "
        "is_active_member=%d estimated_salary=%.2f",
        credit_score,
        geography,
        gender,
        age,
        tenure,
        balance,
        num_of_products,
        has_cr_card,
        is_active_member,
        estimated_salary,
    )
    # Build input DataFrame.
    input_data = pd.DataFrame(
        {
            "CreditScore": [credit_score],
            "Geography": [geography],
            "Gender": [gender],
            "Age": [age],
            "Tenure": [tenure],
            "Balance": [balance],
            "NumOfProducts": [num_of_products],
            "HasCrCard": [has_cr_card],
            "IsActiveMember": [is_active_member],
            "EstimatedSalary": [estimated_salary],
        }
    )
    # Preprocess – returns a numpy array.
    preprocessed_arr = _preprocessor.transform(input_data)
    # The saved model was serialised from a numpy-array training run, so we
    # must NOT wrap the result in a named DataFrame (different column prefixes
    # would cause a ValueError).  Pass the raw array directly.
    prediction: int = int(model.predict(preprocessed_arr)[0])
    logger.info(
        "Prediction complete | result=%d | geography=%s gender=%s age=%.1f",
        prediction,
        geography,
        gender,
        age,
    )
    return prediction
# ---------------------------------------------------------------------------
#   3: Sample features – run this module directly to smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    # Sample row 1 from the dataset (customer with Exited=1)
    print("Testing predict_churn function …")
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
    print(f"Prediction for sample customer: {result} ({'Churn' if result else 'No Churn'})")
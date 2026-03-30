# src/inference.py

import os
import joblib
import pandas as pd
import logging

from pipelines.feature_pipeline import feature_engineering

# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# ----------------------------
# Paths
# ----------------------------
MODEL_PATH = "models"


# ----------------------------
# Load latest PIPELINE (not just model)
# ----------------------------
def load_latest_pipeline():
    files = [f for f in os.listdir(MODEL_PATH) if f.startswith("pipeline_")]

    if not files:
        raise FileNotFoundError("No trained pipeline found")

    latest_file = sorted(files)[-1]
    pipeline_path = os.path.join(MODEL_PATH, latest_file)

    logging.info(f"Loading pipeline: {pipeline_path}")

    return joblib.load(pipeline_path)


# ----------------------------
# Validate RAW input
# ----------------------------
def validate_input(data: pd.DataFrame):
    required_cols = [
        "Age",
        "Income",
        "Credit_Score",
        "Loan_Amount",
        "Loan_Term",
        "Employment_Status"
    ]

    missing_cols = [col for col in required_cols if col not in data.columns]

    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")

    return data[required_cols]


# ----------------------------
# Predict function
# ----------------------------
def predict(data: pd.DataFrame):
    try:
        pipeline = load_latest_pipeline()

        # Step 1: validate RAW input
        data = validate_input(data)

        # Step 2: apply SAME feature engineering
        data = feature_engineering(data)

        # Step 3: pipeline handles preprocessing + model
        preds = pipeline.predict(data)

        logging.info(f"Predictions generated: {preds}")

        return preds

    except Exception as e:
        logging.exception("Prediction failed")
        raise e


# ----------------------------
# Example run
# ----------------------------
if __name__ == "__main__":
    sample = pd.DataFrame([{
        "Age": 30,
        "Income": 50000,
        "Credit_Score": 650,
        "Loan_Amount": 20000,
        "Loan_Term": 36,
        "Employment_Status": "Employed"
    }])

    result = predict(sample)
    print("Prediction:", result)
# deployment/app.py

from fastapi import FastAPI
import pandas as pd
import logging
from src.inference import predict  # Your existing inference pipeline

# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

app = FastAPI(title="Loan Approval API")

# ----------------------------
# API Endpoint
# ----------------------------
@app.post("/predict")
def predict_loan(data: dict):
    """
    Expects JSON payload with keys:
    Age, Income, Credit_Score, Loan_Amount, Loan_Term, Employment_Status

    Example JSON:
    {
        "Age": 30,
        "Income": 50000,
        "Credit_Score": 650,
        "Loan_Amount": 20000,
        "Loan_Term": 36,
        "Employment_Status": "Employed"
    }
    """
    try:
        df = pd.DataFrame([data])
        logging.info(f"Received input: {df.to_dict(orient='records')}")

        # Use your trained model + feature pipeline
        pred = predict(df)[0]
        logging.info(f"Prediction: {pred}")

        return {"loan_approval": int(pred)}

    except Exception as e:
        logging.exception("Prediction failed")
        return {"error": str(e)}
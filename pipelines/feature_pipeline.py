import pandas as pd
import numpy as np
import logging

# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# ----------------------------
# SAFE QCUT (robust binning)
# ----------------------------
def safe_qcut(series, q, labels):
    """
    Safe version of qcut:
    - Handles small datasets (inference-safe)
    - Prevents bin/label mismatch
    """
    try:
        return pd.qcut(series, q=q, labels=labels, duplicates="drop")
    except Exception:
        bins = min(len(series.unique()), q)

        if bins < 2:
            return pd.Series([labels[0]] * len(series), index=series.index)

        return pd.cut(series, bins=bins, labels=labels[:bins])


# ----------------------------
# FEATURE ENGINEERING
# ----------------------------
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("Starting feature engineering...")

    if df is None or df.empty:
        raise ValueError("Input DataFrame is empty or None")

    df = df.copy()

    # ----------------------------
    # BASIC CLEANING (safe)
    # ----------------------------
    if "Employment_Status" in df.columns:
        df["Employment_Status"] = df["Employment_Status"].fillna("Unknown")

    # Fill ONLY numeric columns (safe fix)
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)

    # ----------------------------
    # FEATURE CREATION
    # ----------------------------
    # Safe EMI
    if "Loan_Amount" in df.columns and "Loan_Term" in df.columns:
        df["EMI"] = np.where(
            df["Loan_Term"] == 0,
            0,
            df["Loan_Amount"] / df["Loan_Term"]
        )

    # Ratios (safe divisions)
    if "Income" in df.columns and "Loan_Amount" in df.columns:
        df["Income_Loan_Ratio"] = np.where(
            df["Loan_Amount"] == 0,
            0,
            df["Income"] / df["Loan_Amount"]
        )

        df["Loan_Burden"] = np.where(
            df["Income"] == 0,
            0,
            df["Loan_Amount"] / df["Income"]
        )

    if "Income" in df.columns and "EMI" in df.columns:
        df["Income_per_EMI"] = np.where(
            df["EMI"] == 0,
            0,
            df["Income"] / df["EMI"]
        )

    # ----------------------------
    # CATEGORICAL FEATURES (robust)
    # ----------------------------
    if "Credit_Score" in df.columns:
        df["Credit_Category"] = safe_qcut(
            df["Credit_Score"],
            q=3,
            labels=["Poor", "Average", "Good"]
        )

    if "Income" in df.columns:
        df["Income_Group"] = safe_qcut(
            df["Income"],
            q=3,
            labels=["Low", "Medium", "High"]
        )

    if "Age" in df.columns:
        df["Age_Group"] = safe_qcut(
            df["Age"],
            q=3,
            labels=["Young", "Middle", "Senior"]
        )

    # ----------------------------
    # FINAL LOG
    # ----------------------------
    logging.info(
        "Feature engineering completed. Added features: "
        "['EMI', 'Income_Loan_Ratio', 'Loan_Burden', "
        "'Income_per_EMI', 'Credit_Category', "
        "'Income_Group', 'Age_Group']"
    )

    logging.info(f"New DataFrame shape: {df.shape}")

    return df


# ----------------------------
# SAVE FEATURES (optional utility)
# ----------------------------
def save_features(df: pd.DataFrame, path: str):
    import joblib
    joblib.dump(df, path)
    logging.info(f"Features saved at {path}")
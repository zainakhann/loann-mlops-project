import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import yaml

# ----------------------------
# Load config
# ----------------------------
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)


def build_preprocessor(df):
    """
    Build preprocessing pipeline using config-defined features
    (fallback to auto-detection if not provided)
    """

    if df is None:
        raise ValueError("DataFrame cannot be None")

    target = config["data"]["target_column"]

    # ----------------------------
    # Prefer config features (stable)
    # ----------------------------
    numeric_features = config["data"].get("numeric_features")
    categorical_features = config["data"].get("categorical_features")

    # ----------------------------
    # Fallback to auto-detect (only if missing)
    # ----------------------------
    if not numeric_features or not categorical_features:
        numeric_features = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        categorical_features = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # ----------------------------
    # Remove target if present
    # ----------------------------
    if target in numeric_features:
        numeric_features.remove(target)

    if target in categorical_features:
        categorical_features.remove(target)

    print(f"Preprocessor created | Numeric: {numeric_features} | Categorical: {categorical_features}")

    # ----------------------------
    # Pipelines
    # ----------------------------
    numeric_pipeline = Pipeline([
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    # ----------------------------
    # Column Transformer
    # ----------------------------
    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features)
    ])

    return preprocessor
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
import os
import logging
from dotenv import load_dotenv
import yaml

# ----------------------------
# Load environment variables / secrets
# ----------------------------
load_dotenv(dotenv_path="secrets/.env")

# ----------------------------
# Ensure required directories exist
# ----------------------------
os.makedirs("logs", exist_ok=True)

# ----------------------------
# Configure logging
# ----------------------------
logging.basicConfig(
    filename='logs/loan_pipeline.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)

# ----------------------------
# Load config.yaml safely
# ----------------------------
try:
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    logging.info("Config loaded successfully")
except FileNotFoundError:
    logging.error("config/config.yaml not found")
    raise

# ----------------------------
# Config variables
# ----------------------------
DATA_PATH = config["data"]["raw_path"]
TARGET_COL = config["data"]["target_column"]
TEST_SIZE = config["data"]["test_size"]
RANDOM_STATE = config["project"]["random_state"]

# ----------------------------
# Load data temporarily to detect features
# ----------------------------
df_temp = pd.read_csv(DATA_PATH)

# Numeric features
FEATURES_NUMERIC = config["data"].get("numeric_features") or df_temp.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()
if TARGET_COL in FEATURES_NUMERIC:
    FEATURES_NUMERIC.remove(TARGET_COL)

# Categorical features
FEATURES_CATEGORICAL = config["data"].get("categorical_features") or df_temp.select_dtypes(
    include=["object", "category"]
).columns.tolist()

PIPELINE_PATH = config["paths"]["pipeline_path"]

logging.info(f"Numeric features: {FEATURES_NUMERIC}")
logging.info(f"Categorical features: {FEATURES_CATEGORICAL}")

# ----------------------------
# Data Utilities
# ----------------------------
def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """
    Load CSV data and handle missing values only.
    (No encoding here — handled in pipeline)
    """
    df = pd.read_csv(path)
    logging.info(f"Data loaded from {path}, shape: {df.shape}")

    # Handle numeric columns
    for col in FEATURES_NUMERIC:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Handle categorical columns
    for col in FEATURES_CATEGORICAL:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])

    # Ensure target is integer if exists
    if TARGET_COL in df.columns:
        df[TARGET_COL] = df[TARGET_COL].astype(int)

    logging.info("Missing values handled (no encoding applied)")
    return df


def split_data(
    df: pd.DataFrame,
    target: str = TARGET_COL,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE
):
    """
    Split DataFrame into train and test sets.
    """
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in DataFrame")

    X = df.drop(columns=[target])
    y = df[target]

    logging.info(f"Splitting data: test_size={test_size}, random_state={random_state}")

    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def save_pipeline(pipeline, path: str = PIPELINE_PATH):
    """
    Save trained pipeline/model to disk.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(pipeline, path)
    logging.info(f"Pipeline saved at {path}")


def load_pipeline(path: str = PIPELINE_PATH):
    """
    Load trained pipeline/model from disk.
    """
    if not os.path.exists(path):
        logging.error(f"Pipeline file not found at '{path}'")
        raise FileNotFoundError(f"Pipeline file not found at '{path}'")

    logging.info(f"Pipeline loaded from {path}")
    return joblib.load(path)
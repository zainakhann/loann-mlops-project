import logging
import yaml
import os

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from pipelines.preprocessing_pipeline import build_preprocessor

# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# ----------------------------
# Load config
# ----------------------------
CONFIG_PATH = "config/config.yaml"

if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError(f"{CONFIG_PATH} not found")

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)
    logging.info("Config loaded successfully")

# ----------------------------
# Model Config
# ----------------------------
MODEL_CONFIG = config.get("model", {})
BASE_PARAMS = MODEL_CONFIG.get("base_params", {})

RANDOM_STATE = config.get("project", {}).get("random_state", 42)

# ----------------------------
# Pipeline Builder (DF-based)
# ----------------------------
def build_model_pipeline(df):
    """
    Build an end-to-end ML pipeline using a DataFrame.
    Includes preprocessing (auto-detect columns) + RandomForest model.
    """

    if df is None or df.empty:
        raise ValueError("Input DataFrame cannot be None or empty")

    logging.info("Building model pipeline (DF-based)...")

    # ----------------------------
    # Preprocessing
    # ----------------------------
    preprocessor = build_preprocessor(df)

    # ----------------------------
    # RandomForest Model
    # ----------------------------
    model = RandomForestClassifier(
        n_estimators=BASE_PARAMS.get("n_estimators", 200),
        max_depth=BASE_PARAMS.get("max_depth", 10),
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced"
    )

    # ----------------------------
    # Combine into Pipeline
    # ----------------------------
    pipeline = Pipeline(steps=[
        ("preprocessing", preprocessor),
        ("classifier", model)
    ])

    logging.info("Pipeline built successfully ✅")

    return pipeline
import os
import logging
import yaml
import joblib
from datetime import datetime

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score

import mlflow
import mlflow.sklearn

from src.utils import load_data, split_data
from pipelines.feature_pipeline import feature_engineering
from pipelines.model_pipeline import build_model_pipeline

# ----------------------------
# Load Config
# ----------------------------
CONFIG_PATH = "config/config.yaml"

if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError("config/config.yaml not found")

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

# ----------------------------
# Config Values
# ----------------------------
TUNING_CONFIG = config["model"].get("tuning", {})
MODEL_PATH = config["paths"].get("models", "models/")
TARGET_COL = config["data"].get("target_column", "Loan_Approved")

SCORING = TUNING_CONFIG.get("scoring", "f1")
CV_FOLDS = TUNING_CONFIG.get("cv_folds", 3)
RAW_PARAM_GRID = TUNING_CONFIG.get("param_grid", {})

RANDOM_STATE = config["project"].get("random_state", 42)

# MLflow
MLFLOW_URI = config["mlflow"].get("tracking_uri")
EXPERIMENT_NAME = config["mlflow"].get("experiment_name")

# ----------------------------
# Directories
# ----------------------------
os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs("logs", exist_ok=True)

# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(
    filename="logs/hyperparameter_tuning.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger("").addHandler(console)

# ----------------------------
# Hyperparameter Tuning
# ----------------------------
def run_tuning():
    logging.info("===== HYPERPARAMETER TUNING STARTED =====")

    if not TUNING_CONFIG.get("enabled", False):
        logging.warning("Tuning is disabled in config.yaml")
        return None

    if not RAW_PARAM_GRID:
        raise ValueError("Param grid is empty in config.yaml")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    try:
        # ----------------------------
        # Load & Prepare Data
        # ----------------------------
        df = load_data(config["data"]["raw_path"])
        logging.info(f"Data loaded from {config['data']['raw_path']}, shape: {df.shape}")

        # Feature Engineering
        logging.info("Starting feature engineering...")
        df_fe = feature_engineering(df)
        logging.info(f"Feature engineering completed. Added features: {list(set(df_fe.columns) - set(df.columns))}")
        logging.info(f"New DataFrame shape: {df_fe.shape}")

        # Split Data
        X_train, X_test, y_train, y_test = split_data(df_fe, target=TARGET_COL)
        logging.info(f"Splitting data: test_size=0.2, random_state={RANDOM_STATE}")

        # ----------------------------
        # Build Pipeline (pass df_fe)
        # ----------------------------
        pipeline = build_model_pipeline(df_fe)

        # ----------------------------
        # Fix Param Grid for pipeline
        # ----------------------------
        param_grid = {f"classifier__{k}": v for k, v in RAW_PARAM_GRID.items()}

        # ----------------------------
        # Grid Search
        # ----------------------------
        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=CV_FOLDS,
            n_jobs=-1,
            verbose=1,
            scoring=SCORING
        )

        logging.info(f"Grid Search started | Scoring: {SCORING}")
        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_
        best_params = grid.best_params_
        best_cv_score = grid.best_score_

        logging.info(f"✅ Best Params: {best_params}")
        logging.info(f"✅ Best CV Score ({SCORING}): {best_cv_score:.4f}")

        # ----------------------------
        # Test Evaluation
        # ----------------------------
        y_pred = best_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        logging.info(f"📊 Test Accuracy: {acc:.4f}")
        logging.info(f"📊 Test F1 Score: {f1:.4f}")

        # ----------------------------
        # Save Model
        # ----------------------------
        model_path = os.path.join(MODEL_PATH, f"best_model_{timestamp}.pkl")
        joblib.dump(best_model, model_path)

        # ----------------------------
        # MLflow Tracking
        # ----------------------------
        if MLFLOW_URI:
            mlflow.set_tracking_uri(MLFLOW_URI)
            mlflow.set_experiment(EXPERIMENT_NAME)

            with mlflow.start_run(run_name=f"grid_search_{timestamp}"):
                mlflow.log_params(best_params)
                mlflow.log_metric("cv_score", best_cv_score)
                mlflow.log_metric("test_accuracy", acc)
                mlflow.log_metric("test_f1", f1)
                mlflow.sklearn.log_model(best_model, "model")

        # ----------------------------
        # Metadata
        # ----------------------------
        metadata = {
            "timestamp": timestamp,
            "best_params": best_params,
            "cv_score": best_cv_score,
            "scoring_metric": SCORING,
            "test_accuracy": acc,
            "test_f1_score": f1,
            "model_path": model_path
        }

        metadata_path = os.path.join(MODEL_PATH, f"tuning_metadata_{timestamp}.pkl")
        joblib.dump(metadata, metadata_path)

        logging.info(f"💾 Model saved at: {model_path}")
        logging.info(f"📦 Metadata saved at: {metadata_path}")
        logging.info("===== HYPERPARAMETER TUNING COMPLETED =====")

        return metadata

    except Exception as e:
        logging.exception("❌ Hyperparameter tuning failed")
        raise e

if __name__ == "__main__":
    run_tuning()
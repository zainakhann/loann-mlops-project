# src/train.py

import os
import logging
import joblib
from datetime import datetime
import subprocess
import hashlib

import yaml
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

import mlflow
import mlflow.sklearn

from src.utils import load_data, split_data
from pipelines.feature_pipeline import feature_engineering, save_features
from pipelines.model_pipeline import build_model_pipeline

# ----------------------------
# Load config
# ----------------------------
CONFIG_PATH = "config/config.yaml"

if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError("config/config.yaml not found")

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

# ----------------------------
# Config values
# ----------------------------
LOG_PATH = config["paths"].get("logs", "logs/")
MODEL_PATH = config["paths"].get("models", "models/")
FEATURE_STORE_PATH = config["features"].get("feature_store_path", "feature_store/")

TARGET_COL = config["data"].get("target_column", "Loan_Approved")
TEST_SIZE = config["data"].get("test_size", 0.2)
RANDOM_STATE = config["project"].get("random_state", 42)

MLFLOW_URI = config["mlflow"].get("tracking_uri")
EXPERIMENT_NAME = config["mlflow"].get("experiment_name", "loan_prediction_model")

# ----------------------------
# Ensure directories
# ----------------------------
os.makedirs(LOG_PATH, exist_ok=True)
os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(FEATURE_STORE_PATH, exist_ok=True)

# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(
    filename=os.path.join(LOG_PATH, "training.log"),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger("").addHandler(console)

# ----------------------------
# Helpers
# ----------------------------
def get_git_commit_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
    except Exception:
        return "N/A"

def get_dvc_checksum(path):
    if not os.path.exists(path):
        return "N/A"

    hash_md5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

# ----------------------------
# Training Pipeline
# ----------------------------
def main():
    logging.info("===== TRAINING PIPELINE STARTED =====")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # MLflow setup
    if MLFLOW_URI:
        mlflow.set_tracking_uri(MLFLOW_URI)

    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=f"train_{timestamp}"):

        # ----------------------------
        # Load Data
        # ----------------------------
        df = load_data()

        if df.empty:
            raise ValueError("Dataset is empty")

        logging.info(f"Data Loaded | Shape: {df.shape}")

        # ----------------------------
        # Feature Engineering
        # ----------------------------
        df = feature_engineering(df)

        feature_path = os.path.join(FEATURE_STORE_PATH, f"features_{timestamp}.pkl")
        save_features(df, path=feature_path)

        logging.info(f"Feature Engineering Done | Shape: {df.shape}")

        # ----------------------------
        # Split Data
        # ----------------------------
        X_train, X_test, y_train, y_test = split_data(
            df,
            target=TARGET_COL,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE
        )

        logging.info(f"Split Done | Train: {X_train.shape} | Test: {X_test.shape}")

        # ----------------------------
        # Build Pipeline
        # ----------------------------
        pipeline: Pipeline = build_model_pipeline(X_train)

        logging.info("Model Pipeline Created")

        # ----------------------------
        # Train Model
        # ----------------------------
        pipeline.fit(X_train, y_train)

        logging.info("Model Training Completed")

        # ----------------------------
        # Evaluate Model
        # ----------------------------
        y_pred = pipeline.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        logging.info(f"Accuracy: {acc:.4f}")
        logging.info(f"Precision: {prec:.4f}")
        logging.info(f"Recall: {rec:.4f}")
        logging.info(f"F1 Score: {f1:.4f}")

        # ----------------------------
        # Save Model & Pipeline (STEP 17 UPGRADE)
        # ----------------------------
        model_path = os.path.join(MODEL_PATH, f"model_{timestamp}.pkl")
        pipeline_path = os.path.join(MODEL_PATH, f"pipeline_{timestamp}.pkl")

        joblib.dump(pipeline, model_path)
        joblib.dump(pipeline, pipeline_path)

        logging.info(f"Model Saved at: {model_path}")
        logging.info(f"Pipeline Saved at: {pipeline_path}")

        # ----------------------------
        # MLflow Logging
        # ----------------------------
        mlflow.log_param("model", "RandomForest")
        mlflow.log_param("git_commit", get_git_commit_hash())

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)

        mlflow.sklearn.log_model(pipeline, "model")

        # Organized artifacts
        mlflow.log_artifact(model_path, artifact_path="model")
        mlflow.log_artifact(pipeline_path, artifact_path="pipeline")
        mlflow.log_artifact(feature_path, artifact_path="features")

        # DVC checksum
        dvc_checksum = get_dvc_checksum("data/loan.csv")
        mlflow.log_param("dvc_checksum", dvc_checksum)

        # ----------------------------
        # Metadata
        # ----------------------------
        metadata = {
            "timestamp": timestamp,
            "model_path": model_path,
            "pipeline_path": pipeline_path,
            "features_path": feature_path,
            "metrics": {
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1_score": f1
            },
            "git_commit": get_git_commit_hash(),
            "dvc_checksum": dvc_checksum
        }

        metadata_path = os.path.join(MODEL_PATH, f"metadata_{timestamp}.pkl")
        joblib.dump(metadata, metadata_path)

        mlflow.log_artifact(metadata_path, artifact_path="metadata")

        logging.info("Metadata Saved")
        logging.info("===== TRAINING PIPELINE COMPLETED =====")


if __name__ == "__main__":
    main()
import os
import joblib
import mlflow


def save_pipeline_artifact(model_pipeline, filename="loan_pipeline.pkl"):
    os.makedirs("models", exist_ok=True)

    pipeline_path = os.path.join("models", filename)

    # Save locally
    joblib.dump(model_pipeline, pipeline_path)

    # Log to MLflow (NO new run here)
    mlflow.log_artifact(pipeline_path)

    print(f"✅ Pipeline saved & logged: {pipeline_path}")
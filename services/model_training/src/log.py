import logging
import os

import joblib
import mlflow
from hdbscan.validity import validity_index
from sklearn.metrics import silhouette_score

from . import config


def log_experiment(
    run_id, famd_model, scalers, clusterer, noise_clf, X_scaled, df_final
):
    """Logs all parameters, metrics, and artifacts to MLflow."""
    logging.info("--- Logging Experiment to MLflow ---")

    # --- Log Parameters ---
    mlflow.log_param("famd_components", config.FAMD_COMPONENTS)
    mlflow.log_params(config.HDBSCAN_PARAMS)

    # --- Log Metrics ---
    # Initial metrics (before noise handling)
    initial_noise_ratio = (df_final["cluster"] == -1).sum() / len(df_final)
    dbcv_score = validity_index(
        X_scaled, df_final["cluster"].values, metric="cityblock"
    )
    mlflow.log_metrics(
        {"initial_noise_ratio": initial_noise_ratio, "dbcv_score": dbcv_score}
    )

    # Final metrics (after noise handling)
    final_silhouette = silhouette_score(X_scaled, df_final["cluster_final"])
    mlflow.log_metric("final_silhouette_score", final_silhouette)
    logging.info(
        f"Final Silhouette Score (after noise handling): {final_silhouette:.4f}"
    )
    # --- Log Artifacts ---
    # Save models
    joblib.dump(famd_model, os.path.join(config.ARTIFACT_PATH, "famd_model.joblib"))
    joblib.dump(
        scalers["behavioral"],
        os.path.join(config.ARTIFACT_PATH, "behavioral_scaler.joblib"),
    )
    joblib.dump(clusterer, os.path.join(config.ARTIFACT_PATH, "hdbscan_model.joblib"))
    if noise_clf:
        joblib.dump(
            noise_clf,
            os.path.join(config.ARTIFACT_PATH, "noise_classifier_model.joblib"),
        )

    # Save final clustered data
    df_final.to_parquet(
        os.path.join(config.ARTIFACT_PATH, "final_clustered_stores.parquet")
    )

    # Log the entire artifacts directory
    mlflow.log_artifacts(config.ARTIFACT_PATH, artifact_path="results")
    logging.info(
        f"Clustering complete. Artifacts saved to '{config.ARTIFACT_PATH}' and logged to MLflow run {run_id}."
    )

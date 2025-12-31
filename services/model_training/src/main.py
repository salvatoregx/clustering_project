"""
Model Training Pipeline for Retail Store Clustering.

This is the main entry point for the model training service. It orchestrates
the pipeline by loading data, preprocessing, clustering, and logging the
results with MLflow.
"""

import logging

import mlflow
from mlflow.tracking import MlflowClient

from . import cluster, config, data, log, preprocess

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def main():
    """Main execution function for the model training pipeline."""
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)

    client = MlflowClient()
    experiment = client.get_experiment_by_name(config.MLFLOW_EXPERIMENT_NAME)

    if experiment and experiment.lifecycle_stage == "deleted":
        logging.warning(
            f"Experiment '{config.MLFLOW_EXPERIMENT_NAME}' was deleted. Restoring it..."
        )
        client.restore_experiment(experiment.experiment_id)

    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run() as run:
        logging.info(f"MLflow Run ID: {run.info.run_id}")
        mlflow.set_tag("service", "model_training")

        # 1. Load Data
        df_original = data.load_and_prepare_data()

        # 2. Feature Selection & Preprocessing
        df_context, df_behavioral = preprocess.select_and_transform_features(
            df_original
        )
        X_final, famd_model, behavioral_scaler = preprocess.run_preprocessing(
            df_context, df_behavioral
        )

        # 3. Clustering
        df_clustered, clusterer_model, noise_classifier_model = cluster.run_clustering(
            X_final, df_original
        )

        # 4. Logging
        scalers = {"behavioral": behavioral_scaler}
        log.log_experiment(
            run_id=run.info.run_id,
            famd_model=famd_model,
            scalers=scalers,
            clusterer=clusterer_model,
            noise_clf=noise_classifier_model,
            X_scaled=X_final,
            df_final=df_clustered,
        )


if __name__ == "__main__":
    main()

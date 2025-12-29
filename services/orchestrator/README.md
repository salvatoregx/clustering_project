# Orchestrator Service

This service is the central control plane for the platform. It uses **Dagster** to define, schedule, and monitor the data pipeline.

## Architecture: Code Locations (gRPC)

Instead of a monolithic image containing all dependencies, this project uses Dagster's **gRPC Code Location** pattern.

*   The Orchestrator container only runs the Dagster Webserver and Daemon.
*   It connects to the other services (`data_gen`, `etl`, `model_training`) over the Docker network on port `4000`.
*   Each service runs its own lightweight Dagster gRPC server that serves its specific asset definitions.

**Why?** This ensures complete **dependency isolation**. The ETL service can use PySpark and Java, while the Model Training service uses Scikit-learn, without conflict or bloated images.

## Pipeline

1.  **`raw_data_parquet`** (from `data_gen`): Generates the raw data.
2.  **`store_features_parquet`** (from `etl`): Depends on raw data; runs ETL.
3.  **`clustering_model`** (from `model_training`): Depends on features; runs modeling.

The pipeline can be triggered manually via the UI or scheduled.

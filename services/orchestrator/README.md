# Orchestrator Service

This service is the central control plane for the platform. It uses **Dagster** to define, schedule, and monitor the data pipeline.

## Architecture: Code Locations (gRPC)

Instead of a monolithic image containing all dependencies, this project uses Dagster's **gRPC Code Location** pattern.

*   The Orchestrator container only runs the Dagster Webserver and Daemon.
*   It connects to the other services (`data_gen`, `etl`, `model_training`) over the Docker network on port `4000`.
*   Each service runs its own lightweight Dagster gRPC server that serves its specific asset definitions.

**Why?** This ensures complete **dependency isolation**. The ETL service can use PySpark and Java, while the Model Training service uses Scikit-learn, without conflict or bloated images.

## Pipeline

The pipeline is defined as a graph of data assets, where each asset is a table or file produced by one of the services.

1.  **`raw_data_parquet`** (from `data_gen`): Generates the raw, synthetic data.
2.  **`store_features_parquet`** (from `etl`): Depends on the raw data; runs the PySpark ETL job.
3.  **`clustering_model`** (from `model_training`): Depends on the processed features; runs the modeling pipeline.

### Triggering Runs

Because each service is an independent code location, the "Materialize All" button will not work across services. You must materialize assets in the correct dependency order:

1.  In the Dagster UI, select the `raw_data_parquet` asset and click "Materialize".
2.  Once complete, select the `store_features_parquet` asset and click "Materialize".
3.  Finally, select the `clustering_model` asset and click "Materialize".

Alternatively, the assets are configured with an `AutoMaterializePolicy`, so the Dagster Daemon will automatically trigger runs in the correct order as upstream data changes.

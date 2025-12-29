# Model Training Service

This service performs the machine learning workflow to cluster retail stores. It loads the processed features, applies dimensionality reduction, clusters the data, and logs the results.

## Features

*   **Dimensionality Reduction**: Uses **FAMD** (Factor Analysis of Mixed Data) via the `prince` library to handle both numerical and categorical features (e.g., Region, Store Type) simultaneously.
*   **Clustering**: Uses **HDBSCAN**, a density-based algorithm that finds clusters of varying shapes and sizes without pre-specifying `k`.
*   **Noise Handling**: Trains a `RandomForestClassifier` on the confident clusters to classify "noise" points identified by HDBSCAN, ensuring every store is assigned a cluster.
*   **Experiment Tracking**: Logs all parameters (FAMD components, HDBSCAN settings), metrics (Silhouette Score, DBCV), and artifacts (models, datasets) to **MLflow**.
*   **Dagster Integration**: Exposes a gRPC server for the Orchestrator.

## Tech Stack

*   **Python 3.11**
*   **Scikit-learn**: Preprocessing and classification.
*   **Prince**: FAMD implementation.
*   **HDBSCAN**: Clustering algorithm.
*   **MLflow**: Experiment tracking and model registry.
*   **Dagster**: Orchestration integration.

## Output

*   **Artifacts**: Saved to `/opt/data/artifacts` and logged to MLflow.
    *   `final_clustered_stores.parquet`: The dataset with assigned clusters.
    *   `famd_model.joblib`
    *   `hdbscan_model.joblib`
    *   `noise_classifier_model.joblib`
    *   `behavioral_scaler.joblib`

## Design Choice: FAMD Scaling
We scale the FAMD components by the square root of their eigenvalues (singular values) before clustering. This preserves the relative importance of the components (variance explained) when calculating distances, rather than treating all dimensions equally.

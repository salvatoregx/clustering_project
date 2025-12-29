# Retail Store Clustering Platform

This project is a production-grade data science platform designed to segment retail stores based on their physical characteristics, business context, and sales performance. It demonstrates a full end-to-end workflow from synthetic data generation to model deployment and visualization.

## Architecture

The project follows a **Microservices Architecture**, where each stage of the pipeline is isolated in its own container. This ensures dependency isolation, scalability, and maintainability.

### Services

1.  **`data_gen`**: Generates realistic synthetic retail data (Stores, Products, Sales, Inventory) using `Faker` and `Pandas`. It simulates business logic like store clusters and weekly seasonality.
2.  **`etl`**: A PySpark-based ETL pipeline that ingests raw transactional data, validates it with `Great Expectations`, and aggregates it into store-level features. It handles data cleaning and advanced imputation.
3.  **`model_training`**: The machine learning core. It uses `Prince` for Factor Analysis of Mixed Data (FAMD) to handle mixed feature types and `HDBSCAN` for density-based clustering. It logs experiments to `MLflow`.
4.  **`orchestrator`**: A `Dagster` instance that orchestrates the pipeline. It treats the other services as "Code Locations" (gRPC servers), triggering execution within their respective environments.
5.  **`streamlit_app`**: A user-facing dashboard for monitoring data integrity, visualizing clusters (t-SNE), and providing actionable business recommendations.
6.  **`mlflow`**: A tracking server for experiment management, model versioning, and artifact storage.

## Tech Stack & Design Choices

| Component | Tool | Why? |
| :--- | :--- | :--- |
| **Orchestration** | **Dagster** | Provides a data-aware orchestration layer with excellent UI for lineage and asset management. We use the gRPC pattern to keep service dependencies isolated. |
| **ETL** | **PySpark** | Chosen for its ability to handle large-scale transactional data efficiently. Essential for aggregating millions of sales rows into store features. |
| **Validation** | **Great Expectations** | Ensures data quality at the ingestion layer, preventing bad data from corrupting the model. |
| **ML Model** | **FAMD + HDBSCAN** | **FAMD** reduces dimensionality for mixed (cat/num) data. **HDBSCAN** is a robust density-based clustering algorithm that doesn't require pre-specifying `k` and handles noise well. |
| **Tracking** | **MLflow** | Standard for MLOps. Tracks parameters, metrics, and saves trained models as artifacts. |
| **App** | **Streamlit** | Rapid development of interactive data apps. Perfect for visualizing t-SNE plots and business profiles. |
| **Containerization** | **Docker / Podman** | Ensures reproducibility across environments. |

## Getting Started

### Prerequisites
*   Docker or Podman
*   `docker-compose` or `podman-compose`
*   `make` (optional, for convenience)

### Running the Platform

1.  **Start the Platform**:
    This builds all images and starts the services in the background.
    ```bash
    make run-platform
    # OR
    podman-compose up --build --detach
    ```

2.  **Access the Interfaces**:
    *   **Dagster UI (Orchestrator):** http://localhost:3000
        *   Go here to trigger the pipeline run (`Materialize All`).
    *   **Streamlit Dashboard:** http://localhost:8501
        *   View the results and data integrity checks after the pipeline runs.
    *   **MLflow UI:** http://localhost:5000
        *   Inspect model training runs and artifacts.

3.  **Stop the Platform**:
    ```bash
    make stop
    ```

## Project Structure

```
├── Makefile             # Shortcuts for common commands
├── compose.yaml         # Docker Compose definition
├── data/                # Mounted volume for data persistence
├── services/
│   ├── data_gen/        # Synthetic Data Generator
│   ├── etl/             # PySpark ETL Pipeline
│   ├── model_training/  # Scikit-learn/HDBSCAN Modeling
│   ├── orchestrator/    # Dagster Orchestrator
│   └── streamlit_app/   # Visualization Dashboard
└── ...
```

## Development

Each service has its own `pyproject.toml` for dependency management.

To run tests for a specific service (e.g., `data_gen`):
```bash
cd services/data_gen
poetry install
pytest
```

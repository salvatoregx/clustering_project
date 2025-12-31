# From Raw Data to Retail Strategy: A Deep Dive into Store Segmentation
[![CI](https://github.com/salvatoregx/clustering_project/actions/workflows/ci.yml/badge.svg)](https://github.com/salvatoregx/clustering_project/actions/workflows/ci.yml)

This flagship portfolio project showcases a complete, production-grade data science solution for strategic retail segmentation. It moves beyond simplistic revenue-based groupings to uncover nuanced store archetypes by analyzing a rich blend of performance, behavioral, and contextual data. The outcome is a powerful, interactive platform that translates advanced unsupervised learning models (FAMD + HDBSCAN) into actionable business intelligence, enabling targeted strategies for marketing, inventory, and operations.

The project is built as a robust, scalable microservices-based platform, showcasing not only advanced modeling but also the principles of production-grade data science.

## Architecture

The project is built on a **Microservices Architecture**, with each pipeline stage isolated in its own container. This modern approach, orchestrated by Dagster, ensures complete dependency isolation and mirrors a real-world production environment. For example, the ETL service can leverage PySpark and its Java dependencies without conflicting with the Python-native ML libraries in the modeling service.

### Services

1.  **`data_gen`**: Generates realistic synthetic retail data (Stores, Products, Sales, Inventory) using `Faker` and `Pandas`. It simulates business logic like store clusters and weekly seasonality.
2.  **`etl`**: A PySpark-based ETL pipeline that ingests raw transactional data, validates it with `Great Expectations`, and aggregates it into store-level features. It handles data cleaning and advanced imputation.
3.  **`model_training`**: The data science core. It uses `Prince` for Factor Analysis of Mixed Data (FAMD) to handle mixed feature types and `HDBSCAN` for density-based clustering. It logs experiments to `MLflow`.
4.  **`orchestrator`**: A `Dagster` instance that orchestrates the pipeline. It treats the other services as "Code Locations" (gRPC servers), triggering execution within their respective environments.
5.  **`streamlit_app`**: A user-facing dashboard for visualizing clusters, exploring their business profiles, and understanding strategic recommendations.
6.  **`mlflow`**: A tracking server for experiment management, model versioning, and artifact storage.

## Tech Stack & Rationale

| Component | Tool | Rationale from a Data Science Perspective |
| :--- | :--- | :--- |
| **Clustering Model** | **FAMD + HDBSCAN** | This combination is the core of the data science strategy. **FAMD** was chosen for its sophisticated ability to perform dimensionality reduction on mixed-type datasets, preserving information from both categorical and numerical features without the drawbacks of one-hot encoding. **HDBSCAN** is a powerful, density-based algorithm that identifies clusters of varying shapes and handles outliers gracefully—a significant advantage over traditional methods like K-Means that require pre-specifying the number of clusters. |
| **ETL & Feature Eng.** | **PySpark** | Demonstrates the ability to scale feature engineering beyond in-memory `pandas` limitations. The pipeline aggregates millions of simulated transactions into a rich, store-level feature set, a common task in enterprise-level data science. The imputation of missing `size_m2` values using a regression model showcases a more advanced, data-driven approach to data cleaning. |
| **Data Quality** | **Great Expectations** | A proactive approach to data quality. By defining and enforcing data contracts at the ingestion stage, we ensure the reliability of the features fed into the model, preventing the "garbage in, garbage out" problem and increasing stakeholder trust in the results. |
| **Orchestration** | **Dagster** | Chosen for its asset-centric view, which provides clear data lineage from raw data to final model. This is crucial for governance, reproducibility, and explaining model predictions to stakeholders. |
| **Results & Insights** | **Streamlit** | An effective tool for **storytelling with data**. It translates complex model outputs into an interactive, business-friendly dashboard, allowing stakeholders to explore cluster personas and understand the actionable recommendations derived from the analysis. |
| **Experiment Tracking**| **MLflow** | The industry standard for MLOps, used here to log model parameters, metrics (DBCV, Silhouette Score), and version trained model artifacts, ensuring reproducibility and facilitating model comparison. |
| **Containerization** | **Docker / Podman** | Ensures reproducibility across environments. |

## Getting Started

### Prerequisites
*   Docker or Podman
*   `docker-compose` or `podman-compose`
*   `make` (optional, for convenience)

### Running the Platform

**Note on First Build:** The initial build process may take several minutes as it constructs container images for each microservice, including downloading dependencies like PySpark. Subsequent builds will be significantly faster due to Docker/Podman layer caching.

1.  **Start the Platform**:
    This command builds all images and starts the services in the background.
    ```bash
    make run-platform
    # OR
    podman-compose -f compose.yaml up --build --detach
    ```

2.  **Access the Dagster UI (Orchestrator):**
    *   Navigate to **http://localhost:3000**

3.  **Trigger the Pipeline**:
    *   In the Dagster UI, you will see assets from three different code locations: `data_gen`, `etl`, and `model_training`.
    *   **Important**: As all services are independent code spaces, "Materialize all" will not work. Each service's assets must be materialized manually in sequence if their pre-configured automation is not enabled.
    *   **Step 1:** Select the `raw_data_parquet` asset and click "Materialize".
    *   **Step 2:** Once complete, select the `store_features_parquet` asset and click "Materialize".
    *   **Step 3:** Finally, select the `clustering_model` asset and click "Materialize".

4.  **Explore the Results**:
    *   **Streamlit Dashboard:** http://localhost:8501
        *   Explore cluster profiles, visualizations, and business recommendations.
    *   **MLflow UI:** http://localhost:5000
        *   Inspect the model training run, including parameters, metrics (DBCV, Silhouette Score), and saved model artifacts.

5.  **Stop the Platform**:
    ```bash
    make stop
    ```

## Development & Experimentation (Lab Mode)

A core philosophy of this project is to balance production-readiness with the iterative, experimental nature of data science. The "Lab Mode" is a purpose-built environment that facilitates this hybrid approach.

**Start the Lab Environment:**
```bash
make lab
```
This command starts all platform services but overrides the `model_training` service to launch a **Jupyter Notebook server** instead of the gRPC server.

**Key Features:**
*   **Interactive Experimentation**: The Jupyter environment comes pre-installed with all dependencies from the `model_training` service.
*   **Live Data Access**: The `data/` volume is mounted, giving you direct access to the same data used by the production pipeline.
*   **Hyperparameter Tuning**: The `services/model_training/src/notebooks/fine_tuning.ipynb` notebook provides a complete workflow for running a grid search on model parameters (FAMD components, HDBSCAN settings) and visualizing the trade-offs between key metrics like DBCV and Silhouette Score.

This setup empowers data scientists to rapidly test hypotheses, tune models, and validate changes before integrating them back into the main Dagster pipeline, demonstrating a realistic and effective MLOps workflow.

## Project Structure

```
├── README.md            # This file
├── Makefile             # Shortcuts for common commands
├── compose.yaml         # Podman Compose definition
├── compose.lab.yaml     # Overwrite for Lab mode
├── data/                # Mounted volume for data persistence
├── services/
│   ├── data_gen/        # Synthetic Data Generator
│   ├── etl/             # PySpark ETL Pipeline
│   ├── model_training/  # Scikit-learn/HDBSCAN Modeling
│   ├── orchestrator/    # Dagster Orchestrator
│   └── streamlit_app/   # Visualization Dashboard
├── tests/               # Integration tests
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

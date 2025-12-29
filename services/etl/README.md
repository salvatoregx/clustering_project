# ETL Service

This service handles the Extraction, Transformation, and Loading of data. It ingests raw Parquet files, validates them, performs feature engineering, and aggregates transactional data into store-level features.

## Features

*   **Data Validation**: Uses **Great Expectations** to validate raw data (e.g., checking for nulls, ensuring positive sales quantities) before processing. Results are saved as JSON for the dashboard.
*   **Scalable Processing**: Uses **PySpark** to handle large volumes of sales and inventory data efficiently.
*   **Feature Engineering**:
    *   **Imputation**: Uses Linear Regression to impute missing `size_m2` values based on `storage_capacity_m2`.
    *   **Aggregation**: Calculates metrics like `Total Revenue`, `Revenue per Sqm`, `Avg Ticket`, and `Profit Margin`.
    *   **Pivoting**: Creates category-level revenue features.
*   **Dagster Integration**: Exposes a gRPC server for the Orchestrator.

## Tech Stack

*   **Python 3.11**
*   **PySpark**: Distributed data processing.
*   **Great Expectations**: Data quality validation.
*   **Dagster**: Orchestration integration.

## Output

*   **Processed Data**: `/opt/data/processed/store_features.parquet`
    *   A single Parquet file containing aggregated features for each store, ready for modeling.
*   **Validation Results**: `/opt/data/validation/*.json`
    *   JSON reports of data quality checks.

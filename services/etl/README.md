# ETL & Feature Engineering Service

This service transforms raw, event-level data into a high-quality, aggregated feature set ready for machine learning. It demonstrates best practices in data validation, scalable processing, and thoughtful feature engineering.

## Key Data Science Contributions

*   **Data Validation with Great Expectations**: Before any processing, raw data is validated against a suite of expectations (e.g., sales quantities must be positive, IDs must not be null). This is a critical first step in any robust ML pipeline to prevent data quality issues from silently corrupting model results.
*   **Scalable Feature Aggregation with PySpark**: The pipeline is built with PySpark to demonstrate the ability to handle large-scale transactional data that would not fit into memory with `pandas`. It aggregates millions of sales records to create meaningful store-level features.
*   **Feature Engineering**:
    *   **Advanced Imputation Strategy**: Instead of using a simple mean or median, missing `size_m2` values are imputed using a **Linear Regression model** trained on the highly correlated `storage_capacity_m2` feature. This more sophisticated approach leads to a more accurate and representative feature set.
    *   **Rich Feature Creation**: The service engineers a variety of features crucial for clustering, including performance metrics (`Total Revenue`, `Revenue per Sqm`), behavioral indicators (`Avg Ticket Item`), and product mix insights (pivoted category-level revenues).

## Tech Stack

*   **Python 3.11**
*   **PySpark**: Distributed data processing.
*   **Great Expectations**: Data quality validation.

## Output

*   **Processed Data**: `/opt/data/processed/store_features.parquet`
    *   A single Parquet file containing aggregated features for each store, ready for modeling.
*   **Validation Results**: `/opt/data/validation/*.json`
    *   JSON reports of data quality checks.

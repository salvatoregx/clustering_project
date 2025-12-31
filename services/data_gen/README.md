# Synthetic Data Generation Service

This service generates a realistic, multi-faceted dataset that serves as a robust testbed for the entire data science pipeline. The goal is to simulate the complexities and imperfections of real-world retail data, ensuring that the downstream ETL and modeling components are built to be resilient and effective.

## Simulating Real-World Data Challenges

The generation process goes beyond random data to incorporate patterns and challenges commonly found in retail analytics:

*   **Inherent Store Tiers**: Stores are assigned a "cluster" label (`Gold/Silver/Bronze`) at generation time, which drives their baseline sales volume. This provides a ground truth to evaluate whether the unsupervised clustering model can successfully rediscover these underlying performance tiers.
*   **Seasonality**: Sales data is simulated with weekly seasonality, with higher volumes on weekends, to mimic real customer behavior.
*   **Data Quality Issues**: To test the robustness of the ETL pipeline, null values are intentionally introduced into key fields like `size_m2`. This forces the downstream feature engineering process to include a thoughtful imputation strategy rather than assuming perfect data.
*   **Coherent History**: Sales and inventory data are generated in a synchronized, week-by-week process. This ensures that inventory levels are logically consistent with sales velocity, creating a more realistic dataset for any time-series or stock-level analysis.

## Tech Stack

*   **Python 3.11**
*   **Faker**: For generating realistic names, addresses, and dates.
*   **Pandas / NumPy**: For vectorized data generation and manipulation.
*   **PyArrow**: For efficient writing of partitioned Parquet files, a common format in big data ecosystems.
*   **Dagster**: For orchestration integration.

## Output

Data is saved to `/opt/data/raw` (mounted volume):
*   `dim_product.parquet`
*   `dim_store_physical.parquet`
*   `dim_store_management.parquet`
*   `fact_sales/` (Partitioned by Year/Month)
*   `fact_inventory/` (Partitioned by Year)

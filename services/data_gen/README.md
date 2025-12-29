# Data Generation Service

This service is responsible for creating the synthetic dataset used by the rest of the platform. It simulates a retail environment with stores, products, and historical transaction data.

## Features

*   **Realistic Entities**: Generates Stores with physical (size, location) and management attributes. Generates Products with categories, costs, and prices.
*   **Geographic Coherence**: Ensures stores are assigned to valid City/State/Region hierarchies using a predefined map.
*   **Synchronized History**: Generates 3 years of Sales and Inventory data. Sales volume is driven by store "clusters" (Gold/Silver/Bronze) and weekly seasonality. Inventory levels are consistent with the active assortment.
*   **Data Quality Injection**: Intentionally introduces null values (e.g., in `size_m2`) to test the ETL pipeline's imputation logic.
*   **Dagster Integration**: Exposes a gRPC server to allow the Orchestrator to trigger generation.

## Tech Stack

*   **Python 3.11**
*   **Faker**: For generating realistic names, addresses, and dates.
*   **Pandas / NumPy**: For vectorized data generation and manipulation.
*   **PyArrow**: For efficient writing of partitioned Parquet files.
*   **Dagster**: For orchestration integration.

## Output

Data is saved to `/opt/data/raw` (mounted volume):
*   `dim_product.parquet`
*   `dim_store_physical.parquet`
*   `dim_store_management.parquet`
*   `fact_sales/` (Partitioned by Year/Month)
*   `fact_inventory/` (Partitioned by Year)

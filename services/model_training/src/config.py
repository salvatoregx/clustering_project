import os

# --- Paths ---
PROCESSED_DATA_PATH = "/opt/data/processed/store_features.parquet"
ARTIFACT_PATH = "/opt/data/artifacts"
os.makedirs(ARTIFACT_PATH, exist_ok=True)

# --- MLflow ---
MLFLOW_TRACKING_URI = "http://mlflow:5000"
MLFLOW_EXPERIMENT_NAME = "Retail Store Clustering"

# --- Data Loading ---
SAFE_PANDAS_LIMIT = 1_000_000 # Rows

# --- Feature Engineering ---
DATE_COL = 'opening_date'

# --- Modeling ---
# Columns for dimensionality reduction (mixed data)
CONTEXT_COLS_CAT = ['regional', 'placement']
CONTEXT_COLS_NUM = ['anos_de_loja', 'size_m2']

# Columns for direct use in clustering (behavioral)
BEHAVIORAL_COLS = [
    'total_revenue', 
    'avg_ticket_item', 
    'revenue_per_sqm', 
    'avg_inventory_depth', 
    'unique_products_stocked'
]

# FAMD parameters
FAMD_COMPONENTS = 5

# HDBSCAN parameters
HDBSCAN_PARAMS = {
    "min_cluster_size": 5,
    "min_samples": 1,
    "cluster_selection_epsilon": 1.0,
    "metric": 'manhattan',
    "cluster_selection_method": 'eom'
}

HDBSCAN_TEST_METRIC = 'cityblock'

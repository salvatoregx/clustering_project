import pandas as pd
from . import config

def load_and_prepare_data():
    """
    Loads the feature-engineered data from the ETL service and performs
    final minor transformations inspired by the notebook.
    """
    print("--- Loading and Preparing Data ---")
    df = pd.read_parquet(config.PROCESSED_DATA_PATH)
    
    if 'store_id' in df.columns:
        df.set_index('store_id', inplace=True)
    
    # Feature Engineering: Create 'anos_de_loja' (store age)
    df[config.DATE_COL] = pd.to_datetime(df[config.DATE_COL])
    df['anos_de_loja'] = (pd.Timestamp.now() - df[config.DATE_COL]).dt.days / 365.25
    
    print(f"Loaded {len(df)} stores for clustering.")
    return df

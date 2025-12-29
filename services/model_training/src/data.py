import pandas as pd
import logging
from . import config

def load_and_prepare_data():
    """
    Loads the feature-engineered data from the ETL service and performs
    final minor transformations inspired by the notebook.
    """
    logging.info("--- Loading and Preparing Data ---")
    df = pd.read_parquet(config.PROCESSED_DATA_PATH)
    
    if len(df) > config.SAFE_PANDAS_LIMIT:
        logging.warning(f"Dataframe size ({len(df)} rows) exceeds safe limit of {config.SAFE_PANDAS_LIMIT}. Consider sampling.")
        # For this project, we'll proceed, but in production, you might sample or raise an error.
        # df = df.sample(n=config.SAFE_PANDAS_LIMIT, random_state=42)

    
    if 'store_id' in df.columns:
        df.set_index('store_id', inplace=True)
    
    # Feature Engineering: Create 'anos_de_loja' (store age)
    df[config.DATE_COL] = pd.to_datetime(df[config.DATE_COL])
    df['anos_de_loja'] = (pd.Timestamp.now() - df[config.DATE_COL]).dt.days / 365.25
    
    logging.info(f"Loaded {len(df)} stores for clustering.")
    return df

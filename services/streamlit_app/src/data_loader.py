import streamlit as st
import pandas as pd
import os
from . import config

@st.cache_data
def load_clustered_data():
    """Loads the clustered data from the model training artifacts."""
    if os.path.exists(config.CLUSTERED_DATA_PATH):
        return pd.read_parquet(config.CLUSTERED_DATA_PATH)
    return None
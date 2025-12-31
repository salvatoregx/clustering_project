import streamlit as st
import pandas as pd
import os
from tabs import config


def load_clustered_data():
    """Loads the clustered data, checking for file updates to invalidate cache."""
    if not os.path.exists(config.CLUSTERED_DATA_PATH):
        return None

    # Pass modification time to force cache invalidation when file changes
    mtime = os.path.getmtime(config.CLUSTERED_DATA_PATH)
    return _load_data_cached(mtime)


@st.cache_data
def _load_data_cached(mtime):
    """Actual data loading, cached based on mtime."""
    return pd.read_parquet(config.CLUSTERED_DATA_PATH)

import streamlit as st

from . import data_loader
from .tabs import data_integrity, visualization, business_actions

def main():
    """Main function to run the Streamlit application."""
    st.set_page_config(page_title="Retail Clustering Dashboard", layout="wide")
    st.title("Retail Store Clustering & Monitoring")

    # --- Tab Definitions ---
    tab_integrity, tab_viz, tab_actions = st.tabs([
        "Data Integrity", 
        "Cluster Visualization", 
        "Business Actions"
    ])

    with tab_integrity:
        data_integrity.show()

    # Load data required for the other tabs
    df_clustered = data_loader.load_clustered_data()

    if df_clustered is None:
        st.error("Clustered data not found. Please run the full pipeline to generate artifacts.")
    else:
        with tab_viz:
            visualization.show(df_clustered)
        with tab_actions:
            business_actions.show(df_clustered)

if __name__ == "__main__":
    main()

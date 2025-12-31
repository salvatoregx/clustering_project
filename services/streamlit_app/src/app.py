import streamlit as st

import data_loader
from tabs import data_integrity, visualization, business_actions

def main():
    """Main function to run the Streamlit application."""
    st.set_page_config(page_title="Retail Store Segmentation", layout="wide")
    st.title("Retail Store Segmentation & Strategy")
    st.markdown("An end-to-end platform for identifying store archetypes and deriving actionable business insights.")

    # Load data first
    df_clustered = data_loader.load_clustered_data()

    if df_clustered is None:
        st.error("Clustered data not found. Please run the full pipeline via Dagster to generate model artifacts.")
        st.info("To run the pipeline: `make run-platform`, then trigger the materializations in the Dagster UI at http://localhost:3000.")
        return

    # --- High-Level KPIs ---
    st.header("Platform Overview")
    total_stores = len(df_clustered)
    num_clusters = df_clustered['cluster_final'].nunique()
    inferred_stores = df_clustered['is_inferred'].sum()
    inferred_percentage = (inferred_stores / total_stores) * 100 if total_stores > 0 else 0

    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Total Stores Analyzed", f"{total_stores}")
    kpi2.metric("Identified Clusters", f"{num_clusters}")
    kpi3.metric("Inferred Cluster Assignments", f"{inferred_percentage:.1f}%", help="Percentage of stores initially classified as noise by HDBSCAN and subsequently assigned to the nearest cluster.")


    # --- Tab Definitions ---
    tab_viz, tab_actions, tab_integrity = st.tabs([
        "Cluster Explorer",
        "Strategic Insights",
        "Data Integrity"
    ])

    with tab_viz:
        visualization.show(df_clustered)
    with tab_actions:
        business_actions.show(df_clustered)
    with tab_integrity:
        data_integrity.show()

if __name__ == "__main__":
    main()

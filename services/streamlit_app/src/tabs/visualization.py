import streamlit as st
import plotly.express as px
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

def show(df):
    """Displays the Cluster Visualization tab with a t-SNE plot."""
    st.header("2D Cluster Visualization (t-SNE)")

    st.sidebar.subheader("Visualization Settings")
    perplexity = st.sidebar.slider("t-SNE Perplexity", 5, 50, 30, key='tsne_perplexity')

    # Prepare features for t-SNE (Numerical only for dimensionality reduction)
    feature_cols = [
        'total_revenue', 'avg_ticket_item', 'revenue_per_sqm',
        'avg_inventory_depth', 'unique_products_stocked', 'size_m2'
    ]
    available_cols = [c for c in feature_cols if c in df.columns]

    if not available_cols:
        st.error("Required features for visualization are not available in the dataset.")
        return

    X = df[available_cols].fillna(0)
    X_scaled = StandardScaler().fit_transform(X)

    # Run t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    projections = tsne.fit_transform(X_scaled)

    df_plot = df.copy()
    df_plot['tsne_1'] = projections[:, 0]
    df_plot['tsne_2'] = projections[:, 1]
    df_plot['Cluster_Label'] = df_plot['cluster_final'].astype(str)

    fig = px.scatter(
        df_plot, x='tsne_1', y='tsne_2',
        color='Cluster_Label',
        hover_data=['store_name', 'regional', 'total_revenue'],
        title="Store Clusters (t-SNE Projection)",
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)
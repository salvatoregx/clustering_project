import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler

def show(df):
    """Displays the Cluster Visualization tab with a t-SNE plot."""
    st.header("Cluster Explorer")

    # --- Sidebar for controls ---
    st.sidebar.subheader("Visualization Settings")
    perplexity = st.sidebar.slider("t-SNE Perplexity", 5, 50, 30, key='tsne_perplexity')
    
    # --- Main t-SNE Plot ---
    st.subheader("2D Map of Store Clusters (t-SNE)")

    # Prepare features for t-SNE (Numerical only for dimensionality reduction)
    feature_cols = [
        'total_revenue', 'avg_ticket_item', 'revenue_per_sqm',
        'avg_inventory_depth', 'unique_products_stocked', 'size_m2', 'anos_de_loja'
    ]
    available_cols = [c for c in feature_cols if c in df.columns]

    if not available_cols:
        st.error("Required features for visualization are not available in the dataset.")
        return

    X = df[available_cols].fillna(0)
    
    # Use MinMaxScaler for t-SNE as it's sensitive to scale
    X_scaled = MinMaxScaler().fit_transform(X)

    # Run t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, n_iter=300)
    projections = tsne.fit_transform(X_scaled)

    df_plot = df.copy()
    df_plot['tsne_1'] = projections[:, 0]
    df_plot['tsne_2'] = projections[:, 1]
    df_plot['Cluster'] = df_plot['cluster_final'].astype(str) # For discrete colors

    fig_tsne = px.scatter(
        df_plot, x='tsne_1', y='tsne_2',
        color='Cluster',
        hover_name='store_name',
        hover_data={
            'regional': True,
            'total_revenue': ':.2f',
            'size_m2': ':.0f',
            'Cluster': False,
            'tsne_1': False,
            'tsne_2': False
        },
        title="Store Clusters (t-SNE Projection)",
        template="plotly_white"
    )
    fig_tsne.update_traces(marker=dict(size=8, opacity=0.8, line=dict(width=1, color='DarkSlateGrey')))
    st.plotly_chart(fig_tsne, use_container_width=True)

    st.divider()

    # --- Cluster Deep Dive Section ---
    st.subheader("Cluster Deep Dive")
    clusters = sorted(df['cluster_final'].unique())
    selected_cluster = st.selectbox("Select a cluster to analyze:", clusters, format_func=lambda x: f"Cluster {x}")

    if selected_cluster is not None:
        df_cluster = df[df['cluster_final'] == selected_cluster]
        
        col1, col2 = st.columns(2)

        # Radar Chart for Profile Comparison
        with col1:
            st.markdown("##### Cluster Profile vs. Average")
            
            # Normalize data for radar chart (0-1 scale)
            scaler = MinMaxScaler()
            df_radar_norm = pd.DataFrame(scaler.fit_transform(df[available_cols]), columns=available_cols)
            df_radar_norm['cluster_final'] = df['cluster_final'].values
            
            cluster_means_norm = df_radar_norm[df_radar_norm['cluster_final'] == selected_cluster][available_cols].mean()
            overall_means_norm = df_radar_norm[available_cols].mean()

            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=cluster_means_norm.values,
                theta=available_cols,
                fill='toself',
                name=f'Cluster {selected_cluster}'
            ))
            fig_radar.add_trace(go.Scatterpolar(
                r=overall_means_norm.values,
                theta=available_cols,
                fill='toself',
                name='Overall Average'
            ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=True,
                height=400
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        # Categorical Feature Distribution
        with col2:
            st.markdown("##### Categorical Feature Distribution")
            cat_cols = ['placement', 'regional']
            
            for col in cat_cols:
                if col in df.columns:
                    dist_df = df_cluster[col].value_counts(normalize=True).mul(100).rename('percentage').reset_index()
                    fig_bar = px.bar(
                        dist_df, 
                        x=col, 
                        y='percentage', 
                        title=f'Distribution of "{col.title()}"',
                        text_auto='.1f'
                    )
                    fig_bar.update_traces(texttemplate='%{y:.1f}%', textposition='outside')
                    fig_bar.update_layout(yaxis_title="Percentage (%)", height=200, margin=dict(t=30, b=10))
                    st.plotly_chart(fig_bar, use_container_width=True)
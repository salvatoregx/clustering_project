import streamlit as st

def show(df):
    """Displays the Business Actions tab with cluster profiles and recommendations."""
    st.header("Cluster Profiles & Recommendations")

    # Group by cluster and calculate mean metrics
    metrics = ['total_revenue', 'revenue_per_sqm', 'avg_ticket_item', 'size_m2']
    available_metrics = [c for c in metrics if c in df.columns]

    if not available_metrics:
        st.error("Required metrics for profiling are not available in the dataset.")
        return

    profile = df.groupby('cluster_final')[available_metrics].mean().reset_index()

    # Add store count to profiles
    counts = df['cluster_final'].value_counts().reset_index()
    counts.columns = ['cluster_final', 'store_count']
    profile = profile.merge(counts, on='cluster_final')

    def get_recommendation(row):
        """Generates a business recommendation based on simple heuristics."""
        if row['revenue_per_sqm'] > profile['revenue_per_sqm'].mean() and row['size_m2'] > profile['size_m2'].mean():
            return "Flagship Stores: Prioritize inventory, launch exclusive events."
        elif row['revenue_per_sqm'] < profile['revenue_per_sqm'].mean() and row['size_m2'] > profile['size_m2'].mean():
            return "Underperforming Large Format: Optimize layout, review product assortment."
        elif row['avg_ticket_item'] > profile['avg_ticket_item'].mean():
            return "High Value Boutique: Send premium catalog, focus on loyalty perks."
        else:
            return "Standard/Discount Focus: Send discount coupons, promote bundle offers."

    profile['Action Plan'] = profile.apply(get_recommendation, axis=1)

    for _, row in profile.iterrows():
        cluster_id = int(row['cluster_final'])
        with st.expander(f"Cluster {cluster_id} ({row['store_count']} stores)", expanded=True):
            col1, col2 = st.columns([1, 2])
            col1.metric("Avg. Revenue", f"R$ {row['total_revenue']:,.2f}")
            col1.metric("Revenue / Sqm", f"R$ {row['revenue_per_sqm']:.2f}")
            col2.info(f"**Strategy:** {row['Action Plan']}")
            col2.dataframe(row.to_frame().T.drop(columns=['Action Plan']), hide_index=True)
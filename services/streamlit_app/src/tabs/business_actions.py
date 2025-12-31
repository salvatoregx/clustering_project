import streamlit as st
import pandas as pd

def generate_cluster_persona(cluster_profile, overall_profile):
    """Generates a business persona and recommendation based on cluster metrics."""
    persona = []
    recommendations = []

    # Revenue per Sqm
    if cluster_profile['revenue_per_sqm'] > overall_profile['revenue_per_sqm'] * 1.2:
        persona.append("High Efficiency")
        recommendations.append("Capitalize on high foot traffic; consider premium product placement.")
    elif cluster_profile['revenue_per_sqm'] < overall_profile['revenue_per_sqm'] * 0.8:
        persona.append("Low Efficiency")
        recommendations.append("Review store layout and staff performance; potential for operational improvements.")

    # Size
    if cluster_profile['size_m2'] > overall_profile['size_m2'] * 1.2:
        persona.append("Large Format")
        recommendations.append("Utilize space for in-store events or expanded product lines.")
    elif cluster_profile['size_m2'] < overall_profile['size_m2'] * 0.8:
        persona.append("Small Format / Boutique")
        recommendations.append("Focus on curated, high-turnover inventory.")

    # Avg Ticket
    if cluster_profile['avg_ticket_item'] > overall_profile['avg_ticket_item'] * 1.1:
        persona.append("High-Value Shoppers")
        recommendations.append("Target with loyalty programs and premium service offerings.")
    else:
        persona.append("Volume-Driven")
        recommendations.append("Focus on promotions, bundle deals, and high-volume items.")

    if not persona:
        return "Standard Performers", ["Monitor performance and maintain current strategy."]

    return ", ".join(persona), recommendations


def show(df):
    """Displays the Business Actions tab with cluster profiles and recommendations."""
    st.header("Cluster Personas & Strategic Recommendations")
    st.markdown("Each cluster is analyzed against the fleet average to generate a persona and suggest targeted business actions.")

    # Calculate overall and cluster-level profiles
    metrics = ['total_revenue', 'revenue_per_sqm', 'avg_ticket_item', 'size_m2']
    available_metrics = [c for c in metrics if c in df.columns]

    if not available_metrics:
        st.error("Required metrics for profiling are not available in the dataset.")
        return

    overall_profile = df[available_metrics].mean()
    cluster_profiles = df.groupby('cluster_final')[available_metrics].mean()
    cluster_counts = df['cluster_final'].value_counts().rename("store_count")
    cluster_profiles = cluster_profiles.join(cluster_counts)

    # Sort clusters by store count for better presentation
    cluster_profiles = cluster_profiles.sort_values("store_count", ascending=False)

    for cluster_id, profile_row in cluster_profiles.iterrows():
        st.subheader(f"Cluster {int(cluster_id)}: Profile")
        
        persona, recommendations = generate_cluster_persona(profile_row, overall_profile)
        st.markdown(f"**Persona:** `{persona}`")

        cols = st.columns(len(available_metrics) + 1)
        for i, metric in enumerate(available_metrics):
            value = profile_row[metric]
            avg_value = overall_profile[metric]
            delta = ((value - avg_value) / avg_value) * 100 if avg_value != 0 else 0
            cols[i].metric(
                label=metric.replace('_', ' ').title(),
                value=f"{value:,.0f}",
                delta=f"{delta:.1f}% vs. Average"
            )
        
        cols[-1].metric(label="Store Count", value=int(profile_row['store_count']))

        with st.container(border=True):
            st.markdown("##### Recommended Actions")
            for rec in recommendations:
                st.markdown(f"- {rec}")
        
        st.divider()
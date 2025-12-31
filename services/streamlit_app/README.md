# Store Segmentation Insights Dashboard

This Streamlit application serves as the primary interface for business stakeholders and data scientists to interact with the results of the clustering pipeline. It is designed to translate complex model outputs into clear, actionable insights.

## Purpose: Bridging ML and Business Strategy

The core goal of this dashboard is to make the store segmentation results accessible and useful. It moves beyond raw data and model artifacts to tell a story about the different store archetypes, enabling data-driven decision-making for marketing, operations, and inventory management.

## Dashboard Tabs

The application is organized into several key tabs, each tailored to a specific audience and purpose:

### 1. Cluster Explorer
This is the analytical deep-dive section.
*   **t-SNE Visualization**: Provides an interactive 2D map of the store clusters, allowing users to visually inspect the separation and density of the identified groups. Hovering over points reveals key store-level information.
*   **Interactive Profiling**: Users can select a specific cluster and immediately see its profile visualized on a **radar chart**, comparing its key metrics against the fleet-wide average. This makes it intuitive to spot what makes a cluster unique (e.g., "high revenue, small footprint").

### 2. Strategic Insights
This tab is designed for business leaders and strategists.
*   **Cluster Personas**: Each cluster is presented as a distinct "persona" (e.g., "High-Efficiency Performers", "Large-Format Underachievers") based on its defining characteristics.
*   **KPI Summaries**: Key performance indicators for each cluster are displayed in a clear, metric-card format, showing performance relative to the average.
*   **Actionable Recommendations**: Based on the cluster persona, a list of targeted, strategic recommendations is provided to guide business actions.

### 3. Data Integrity
This tab provides transparency into the health of the data pipeline.
*   **Great Expectations Reports**: It displays the results of the latest data validation run from the ETL service, showing which data quality checks passed or failed. This builds trust in the data and the models built upon it.

## Tech Stack

*   **Streamlit**: For rapid development of the interactive web application.
*   **Plotly**: For creating rich, interactive visualizations like the t-SNE map and radar charts.
*   **Pandas**: For data manipulation and aggregation within the app.
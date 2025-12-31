# Advanced Store Segmentation using Mixed-Data Factor Analysis and Density-Based Clustering

This service is the data science core of the project, responsible for uncovering hidden store archetypes. It moves beyond simple heuristics by applying a sophisticated, two-stage unsupervised learning approach designed to handle the complexity of real-world retail data.

## Modeling Strategy & Rationale

The modeling pipeline is designed to be both statistically robust and pragmatically useful for business application.

### 1. Dimensionality Reduction: FAMD

*   **Challenge**: The feature set contains a mix of numerical data (e.g., revenue, size) and categorical data (e.g., store placement, region). Standard techniques like PCA are unsuitable, and simple one-hot encoding of categorical features can lead to an overly sparse, high-dimensional space (the "curse of dimensionality"), which degrades the performance of distance-based clustering algorithms.
*   **Solution**: **Factor Analysis of Mixed Data (FAMD)** is employed. It is a principal component method that generalizes PCA (for numerical data) and MCA (for categorical data), allowing for a unified and more nuanced reduction of the feature space. This approach effectively captures the underlying variance across all feature types.

### 2. Clustering: HDBSCAN

*   **Challenge**: Store performance and characteristics rarely fall into neat, spherical groups. K-Means and other centroid-based methods often fail to capture the complex, arbitrary shapes of real-world clusters and are highly sensitive to the pre-specified number of clusters, `k`.
*   **Solution**: **HDBSCAN** (Hierarchical Density-Based Spatial Clustering of Applications with Noise) is used. Its key advantages are:
    *   **No `k` Required**: It determines the number of clusters automatically based on data density.
    *   **Arbitrary Cluster Shapes**: It can identify clusters of non-spherical shapes.
    *   **Noise Identification**: It robustly identifies and labels outliers as "noise" points, preventing them from distorting the primary clusters. This is invaluable for identifying true anomalies.

### 3. Pragmatic Noise Handling

*   **Challenge**: While identifying noise is statistically valuable, business stakeholders often require every store to be assigned to a segment for operational purposes.
*   **Solution**: A `RandomForestClassifier` is trained on the core, high-confidence points identified by HDBSCAN. This model then predicts the most likely cluster for the "noise" points, providing a complete segmentation while maintaining a record of which assignments were inferred. This demonstrates a practical approach to bridging the gap between a pure statistical model and business needs.

### 4. Experiment Tracking

*   Every pipeline run is logged as an **MLflow** experiment. This captures:
    *   **Parameters**: `n_components` for FAMD, `min_cluster_size` for HDBSCAN, etc.
    *   **Metrics**: Key validation metrics like **DBCV** (for density-based clusters) and **Silhouette Score** (for final assignments) are logged to evaluate model quality.
    *   **Artifacts**: The trained FAMD, HDBSCAN, and classifier models are versioned and saved, along with the final clustered dataset.

## Tech Stack

*   **Python 3.11**
*   **Scikit-learn**: Preprocessing and classification.
*   **Prince**: FAMD implementation.
*   **HDBSCAN**: Clustering algorithm.
*   **MLflow**: Experiment tracking and model registry.

## Experimentation in Lab Mode

This service is designed for iterative development. You can easily experiment with model parameters by running the platform in "Lab Mode":
```bash
# From the project root
make lab
```
This starts a Jupyter server within this service's container, with all dependencies and data volumes mounted.

Navigate to the `notebooks/fine_tuning.ipynb` notebook to:
*   Run a hyperparameter grid search over FAMD and HDBSCAN.
*   Analyze key clustering metrics like DBCV and Silhouette Score.
*   Visualize the trade-offs to find the optimal parameters for the `config.py` file.

This hybrid approach allows for robust, production-style orchestration while providing the flexibility needed for data science experimentation.

## Output

*   **Artifacts**: Saved to `/opt/data/artifacts` and logged to MLflow.
    *   `final_clustered_stores.parquet`: The dataset with assigned clusters.
    *   `famd_model.joblib`
    *   `hdbscan_model.joblib`
    *   `noise_classifier_model.joblib`
    *   `behavioral_scaler.joblib`

## Design Choice: FAMD Scaling
We scale the FAMD components by the square root of their eigenvalues (singular values) before clustering. This preserves the relative importance of the components (variance explained) when calculating distances, rather than treating all dimensions equally.

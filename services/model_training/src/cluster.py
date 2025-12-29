import pandas as pd
import hdbscan
from sklearn.ensemble import RandomForestClassifier
from . import config
import logging

def run_clustering(X_scaled: pd.DataFrame, original_df: pd.DataFrame):
    """
    Runs HDBSCAN clustering and handles noise points using a classifier.
    """
    logging.info("--- Running Clustering (HDBSCAN) ---")
    
    # 1. Run HDBSCAN
    clusterer = hdbscan.HDBSCAN(**config.HDBSCAN_PARAMS)
    initial_labels = clusterer.fit_predict(X_scaled)
    
    df_clustered = original_df.copy()
    df_clustered['cluster'] = initial_labels
    
    noise_ratio = (df_clustered['cluster'] == -1).sum() / len(df_clustered) if len(df_clustered) > 0 else 0
    logging.info(f"HDBSCAN found {df_clustered['cluster'].nunique()} clusters with {noise_ratio:.2%} noise.")

    # 2. Noise Handling
    df_clustered['cluster_final'] = df_clustered['cluster']
    df_clustered['is_inferred'] = False
    noise_classifier = None
    
    noise_mask = df_clustered['cluster'] == -1
    if noise_mask.sum() > 0:
        logging.info(f"Classifying {noise_mask.sum()} noise points...")
        clean_mask = ~noise_mask
        
        # Train a classifier on the clean clusters
        noise_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        noise_classifier.fit(X_scaled[clean_mask], df_clustered.loc[clean_mask, 'cluster'])
        
        # Predict clusters for noise points
        predicted_clusters = noise_classifier.predict(X_scaled[noise_mask])
        df_clustered.loc[noise_mask, 'cluster_final'] = predicted_clusters
        df_clustered.loc[noise_mask, 'is_inferred'] = True

    return df_clustered, clusterer, noise_classifier

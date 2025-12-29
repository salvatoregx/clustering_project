import logging
import pandas as pd
import numpy as np
import prince
from sklearn.preprocessing import StandardScaler
from . import config

def select_and_transform_features(df: pd.DataFrame):
    """Selects features and applies transformations."""
    
    # Apply log transform to size_m2 to handle skew
    df_context = df[config.CONTEXT_COLS_CAT + config.CONTEXT_COLS_NUM].copy()
    df_context['size_m2'] = np.log1p(df_context['size_m2'])

    df_behavioral = df[config.BEHAVIORAL_COLS].copy()
    
    return df_context, df_behavioral

def run_preprocessing(df_context: pd.DataFrame, df_behavioral: pd.DataFrame):
    """
    Performs dimensionality reduction on contextual features and scales
    behavioral features.
    """
    logging.info("--- Running Preprocessing (FAMD & Scaling) ---")
    
    # 1. FAMD for contextual features
    famd = prince.FAMD(n_components=config.FAMD_COMPONENTS, random_state=42)
    df_reduced = famd.fit_transform(df_context)
    df_reduced.columns = [f"FAMD_{i+1}" for i in range(famd.n_components)]

    # 2. Scaling for behavioral features
    behavioral_scaler = StandardScaler()
    df_behavioral_scaled = pd.DataFrame(
        behavioral_scaler.fit_transform(df_behavioral),
        index=df_behavioral.index,
        columns=df_behavioral.columns
    )

    # 3. Combine for Clustering
    X_combined = pd.concat([df_reduced, df_behavioral_scaled], axis=1)
    
    # 4. Final scaling on the combined feature set
    final_scaler = StandardScaler()
    X_final_scaled = final_scaler.fit_transform(X_combined)

    return X_final_scaled, famd, behavioral_scaler, final_scaler

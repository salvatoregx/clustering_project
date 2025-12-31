import logging

import numpy as np
import pandas as pd
import prince
from sklearn.preprocessing import StandardScaler

from . import config


def select_and_transform_features(df: pd.DataFrame):
    """Selects features and applies transformations."""

    # Apply log transform to size_m2 to handle skew
    df_context = df[config.CONTEXT_COLS_CAT + config.CONTEXT_COLS_NUM].copy()
    df_context["size_m2"] = np.log1p(df_context["size_m2"])

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
    df_reduced.columns = [f"FAMD_{i + 1}" for i in range(famd.n_components)]

    # 2. Scale FAMD components to preserve relative variance
    # We scale all components by the standard deviation of the first component.
    # This makes FAMD_1 have a variance of 1, and all others < 1, preserving their relative importance.
    famd_scaling_factor = np.sqrt(famd.eigenvalues_[0])
    df_reduced_scaled = df_reduced / famd_scaling_factor
    logging.info(f"FAMD components scaled by a factor of {famd_scaling_factor:.4f}")

    # 3. Scaling for behavioral features
    behavioral_scaler = StandardScaler()
    df_behavioral_scaled = pd.DataFrame(
        behavioral_scaler.fit_transform(df_behavioral),
        index=df_behavioral.index,
        columns=df_behavioral.columns,
    )

    # 4. Combine scaled FAMD components with scaled behavioral features.
    X_final = pd.concat([df_reduced_scaled, df_behavioral_scaled], axis=1)

    return X_final.values, famd, behavioral_scaler

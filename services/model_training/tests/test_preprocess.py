import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import prince
from sklearn.preprocessing import StandardScaler

from src import config, preprocess


class TestPreprocess(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {
                "regional": ["R1", "R1", "R2", "R2", "R1", "R2"],
                "placement": ["Mall", "Street", "Mall", "Street", "Mall", "Street"],
                "anos_de_loja": [1.0, 5.0, 2.0, 4.0, 1.5, 3.5],
                "size_m2": [100.0, 50.0, 120.0, 60.0, 110.0, 55.0],
                "total_revenue": [1000.0, 500.0, 1200.0, 600.0, 1100.0, 550.0],
                "avg_ticket_item": [50.0, 30.0, 55.0, 35.0, 52.0, 32.0],
                "revenue_per_sqm": [10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
                "avg_inventory_depth": [100, 50, 120, 60, 110, 55],
                "unique_products_stocked": [20, 10, 25, 12, 22, 11],
            }
        )
        self.df.index.name = "store_id"

    def test_select_and_transform_features(self):
        """Tests correct feature selection and log transformation."""
        df_context, df_behavioral = preprocess.select_and_transform_features(self.df)

        self.assertCountEqual(
            df_context.columns, config.CONTEXT_COLS_CAT + config.CONTEXT_COLS_NUM
        )
        self.assertCountEqual(df_behavioral.columns, config.BEHAVIORAL_COLS)
        self.assertAlmostEqual(df_context["size_m2"].iloc[0], np.log1p(100.0), places=5)

    def test_run_preprocessing_integration(self):
        """Tests if preprocessing runs end-to-end with real objects, checking shapes and types."""
        df_context, df_behavioral = preprocess.select_and_transform_features(self.df)

        X_final, famd_model, scaler = preprocess.run_preprocessing(
            df_context, df_behavioral
        )

        self.assertIsInstance(X_final, np.ndarray)
        self.assertEqual(X_final.shape[0], len(self.df))
        expected_cols = config.FAMD_COMPONENTS + len(config.BEHAVIORAL_COLS)
        self.assertEqual(X_final.shape[1], expected_cols)
        self.assertIsInstance(famd_model, prince.FAMD)
        self.assertIsInstance(scaler, StandardScaler)

    @patch("src.preprocess.prince.FAMD")
    @patch("src.preprocess.StandardScaler")
    def test_run_preprocessing_mocked_logic(self, MockScaler, MockFAMD):
        """Tests the internal logic of preprocessing by mocking external libraries."""
        mock_famd_instance = MagicMock()
        mock_famd_instance.n_components = 3
        mock_famd_instance.eigenvalues_ = np.array([25, 9, 4])
        mock_famd_instance.fit_transform.return_value = pd.DataFrame(
            np.random.rand(6, 3)
        )
        MockFAMD.return_value = mock_famd_instance

        mock_scaler_instance = MagicMock()
        mock_scaler_instance.fit_transform.return_value = np.random.rand(
            6, len(config.BEHAVIORAL_COLS)
        )
        MockScaler.return_value = mock_scaler_instance

        df_context, df_behavioral = preprocess.select_and_transform_features(self.df)
        X_final, _, _ = preprocess.run_preprocessing(df_context, df_behavioral)

        MockFAMD.assert_called_once_with(
            n_components=config.FAMD_COMPONENTS, random_state=42
        )
        mock_famd_instance.fit_transform.assert_called_once()
        MockScaler.assert_called_once()
        mock_scaler_instance.fit_transform.assert_called_once()
        self.assertEqual(X_final.shape[1], 3 + len(config.BEHAVIORAL_COLS))

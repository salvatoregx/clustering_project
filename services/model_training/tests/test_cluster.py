import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from src import cluster, config

class TestCluster(unittest.TestCase):
    def setUp(self):
        """Create a deterministic dataset with 2 clear clusters and 2 noise points."""
        self.X_scaled = np.array(
            [
                [1, 1], [1.1, 1.1], [1.2, 1],  # Cluster 0
                [10, 10], [10.1, 10.1], [10.2, 10],  # Cluster 1
                [50, 50], [-10, -10],  # Noise
            ]
        )
        self.original_df = pd.DataFrame(index=[f"s{i}" for i in range(8)])

    @patch("src.cluster.hdbscan.HDBSCAN")
    @patch("src.cluster.RandomForestClassifier")
    def test_run_clustering_with_noise_classification(
        self, MockClassifier, MockHDBSCAN
    ):
        """Tests that noise points are correctly classified by a trained model."""
        mock_hdbscan_instance = MagicMock()
        # 8 points total
        initial_labels = np.array([0, 0, 0, 1, 1, 1, -1, -1])
        mock_hdbscan_instance.fit_predict.return_value = initial_labels
        MockHDBSCAN.return_value = mock_hdbscan_instance

        mock_classifier_instance = MagicMock()
        # Predict only for the 2 noise points
        mock_classifier_instance.predict.return_value = np.array([0, 1]) 
        MockClassifier.return_value = mock_classifier_instance

        df_clustered, _, noise_classifier = cluster.run_clustering(
            self.X_scaled, self.original_df
        )

        MockHDBSCAN.assert_called_with(**config.HDBSCAN_PARAMS)
        mock_hdbscan_instance.fit_predict.assert_called_with(self.X_scaled)

        MockClassifier.assert_called_once_with(n_estimators=100, random_state=42)
        fit_args, _ = mock_classifier_instance.fit.call_args
        X_train, y_train = fit_args
        self.assertEqual(len(X_train), 6) # 6 clean points
        
        predict_args, _ = mock_classifier_instance.predict.call_args
        (X_predict,) = predict_args
        self.assertEqual(len(X_predict), 2) # 2 noise points

        expected_final_labels = np.array([0, 0, 0, 1, 1, 1, 0, 1])
        np.testing.assert_array_equal(
            df_clustered["cluster_final"].values, expected_final_labels
        )
        expected_inferred = np.array([False, False, False, False, False, False, True, True])
        np.testing.assert_array_equal(
            df_clustered["is_inferred"].values, expected_inferred
        )

    @patch("src.cluster.hdbscan.HDBSCAN")
    @patch("src.cluster.RandomForestClassifier")
    def test_run_clustering_no_noise(self, MockClassifier, MockHDBSCAN):
        mock_hdbscan_instance = MagicMock()
        mock_hdbscan_instance.fit_predict.return_value = np.array(
            [0, 0, 0, 1, 1, 1, 0, 1]
        )
        MockHDBSCAN.return_value = mock_hdbscan_instance

        df_clustered, _, noise_classifier = cluster.run_clustering(
            self.X_scaled, self.original_df
        )

        MockClassifier.assert_not_called()
        self.assertIsNone(noise_classifier)
        
        pd.testing.assert_series_equal(
            df_clustered["cluster"], df_clustered["cluster_final"], check_names=False
        )
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from src import log, config


class TestLog(unittest.TestCase):
    @patch("joblib.dump")
    @patch("mlflow.log_param")
    @patch("mlflow.log_params")
    @patch("mlflow.log_metric")
    @patch("mlflow.log_metrics")
    @patch("mlflow.log_artifacts")
    @patch("hdbscan.validity.validity_index")
    @patch("sklearn.metrics.silhouette_score")
    def test_log_experiment(
        self,
        mock_silhouette,
        mock_dbcv,
        mock_log_artifacts,
        mock_log_metrics,
        mock_log_metric,
        mock_log_params,
        mock_log_param,
        mock_joblib_dump,
    ):
        """Tests that all relevant parameters, metrics, and artifacts are logged to MLflow."""
        # Arrange
        run_id = "test_run_123"
        mock_famd = MagicMock()
        mock_scalers = {"behavioral": MagicMock()}
        mock_clusterer = MagicMock()
        mock_noise_clf = MagicMock()
        X_scaled = np.random.rand(5, 5) 
        df_final = pd.DataFrame(
            {"cluster": [0, 0, 1, 1, -1], "cluster_final": [0, 0, 1, 1, 0]}
        )

        mock_dbcv.return_value = 0.8
        mock_silhouette.return_value = 0.7

        # Act
        log.log_experiment(
            run_id=run_id,
            famd_model=mock_famd,
            scalers=mock_scalers,
            clusterer=mock_clusterer,
            noise_clf=mock_noise_clf,
            X_scaled=X_scaled,
            df_final=df_final,
        )

        # Assert
        mock_log_param.assert_called_once_with(
            "famd_components", config.FAMD_COMPONENTS
        )
        mock_log_params.assert_called_once_with(config.HDBSCAN_PARAMS)
        mock_log_metrics.assert_called_once_with(
            {"initial_noise_ratio": 0.2, "dbcv_score": 0.8}
        )
        mock_log_metric.assert_called_once_with("final_silhouette_score", 0.7)
        self.assertEqual(mock_joblib_dump.call_count, 4)
        mock_log_artifacts.assert_called_once_with(
            config.ARTIFACT_PATH, artifact_path="results"
        )

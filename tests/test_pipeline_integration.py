import unittest
import os
import shutil
import sys
from pathlib import Path
from unittest.mock import patch

# This test assumes it is run from the project root directory.
# It adds the necessary service source paths to sys.path to allow imports.
SERVICE_ROOT = Path(__file__).parent.parent / 'services'
sys.path.insert(0, str(SERVICE_ROOT))

from data_gen.src import generate as data_gen_main
from data_gen.src import config as data_gen_config
from etl.src import main as etl_main
from etl.src import config as etl_config
from model_training.src import main as model_training_main
from model_training.src import config as model_training_config

class TestPipelineIntegration(unittest.TestCase):
    """
    An integration test suite to verify the end-to-end data flow
    between the data_gen, etl, and model_training services.
    """

    @classmethod
    def setUpClass(cls):
        """Set up a temporary directory for the test data across all tests."""
        cls.temp_dir = Path("./tmp_integration_test_data")
        if cls.temp_dir.exists():
            shutil.rmtree(cls.temp_dir)
        
        # Create subdirectories to mimic the real data volume structure
        cls.raw_path = cls.temp_dir / "raw"
        cls.processed_path = cls.temp_dir / "processed"
        cls.artifacts_path = cls.temp_dir / "artifacts"
        cls.validation_path = cls.temp_dir / "validation"
        
        cls.raw_path.mkdir(parents=True, exist_ok=True)
        cls.processed_path.mkdir(exist_ok=True)
        cls.artifacts_path.mkdir(exist_ok=True)
        cls.validation_path.mkdir(exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        """Clean up the temporary directory after all tests are done."""
        if cls.temp_dir.exists():
            shutil.rmtree(cls.temp_dir)

    @patch('model_training.src.main.mlflow')
    def test_full_pipeline_execution(self, mock_mlflow):
        """Tests the sequential execution of the data generation, ETL, and model training scripts."""
        data_gen_config.OUTPUT_DIR = str(self.raw_path)
        data_gen_config.NUM_STORES, data_gen_config.NUM_PRODUCTS = 10, 20
        
        etl_config.RAW_DATA_PATH, etl_config.PROCESSED_DATA_PATH, etl_config.VALIDATION_PATH = str(self.raw_path), str(self.processed_path), str(self.validation_path)
        etl_config.FEATURES_PATH = str(self.processed_path / "store_features.parquet")
        
        model_training_config.PROCESSED_DATA_PATH = str(self.processed_path / "store_features.parquet")
        model_training_config.ARTIFACT_PATH = str(self.artifacts_path)

        data_gen_main.main()
        self.assertTrue((self.raw_path / "dim_product.parquet").exists())

        etl_main.main()
        self.assertTrue((self.processed_path / "store_features.parquet").exists())
        self.assertGreater(len(list(self.validation_path.glob('*.json'))), 0)

        model_training_main.main()
        self.assertTrue((self.artifacts_path / "famd_model.joblib").exists())
        self.assertTrue((self.artifacts_path / "final_clustered_stores.parquet").exists())
        mock_mlflow.log_artifacts.assert_called_with(str(self.artifacts_path), artifact_path="results")
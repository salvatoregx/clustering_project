import unittest
import os
import subprocess
import tempfile
import shutil
from pathlib import Path

# This test runs the services as containers, respecting their isolated dependencies.
# It requires podman/docker and podman-compose/docker-compose to be installed.

COMPOSE_CMD = os.environ.get("COMPOSE", "podman-compose")
TEST_COMPOSE_FILE = Path(__file__).parent.parent / "compose.test.yaml"


@unittest.skipIf(
    not shutil.which(COMPOSE_CMD.split(" ")[0]),
    f"'{COMPOSE_CMD}' not found. Skipping containerized integration tests.",
)
class TestContainerizedPipelineIntegration(unittest.TestCase):
    """
    Tests the end-to-end pipeline by running each service in its own container,
    verifying the data contracts between them via a shared volume.
    """

    temp_dir = None
    env = None

    @classmethod
    def setUpClass(cls):
        """Builds all service images and sets up a temporary directory for data."""
        print("\nBuilding service images for integration test...")
        subprocess.run(
            [COMPOSE_CMD, "-f", str(TEST_COMPOSE_FILE), "build"],
            check=True,
            capture_output=True,
        )

        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.data_path = Path(cls.temp_dir.name)

        cls.env = os.environ.copy()
        cls.env["DATA_PATH"] = str(cls.data_path)

    @classmethod
    def tearDownClass(cls):
        """Cleans up the temporary directory and stops all test containers."""
        if cls.temp_dir:
            cls.temp_dir.cleanup()

        print("\nCleaning up test containers...")
        subprocess.run(
            [COMPOSE_CMD, "-f", str(TEST_COMPOSE_FILE), "down", "--volumes"],
            check=True,
            capture_output=True,
        )

    def test_01_data_gen_service(self):
        """Tests the data_gen service container."""
        print("\n--- Running data_gen container ---")
        result = subprocess.run(
            [COMPOSE_CMD, "-f", str(TEST_COMPOSE_FILE), "run", "--rm", "data_gen"],
            check=True,
            capture_output=True,
            text=True,
            env=self.env,
        )
        self.assertIn("Generation Complete.", result.stdout)
        self.assertTrue((self.data_path / "raw" / "dim_product.parquet").exists())
        self.assertTrue((self.data_path / "raw" / "fact_sales").is_dir())

    def test_02_etl_service(self):
        """Tests the etl service container, consuming data_gen's output."""
        if not (self.data_path / "raw" / "dim_product.parquet").exists():
            self.skipTest("Skipping ETL test: data_gen output is missing.")

        print("\n--- Running etl container ---")
        result = subprocess.run(
            [COMPOSE_CMD, "-f", str(TEST_COMPOSE_FILE), "run", "--rm", "etl"],
            check=True,
            capture_output=True,
            text=True,
            env=self.env,
        )
        self.assertIn("Writing", result.stdout)
        self.assertTrue(
            (self.data_path / "processed" / "store_features.parquet").exists()
        )
        self.assertGreater(len(list((self.data_path / "validation").glob("*.json"))), 0)

    def test_03_model_training_service(self):
        """Tests the model_training service, consuming etl's output."""
        if not (self.data_path / "processed" / "store_features.parquet").exists():
            self.skipTest("Skipping model_training test: etl output is missing.")

        subprocess.run(
            [COMPOSE_CMD, "-f", str(TEST_COMPOSE_FILE), "up", "-d", "mlflow"],
            check=True,
            env=self.env,
        )
        result = subprocess.run(
            [
                COMPOSE_CMD,
                "-f",
                str(TEST_COMPOSE_FILE),
                "run",
                "--rm",
                "model_training",
            ],
            check=True,
            capture_output=True,
            text=True,
            env=self.env,
        )
        self.assertIn("Clustering complete.", result.stdout)
        self.assertTrue((self.data_path / "artifacts" / "famd_model.joblib").exists())
        self.assertGreater(len(list((self.data_path / "mlflow").iterdir())), 0)

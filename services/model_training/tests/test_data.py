import unittest
from unittest.mock import patch
import pandas as pd
from src import data, config


class TestData(unittest.TestCase):
    @patch("pandas.read_parquet")
    def test_load_and_prepare_data(self, mock_read_parquet):
        """Tests that data is loaded and 'anos_de_loja' is engineered correctly."""
        # Arrange
        mock_df = pd.DataFrame(
            {"store_id": ["s1", "s2"], "opening_date": ["2022-01-01", "2023-01-01"]}
        )
        mock_read_parquet.return_value = mock_df

        # Act
        df_result = data.load_and_prepare_data()

        # Assert
        mock_read_parquet.assert_called_once_with(config.PROCESSED_DATA_PATH)
        self.assertIn("anos_de_loja", df_result.columns)
        self.assertTrue(pd.api.types.is_float_dtype(df_result["anos_de_loja"]))
        self.assertIn("s1", df_result.index)
        self.assertGreater(df_result.loc["s1", "anos_de_loja"], 2)
        self.assertLess(df_result.loc["s2", "anos_de_loja"], 2)

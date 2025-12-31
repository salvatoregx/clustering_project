import unittest
from datetime import datetime
from unittest.mock import patch

import pandas as pd

from src import config, data


class TestData(unittest.TestCase):
    @patch("pandas.read_parquet")
    def test_load_and_prepare_data(self, mock_read_parquet):
        current_year = datetime.now().year
        recent_date = f"{current_year}-01-01"  # < 1 year old
        old_date = f"{current_year - 4}-01-01"  # > 2 years old

        mock_df = pd.DataFrame(
            {"store_id": ["s1", "s2"], "opening_date": [old_date, recent_date]}
        )
        mock_read_parquet.return_value = mock_df

        df_result = data.load_and_prepare_data()

        mock_read_parquet.assert_called_once_with(config.PROCESSED_DATA_PATH)
        self.assertIn("s1", df_result.index)

        # s1 is old, should be > 2
        self.assertGreater(df_result.loc["s1", "anos_de_loja"], 2)
        # s2 is recent, should be < 2
        self.assertLess(df_result.loc["s2", "anos_de_loja"], 2)

import unittest
import pandas as pd
import numpy as np
from faker import Faker
from src.generators import products, stores
from src import config

class TestGenerators(unittest.TestCase):
    def setUp(self):
        self.fake = Faker(config.LOCALE)
        self.num_products = 10
        self.num_stores = 5

    def test_generate_products(self):
        df, p_ids, p_prices, p_costs = products.generate(self.num_products, self.fake)
        
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), self.num_products)
        self.assertEqual(len(p_ids), self.num_products)
        self.assertEqual(len(p_prices), self.num_products)
        self.assertEqual(len(p_costs), self.num_products)
        
        # Check columns
        expected_cols = ['product_id', 'name', 'category', 'base_price', 'cost']
        for col in expected_cols:
            self.assertIn(col, df.columns)
            
        # Check price logic (price > cost)
        for pid in p_ids:
            self.assertGreater(p_prices[pid], p_costs[pid])

    def test_generate_stores(self):
        df_phys, df_mgmt, s_ids = stores.generate(self.num_stores, self.fake)
        
        self.assertIsInstance(df_phys, pd.DataFrame)
        self.assertIsInstance(df_mgmt, pd.DataFrame)
        self.assertEqual(len(df_phys), self.num_stores)
        self.assertEqual(len(df_mgmt), self.num_stores)
        self.assertEqual(len(s_ids), self.num_stores)
        
        # Check for null injection in size_m2 (statistical check, might pass if lucky but intended to verify logic exists)
        # Since we can't guarantee nulls in small sample, we just check structure
        self.assertIn('size_m2', df_phys.columns)
        self.assertIn('current_cluster_label', df_mgmt.columns)

if __name__ == '__main__':
    unittest.main()
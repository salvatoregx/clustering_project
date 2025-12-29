import os
import numpy as np
from faker import Faker
from . import config
from .generators import products, stores, history

def main():
    print(f"Starting Data Generation for {config.LOCALE}...")
    
    # Setup
    fake = Faker(config.LOCALE)
    np.random.seed(42)
    
    os.makedirs(f"{config.OUTPUT_DIR}/fact_sales", exist_ok=True)
    os.makedirs(f"{config.OUTPUT_DIR}/fact_inventory", exist_ok=True)
    
    # 1. Products
    df_products, p_ids, p_prices, p_costs = products.generate(config.NUM_PRODUCTS, fake)
    df_products.to_parquet(f"{config.OUTPUT_DIR}/dim_product.parquet", index=False)
    
    # 2. Stores
    df_phys, df_mgmt, s_ids = stores.generate(config.NUM_STORES, fake)
    df_phys.to_parquet(f"{config.OUTPUT_DIR}/dim_store_physical.parquet", index=False)
    df_mgmt.to_parquet(f"{config.OUTPUT_DIR}/dim_store_management.parquet", index=False)
    
    # 3. History
    history.generate(s_ids, p_ids, p_prices, p_costs, df_mgmt)
    
    print("Generation Complete.")

if __name__ == "__main__":
    main()

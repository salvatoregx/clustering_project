import logging
import os
import shutil

import numpy as np
from faker import Faker

from . import config
from .generators import history, products, stores

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def main():
    logging.info(f"Starting Data Generation for {config.LOCALE}...")

    # Setup
    fake = Faker(config.LOCALE)
    np.random.seed(42)

    # Cleanup existing data to ensure idempotency
    if os.path.exists(config.OUTPUT_DIR):
        logging.info(f"Cleaning up existing data at {config.OUTPUT_DIR}...")
        shutil.rmtree(config.OUTPUT_DIR)

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

    logging.info("Generation Complete.")


if __name__ == "__main__":
    main()

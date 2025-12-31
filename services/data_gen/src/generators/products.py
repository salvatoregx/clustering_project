import logging

import numpy as np
import pandas as pd
from faker import Faker


def generate(num_products: int, fake: Faker):
    """Generates Product Dimension and returns metadata needed for history generation."""
    logging.info("Generating Products...")
    product_ids = [fake.uuid4() for _ in range(num_products)]
    categories = ["Camisetas", "Calças", "Vestidos", "Acessórios", "Casacos"]

    data = []
    product_prices = {}
    product_costs = {}

    for pid in product_ids:
        cat = np.random.choice(categories)
        price_base = np.random.uniform(30, 200)
        cost_factor = np.random.uniform(0.2, 0.6)

        # Store metadata for history generation
        product_prices[pid] = round(price_base, 2)
        product_costs[pid] = round(price_base * cost_factor, 2)

        data.append(
            {
                "product_id": pid,
                "name": f"{cat} {fake.color_name()} {fake.word()}",
                "category": cat,
                "base_price": product_prices[pid],
                "cost": product_costs[pid],
            }
        )

    df = pd.DataFrame(data)
    return df, product_ids, product_prices, product_costs

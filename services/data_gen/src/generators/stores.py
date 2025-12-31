import pandas as pd
import numpy as np
from faker import Faker
from .. import config
import logging


def generate(num_stores: int, fake: Faker):
    """Generates Physical and Management Store Dimensions with Geo-Coherence"""
    logging.info("Generating Stores with Geographic Coherence...")
    store_ids = [fake.uuid4() for _ in range(num_stores)]

    placements = ["Shopping Mall", "Street", "Commercial Bldg", "In-Company"]
    cluster_probs = [0.80, 0.15, 0.05]
    cluster_labels = ["C", "B", "A"]
    physical_data = []
    mgmt_data = []

    regionals_list = list(config.REGIONAL_GEO_MAP.keys())

    for sid in store_ids:
        # 1. Geographic Hierarchy Selection (Top-Down)
        regional = np.random.choice(regionals_list)
        available_states = list(config.REGIONAL_GEO_MAP[regional].keys())
        uf = np.random.choice(available_states)
        city = np.random.choice(config.REGIONAL_GEO_MAP[regional][uf])

        # 2. Physical Characteristics
        placement = np.random.choice(placements, p=[0.6, 0.2, 0.1, 0.1])

        if placement == "In-Company":
            size = np.random.normal(30, 5)
        elif placement == "Shopping Mall":
            size = np.random.normal(150, 40)
        else:
            size = np.random.normal(80, 20)

        # Introduce some nulls for the ETL to handle
        if np.random.rand() < 0.05:
            size_m2 = np.nan
        else:
            size_m2 = round(size, 2)

        storage = size * np.random.uniform(0.1, 0.3)
        street_name = fake.street_name()
        street_number = fake.building_number()
        full_address = f"{street_name}, {street_number} - {city} - {uf}"

        physical_data.append(
            {
                "store_id": sid,
                "size_m2": size_m2,
                "placement": placement,
                "storage_capacity_m2": round(storage, 2),
                "address": full_address,
                "city": city,
                "state": uf,
            }
        )

        # 3. Management Characteristics
        cluster = np.random.choice(cluster_labels, p=cluster_probs)

        mgmt_data.append(
            {
                "store_id": sid,
                "store_name": f"YouCom {city} {street_number} - {fake.word()}",
                "regional": regional,
                "current_cluster_label": cluster,
                "opening_date": fake.date_between(start_date="-5y", end_date="-1m"),
            }
        )

    return pd.DataFrame(physical_data), pd.DataFrame(mgmt_data), store_ids

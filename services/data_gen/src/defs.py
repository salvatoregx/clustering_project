from dagster import asset, Definitions
from . import generate

@asset(group_name="raw_layer")
def raw_data_parquet():
    """Generates synthetic retail data."""
    generate.main()
    return "Data generated in /opt/data/raw"

defs = Definitions(assets=[raw_data_parquet])

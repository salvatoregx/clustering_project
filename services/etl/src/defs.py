from dagster import asset, Definitions, AssetKey, SourceAsset, AutoMaterializePolicy
from . import main as etl_main

# Reference the asset from the data_gen service
raw_data_asset = SourceAsset(key=AssetKey("raw_data_parquet"))


@asset(
    group_name="processed_layer",
    deps=[raw_data_asset],
    auto_materialize_policy=AutoMaterializePolicy.eager(),
)
def store_features_parquet():
    """Aggregates raw data into store features using PySpark."""
    etl_main.main()
    return "Features stored in /opt/data/processed"


defs = Definitions(assets=[store_features_parquet])

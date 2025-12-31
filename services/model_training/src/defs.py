from dagster import (
    AssetKey,
    AutoMaterializePolicy,
    Definitions,
    MetadataValue,
    SourceAsset,
    asset,
)

from . import main as model_main

# Reference the asset from the etl service
store_features_asset = SourceAsset(key=AssetKey("store_features_parquet"))


@asset(
    group_name="model_layer",
    deps=[store_features_asset],
    metadata={
        "dashboard_url": MetadataValue.url("http://localhost:8501"),
    },
    auto_materialize_policy=AutoMaterializePolicy.eager(),
)
def clustering_model():
    """Trains the clustering model and logs to MLflow."""
    model_main.main()
    return "Model artifacts saved"


defs = Definitions(assets=[clustering_model])

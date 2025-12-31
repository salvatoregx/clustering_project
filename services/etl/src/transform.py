import os
from pyspark.sql import SparkSession, DataFrame
import logging
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression

from . import config
from . import validation


def _load_data(spark: SparkSession) -> dict[str, DataFrame]:
    """Loads all raw data sources from Parquet files."""
    logging.info("--- Loading Raw Data ---")
    return {
        "dim_product": spark.read.parquet(
            os.path.join(config.RAW_DATA_PATH, "dim_product.parquet")
        ),
        "dim_store_phys": spark.read.parquet(
            os.path.join(config.RAW_DATA_PATH, "dim_store_physical.parquet")
        ),
        "dim_store_mgmt": spark.read.parquet(
            os.path.join(config.RAW_DATA_PATH, "dim_store_management.parquet")
        ),
        "fact_sales": spark.read.parquet(
            os.path.join(config.RAW_DATA_PATH, "fact_sales")
        ),
        "fact_inventory": spark.read.parquet(
            os.path.join(config.RAW_DATA_PATH, "fact_inventory")
        ),
    }


def _aggregate_features(data: dict[str, DataFrame]) -> DataFrame:
    """Performs aggregations to create store-level features."""
    logging.info("--- Aggregating Features ---")

    # Unpack dataframes
    fact_sales = data["fact_sales"]
    dim_product = data["dim_product"]
    fact_inventory = data["fact_inventory"]

    # A. Sales Metrics
    sales_enriched = fact_sales.join(dim_product, "product_id")

    cat_pivot = (
        sales_enriched.groupBy("store_id")
        .pivot("category")
        .agg(F.sum(F.col("qty") * F.col("unit_price_at_sale")))
        .fillna(0)
    )

    for col_name in cat_pivot.columns:
        if col_name != "store_id":
            cat_pivot = cat_pivot.withColumnRenamed(col_name, f"rev_{col_name.lower()}")

    sales_aggs = sales_enriched.groupBy("store_id").agg(
        F.sum(F.col("qty") * F.col("unit_price_at_sale")).alias("total_revenue"),
        F.sum("qty").alias("total_items_sold"),
        F.countDistinct("date").alias("days_active"),
        F.mean("unit_price_at_sale").alias("avg_ticket_item"),
    )

    # B. Inventory Metrics
    inv_aggs = fact_inventory.groupBy("store_id").agg(
        F.mean("qty_on_hand").alias("avg_inventory_depth"),
        F.countDistinct("product_id").alias("unique_products_stocked"),
    )

    # C. Join all aggregated features to store dimensions
    store_features = (
        data["dim_store_mgmt"]
        .join(data["dim_store_phys"], "store_id")
        .join(sales_aggs, "store_id", "left")
        .join(cat_pivot, "store_id", "left")
        .join(inv_aggs, "store_id", "left")
        .fillna(0)
    )

    return store_features


def _engineer_features(df: DataFrame) -> DataFrame:
    """Cleans data and engineers new features."""
    logging.info("--- Engineering and Cleaning Features ---")

    # A. Sophisticated Imputation: Linear Regression
    # Predict size_m2 based on storage_capacity_m2 (highly correlated)

    # 1. Prepare Features
    assembler = VectorAssembler(inputCols=["storage_capacity_m2"], outputCol="features")
    df_vec = assembler.transform(df)

    # 2. Train Model on known data (size_m2 > 0)
    train_data = df_vec.filter(F.col("size_m2") > 0)

    lr = LinearRegression(featuresCol="features", labelCol="size_m2")
    model = lr.fit(train_data)

    # 3. Predict and Impute
    predictions = model.transform(df_vec)
    df_imputed = predictions.withColumn(
        "size_m2_imputed",
        F.when(F.col("size_m2") > 0, F.col("size_m2")).otherwise(
            F.abs(F.col("prediction"))
        ),
    ).drop("features", "prediction")

    # B. Create new features using the imputed value
    df_engineered = df_imputed.withColumn(
        "revenue_per_sqm",
        F.when(
            F.col("size_m2_imputed") > 0,
            F.col("total_revenue") / F.col("size_m2_imputed"),
        ).otherwise(0),
    )

    # C. Clean up columns for final output
    final_df = df_engineered.drop("size_m2").withColumnRenamed(
        "size_m2_imputed", "size_m2"
    )

    return final_df


def run_transformations(spark: SparkSession):
    """Orchestrates the full transformation process."""
    data = _load_data(spark)
    validation.validate_sales_data(data["fact_sales"])
    store_features_agg = _aggregate_features(data)
    df_final = _engineer_features(store_features_agg)

    return df_final

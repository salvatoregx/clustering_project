from pyspark.sql import SparkSession


def get_spark_session():
    """Creates and returns a SparkSession for the ETL pipeline."""
    return (
        SparkSession.builder.appName("RetailClusteringETL")
        .config("spark.sql.parquet.compression.codec", "snappy")
        .getOrCreate()
    )

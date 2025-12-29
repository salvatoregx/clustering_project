"""
ETL Pipeline for Retail Clustering Project.

This is the main entry point for the ETL service. It orchestrates the
pipeline by initializing a Spark session, running the transformations,
and then stopping the session.
"""
import os
import logging

from . import config
from . import spark
from . import transform

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """Main execution function for the ETL pipeline."""
    spark_session = None
    try:
        spark_session = spark.get_spark_session()
        df_final = transform.run_transformations(spark_session)
        
        # Write Output only if data exists
        if df_final.count() > 0:
            logging.info(f"--- Writing {df_final.count()} rows of Processed Data to {config.FEATURES_PATH} ---")
            df_final.coalesce(1).write.mode("overwrite").parquet(config.FEATURES_PATH)
        else:
            logging.warning("--- No data to write after transformations. Output file will not be created. ---")
    finally:
        if spark_session:
            logging.info("--- Stopping Spark Session ---")
            spark_session.stop()

if __name__ == "__main__":
    main()

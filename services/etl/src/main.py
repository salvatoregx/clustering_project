"""
ETL Pipeline for Retail Clustering Project.

This is the main entry point for the ETL service. It orchestrates the
pipeline by initializing a Spark session, running the transformations,
and then stopping the session.
"""
import os

from . import config
from . import spark
from . import transform

def main():
    """Main execution function for the ETL pipeline."""
    spark_session = None
    try:
        spark_session = spark.get_spark_session()
        df_final = transform.run_transformations(spark_session)
        
        # Write Output
        print(f"--- Writing Processed Data to {config.FEATURES_PATH} ---")
        df_final.coalesce(1).write.mode("overwrite").parquet(config.FEATURES_PATH)
    finally:
        if spark_session:
            print("--- Stopping Spark Session ---")
            spark_session.stop()

if __name__ == "__main__":
    main()


import great_expectations as gx
from pyspark.sql import DataFrame
import logging

def validate_sales_data(df_sales: DataFrame):
    """
    Validates raw sales data using Great Expectations (Fluent API).
    Ensures data integrity before heavy aggregation.
    """
    logging.info("--- Starting Data Validation ---")
    # Using project_root_dir is good practice in containerized/production environments
    context = gx.get_context(project_root_dir='/app')
    
    # Define Data Source & Asset
    datasource = context.sources.add_spark("sales_source")
    asset = datasource.add_dataframe_asset("sales_asset", dataframe=df_sales)
    
    # Create Expectation Suite
    suite = context.add_or_update_expectation_suite("sales_validation_suite")
    
    # Get Validator
    validator = context.get_validator(
        batch_request=asset.build_batch_request(),
        expectation_suite=suite
    )
    
    # Define Expectations
    validator.expect_column_values_to_be_between("qty", min_value=1)
    validator.expect_column_values_to_be_between("unit_price_at_sale", min_value=0.0)
    validator.expect_column_values_to_not_be_null("store_id")
    validator.expect_column_values_to_not_be_null("product_id")
    
    # Run Validation
    results = validator.validate()
    
    if not results["success"]:
        logging.warning(f"WARNING: Data Validation Failed! Success: {results['success']}")
        # In a strict pipeline, one might raise an exception here.
    else:
        logging.info("Data Validation Passed.")
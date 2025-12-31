import json
import logging
import os
from datetime import datetime

import great_expectations as gx
from pyspark.sql import DataFrame

from . import config


def validate_sales_data(df_sales: DataFrame):
    """
    Validates raw sales data using Great Expectations (Fluent API).
    Ensures data integrity before heavy aggregation.
    """
    logging.info("--- Starting Data Validation ---")
    # Using project_root_dir is good practice in containerized/production environments
    context = gx.get_context(mode="ephemeral")

    # Define Data Source, Asset and Batch Definition
    datasource = context.data_sources.add_or_update_spark("sales_source")
    asset = datasource.add_dataframe_asset("sales_asset")
    batch_definition = asset.add_batch_definition_whole_dataframe("sales_batch_def")

    # Create Expectation Suite
    suite_name = "sales_validation_suite"
    suite = context.suites.add(gx.ExpectationSuite(name=suite_name))

    # Build Batch Request passing the DataFrame
    batch_request = batch_definition.build_batch_request(
        batch_parameters={"dataframe": df_sales}
    )

    # Get Validator
    validator = context.get_validator(
        batch_request=batch_request, expectation_suite=suite
    )

    # Define Expectations
    validator.expect_column_values_to_be_between("qty", min_value=1)
    validator.expect_column_values_to_be_between("unit_price_at_sale", min_value=0.0)
    validator.expect_column_values_to_not_be_null("store_id")
    validator.expect_column_values_to_not_be_null("product_id")

    # Run Validation
    results = validator.validate()

    # Save Results for Monitoring Dashboard
    os.makedirs(config.VALIDATION_PATH, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(
        config.VALIDATION_PATH, f"sales_validation_{timestamp}.json"
    )
    with open(output_file, "w") as f:
        json.dump(results.to_json_dict(), f, indent=2)

    if not results["success"]:
        logging.warning(
            f"WARNING: Data Validation Failed! Success: {results['success']}"
        )
        # In a strict pipeline, one might raise an exception here.
    else:
        logging.info("Data Validation Passed.")

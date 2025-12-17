"""
Utilities for working with Delta tables.
"""

from typing import Dict, Any, List, Optional


def create_delta_table(
    results: List[Dict[str, Any]],
    table_name: str = "engineering_drawing_metadata",
    catalog: Optional[str] = None,
    schema: Optional[str] = None,
) -> None:
    """
    Create a Delta table from processing results.
    
    Args:
        results: List of result dictionaries from process_folder
        table_name: Name of the Delta table
        catalog: Unity Catalog name (optional)
        schema: Schema/database name (optional)
    """
    from pyspark.sql import SparkSession
    from pyspark.sql.types import StructType, StructField, StringType, IntegerType
    
    spark = SparkSession.builder.getOrCreate()
    
    # Define schema
    table_schema = StructType([
        StructField("pdf_path", StringType(), True),
        StructField("pdf_filename", StringType(), True),
        StructField("page_index", IntegerType(), True),
        StructField("total_pages", IntegerType(), True),
        StructField("status", StringType(), True),
        StructField("error_message", StringType(), True),
        StructField("rotation_applied", IntegerType(), True),
        StructField("region", StringType(), True),
        StructField("contractor", StringType(), True),
        StructField("project_title", StringType(), True),
        StructField("drawing_title", StringType(), True),
        StructField("unit", StringType(), True),
        StructField("plant", StringType(), True),
        StructField("contr_proj_no", StringType(), True),
        StructField("agnoc_gas_dwg_no", StringType(), True),
        StructField("revision_history", StringType(), True),
        StructField("latest_rev", StringType(), True),
        StructField("latest_rev_date", StringType(), True),
        StructField("latest_rev_description", StringType(), True),
        StructField("revision_count", IntegerType(), True),
        StructField("extraction_errors", StringType(), True),
        StructField("confidence", StringType(), True),
        StructField("original_size", StringType(), True),
        StructField("crop_size", StringType(), True),
    ])
    
    # Create DataFrame
    df = spark.createDataFrame(results, schema=table_schema)
    
    # Build full table name
    if catalog and schema:
        full_table_name = f"{catalog}.{schema}.{table_name}"
    elif schema:
        full_table_name = f"{schema}.{table_name}"
    else:
        full_table_name = table_name
    
    # Write to Delta table
    print(f"\nWriting {len(results)} records to Delta table: {full_table_name}")
    
    df.write \
        .format("delta") \
        .mode("overwrite") \
        .option("overwriteSchema", "true") \
        .saveAsTable(full_table_name)
    
    print(f"Delta table created successfully!")
    print(f"\nTable preview:")
    display(spark.table(full_table_name).limit(10))


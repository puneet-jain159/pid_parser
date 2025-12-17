# Databricks notebook source
# MAGIC %md
# MAGIC # Engineering Drawing OCR Pipeline
# MAGIC This notebook extracts metadata from engineering drawings (P&ID, technical drawings) using Claude vision via Databricks FMAPI.
# MAGIC
# MAGIC ## Architecture
# MAGIC 1. PDF → Full image
# MAGIC 2. Detect rotation on FULL PAGE first
# MAGIC 3. Rotate full image to normalize orientation
# MAGIC 4. Layout detection on rotated image (coordinates now consistent)
# MAGIC 5. Crop from rotated image
# MAGIC 6. OCR on crop

# COMMAND ----------

# MAGIC %pip install pymupdf pillow openai mlflow>=2.20

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup and Configuration

# COMMAND ----------

import sys
import os

# Add the project root to Python path for imports
# Import config to get validated configuration
from config import get_config, get_databricks_client

# Load configuration
cfg = get_config()

# Add project root to path
if cfg.project.project_root not in sys.path:
    sys.path.insert(0, cfg.project.project_root)

# COMMAND ----------

from openai import OpenAI
import mlflow
from mlflow.entities import SpanType

from pid_parser.ocr_pipeline import (
    process_engineering_drawing,
    process_single_drawing_safe,
    process_folder,
)
from pid_parser.delta_utils import create_delta_table

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

# Get Databricks host and token from notebook context
DATABRICKS_HOST, DATABRICKS_TOKEN = get_databricks_client(dbutils)

# OpenAI-compatible client pointing at Databricks FMAPI
fm_client = OpenAI(
    api_key=DATABRICKS_TOKEN,
    base_url=f"{DATABRICKS_HOST}/serving-endpoints",
)

# MLflow Tracing Setup
mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
mlflow.set_experiment(cfg.mlflow.experiment_name_ocr)
mlflow.openai.autolog()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Single File Processing (for testing)

# COMMAND ----------

# # Test with a single file
# # Uncomment and update test file paths in config.json
# # if cfg.ocr_pipeline.test_files.test_pdf and cfg.ocr_pipeline.test_files.test_output:
# #     result = process_engineering_drawing(
# #         pdf_path=cfg.ocr_pipeline.test_files.test_pdf,
# #         fm_client=fm_client,
# #         ocr_model=cfg.models.claude_fmapi_model,
# #         layout_model=cfg.models.layout_model,
# #         page_index=0,
# #         output_path=cfg.ocr_pipeline.test_files.test_output,
# #     )
# #     import json
# #     print(json.dumps(result["metadata"], indent=2))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Batch Processing

# COMMAND ----------

# Configuration values are loaded from config.json
# Update config.json to change these values

# COMMAND ----------

# Run batch processing with MLflow tracing
# All traces from this run will be grouped together in the MLflow UI
with mlflow.start_run(run_name=cfg.mlflow.run_name) as run:
    # Log parameters
    mlflow.log_params({
        "input_folder": cfg.ocr_pipeline.input_folder,
        "output_dir": cfg.ocr_pipeline.output_dir,
        "dpi": cfg.ocr_pipeline.processing.dpi,
        "force_rotation": str(cfg.ocr_pipeline.processing.force_rotation),
    })
    
    # Run batch processing (processes LAST page of each PDF)
    results = process_folder(
        input_folder=cfg.ocr_pipeline.input_folder,
        fm_client=fm_client,
        ocr_model=cfg.models.claude_fmapi_model,
        layout_model=cfg.models.layout_model,
        output_dir=cfg.ocr_pipeline.output_dir,
        dpi=cfg.ocr_pipeline.processing.dpi,
        force_rotation=cfg.ocr_pipeline.processing.force_rotation,
    )
    
    # Log metrics
    successful = sum(1 for r in results if r.get("status") == "success")
    failed = len(results) - successful
    mlflow.log_metrics({
        "total_drawings": len(results),
        "successful_extractions": successful,
        "failed_extractions": failed,
        "success_rate": successful / len(results) if results else 0,
    })
    
    print(f"\nMLflow Run ID: {run.info.run_id}")

# COMMAND ----------

results

# COMMAND ----------

# Create Delta table
create_delta_table(
    results=results,
    table_name=cfg.ocr_pipeline.delta_table.table_name,
    catalog=cfg.ocr_pipeline.delta_table.catalog,
    schema=cfg.ocr_pipeline.delta_table.schema,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Query the Results

# COMMAND ----------

from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()

# COMMAND ----------

# Query successful extractions
spark.sql(f"""
    SELECT 
        pdf_filename,
        page_index,
        drawing_title,
        agnoc_gas_dwg_no,
        unit,
        plant,
        latest_rev,
        latest_rev_date,
        latest_rev_description,
        revision_count,
        confidence
    FROM {cfg.ocr_pipeline.delta_table.catalog}.{cfg.ocr_pipeline.delta_table.schema}.{cfg.ocr_pipeline.delta_table.table_name}
    WHERE status = 'success'
    ORDER BY pdf_filename, page_index
""").display()

# COMMAND ----------

# Query errors
spark.sql(f"""
    SELECT 
        pdf_filename,
        page_index,
        status,
        error_message
    FROM {cfg.ocr_pipeline.delta_table.catalog}.{cfg.ocr_pipeline.delta_table.schema}.{cfg.ocr_pipeline.delta_table.table_name}
    WHERE status != 'success'
""").display()

# COMMAND ----------

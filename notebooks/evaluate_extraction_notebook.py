# Databricks notebook source
# MAGIC %md
# MAGIC # Engineering Drawing OCR - LLM-as-Judge Evaluation
# MAGIC 
# MAGIC This notebook evaluates the OCR extraction pipeline using an LLM as a judge.
# MAGIC All assessments are logged directly to MLflow traces.
# MAGIC 
# MAGIC ## Inputs
# MAGIC - **Predictions**: Delta table `engineering_drawing_metadata` - Results from OCR pipeline
# MAGIC - **Ground Truth**: `Labelled Evaluation Data_11-12-25.csv` - Human-labelled data (CSV format)
# MAGIC 
# MAGIC ## Evaluation Method
# MAGIC - LLM-as-Judge using Claude to assess extraction quality
# MAGIC - Assessments logged to MLflow traces for each record

# COMMAND ----------

# MAGIC %pip install mlflow>=2.20 pandas openai

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup and Imports

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

import pandas as pd
import numpy as np
from openai import OpenAI
import mlflow

from pid_parser.evaluation import (
    load_predictions_from_delta,
    load_ground_truth,
    prepare_evaluation_data,
    run_llm_evaluation,
    calculate_aggregate_metrics,
)
from pid_parser.mlflow_utils import (
    get_experiment_id_from_path,
    fetch_all_traces_for_run,
    extract_trace_info_from_dataframe,
    match_eval_records_to_traces,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

# Get predictions full table name
PREDICTIONS_FULL_TABLE = cfg.evaluation.predictions.get_full_table_name()

# Databricks FMAPI setup
DATABRICKS_HOST, DATABRICKS_TOKEN = get_databricks_client(dbutils)

# OpenAI-compatible client for Claude
fm_client = OpenAI(
    api_key=DATABRICKS_TOKEN,
    base_url=f"{DATABRICKS_HOST}/serving-endpoints",
)

# MLflow setup
mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
mlflow.set_experiment(cfg.mlflow.experiment_name_evaluation)

# Configuration values are loaded from config.json
# Update config.json to change these values

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Data

# COMMAND ----------

# Load and prepare data
predictions_df = load_predictions_from_delta(PREDICTIONS_FULL_TABLE)
ground_truth_df = load_ground_truth(cfg.evaluation.ground_truth_file)
eval_df = prepare_evaluation_data(predictions_df, ground_truth_df)

print(f"\nEvaluation dataset: {len(eval_df)} records")
display(eval_df.head())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Search for OCR Traces by PDF Path

# COMMAND ----------

# Get OCR experiment ID from experiment path
ocr_experiment_path = cfg.mlflow.ocr_experiment_name
print(f"OCR Experiment Path: {ocr_experiment_path}")

ocr_experiment_id = get_experiment_id_from_path(ocr_experiment_path)
if not ocr_experiment_id:
    raise ValueError(f"Could not find experiment ID for: {ocr_experiment_path}")

print(f"OCR Experiment ID: {ocr_experiment_id}")

if not cfg.mlflow.ocr_run_id:
    print("ERROR: OCR_RUN_ID must be set to fetch traces for evaluation")
    raise ValueError("OCR_RUN_ID is required. Please set it in config.json.")

print(f"OCR Run ID: {cfg.mlflow.ocr_run_id}")

# Fetch all traces for the run_id
# mlflow.search_traces takes 'locations' which should be a list of experiment IDs
print("\nFetching all OCR traces for the run...")
ocr_traces_df = fetch_all_traces_for_run(ocr_experiment_id, cfg.mlflow.ocr_run_id)
print(f"Found {len(ocr_traces_df)} OCR traces")

if len(ocr_traces_df) > 0:
    print("\nSample trace structure:")
    print(f"  Columns: {list(ocr_traces_df.columns)}")
    print(f"  First trace ID: {ocr_traces_df.loc[0]['trace_id'] if len(ocr_traces_df) > 0 else 'N/A'}")
    
    # Show sample trace info
    try:
        sample_idx = 0
        spans = ocr_traces_df.loc[sample_idx].get('spans', [])
        if spans and len(spans) > 0:
            attributes = spans[0].get('attributes', {})
            pdf_filename = attributes.get('pdf_filename', 'N/A')
            print(f"  Sample pdf_filename from spans: {pdf_filename}")
    except Exception as e:
        print(f"  Could not extract sample: {e}")

# COMMAND ----------

# Extract trace info mapping (pdf_filename -> {trace_id, trace_name})
print("\nExtracting trace information...")
trace_info_map = extract_trace_info_from_dataframe(ocr_traces_df)
print(f"Extracted {len(trace_info_map)} trace mappings")

# Show sample mappings
if trace_info_map:
    sample_items = list(trace_info_map.items())[:5]
    print("\nSample trace mappings:")
    for pdf_filename, info in sample_items:
        print(f"  {pdf_filename}: trace_id={info['trace_id'][:20]}..., trace_name={info['trace_name'][:50]}...")

# COMMAND ----------

# Match evaluation records to existing OCR traces
eval_df = match_eval_records_to_traces(eval_df, trace_info_map)

# Show matched records
print("\nRecords with matched OCR traces:")
display(eval_df[["pdf_filename", "ocr_trace_id", "ocr_trace_name"]].head(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run LLM-as-Judge Evaluation

# COMMAND ----------

# Run the LLM-as-judge evaluation
# Assessments will be added to existing OCR traces (no new traces created)
evaluation_results = run_llm_evaluation(
    eval_df=eval_df,
    fields=cfg.evaluation.fields_to_evaluate,
    fm_client=fm_client,
    judge_model=cfg.models.judge_model,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Aggregate Results

# COMMAND ----------

# Calculate and display aggregate metrics
aggregate_metrics = calculate_aggregate_metrics(evaluation_results, cfg.evaluation.fields_to_evaluate)

print("\n" + "=" * 80)
print("AGGREGATE EVALUATION METRICS")
print("=" * 80)
print(f"Overall Average Score: {aggregate_metrics['overall_avg_score']:.2%}")
print(f"Records Evaluated: {aggregate_metrics['records_evaluated']}")
print(f"Total Field Assessments: {aggregate_metrics['total_assessments']}")
print(f"\nRating Distribution:")
print(f"  ✓ CORRECT:           {aggregate_metrics['total_correct']}")
print(f"  ~ PARTIALLY_CORRECT: {aggregate_metrics['total_partially_correct']}")
print(f"  ✗ INCORRECT:         {aggregate_metrics['total_incorrect']}")
print(f"  ○ BOTH_NULL:         {aggregate_metrics['total_both_null']}")
print(f"  ! ERRORS:            {aggregate_metrics['total_errors']}")
print(f"\nAccuracy Rate (CORRECT only): {aggregate_metrics['accuracy_rate']:.2%}")

print("\n" + "-" * 40)
print("Per-Field Scores:")
print("-" * 40)
for field, score in aggregate_metrics["field_scores"].items():
    print(f"  {field}: {score:.2%}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## View Detailed Results

# COMMAND ----------

# Create a detailed results DataFrame
detailed_rows = []
for result in evaluation_results:
    for field, assessment in result["assessments"].items():
        detailed_rows.append({
            "pdf_filename": result["pdf_filename"],
            "field": field,
            "rating": assessment["rating"],
            "score": assessment["score"],
            "rationale": assessment["rationale"],
        })

detailed_df = pd.DataFrame(detailed_rows)
display(detailed_df)

# COMMAND ----------

# Show errors and incorrect extractions
print("\n=== INCORRECT EXTRACTIONS ===")
incorrect_df = detailed_df[detailed_df["rating"] == "INCORRECT"]
if len(incorrect_df) > 0:
    display(incorrect_df)
else:
    print("No incorrect extractions found!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC 
# MAGIC All LLM-as-judge assessments have been logged to MLflow traces.
# MAGIC 
# MAGIC To view the traces:
# MAGIC 1. Go to the **Experiments** tab in Databricks
# MAGIC 2. Select the experiment: `Engineering_Drawing_OCR_Evaluation`
# MAGIC 3. Click on the **Traces** tab
# MAGIC 4. Each record has its own trace with nested spans for each field assessment
# MAGIC 
# MAGIC Each trace contains:
# MAGIC - **Parent span**: `evaluate:<pdf_filename>` - Record-level summary
# MAGIC - **Child spans**: `judge_<field>` - Per-field LLM assessments with ratings, scores, and rationales

# COMMAND ----------

"""
Evaluation utilities for OCR extraction quality assessment.
"""

import os
import json
import re
from typing import Dict, Any, List, Optional

import pandas as pd
import numpy as np
from openai import OpenAI

from .api_utils import extract_text_from_response, extract_json_from_response
from .prompts import JUDGE_PROMPT
from .mlflow_utils import add_assessment_to_trace


def load_predictions_from_delta(table_name: str) -> pd.DataFrame:
    """
    Load predictions from Delta table.
    
    Args:
        table_name: Full table name (catalog.schema.table)
    
    Returns:
        DataFrame with predictions
    """
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.getOrCreate()
    
    df = spark.table(table_name).toPandas()
    print(f"Loaded {len(df)} predictions from Delta table: {table_name}")
    print(f"Columns: {list(df.columns)}")
    return df


def load_ground_truth(file_path: str) -> pd.DataFrame:
    """
    Load ground truth from CSV file and map columns to expected format.
    
    Expected CSV columns:
    - Doc Path in Databricks: Full path, extract filename
    - C1 - Document number or Document Number: agnoc_gas_dwg_no
    - C2 -Title: drawing_title
    - C3- Discipline Code: unit (extract from discipline code)
    - C5- Revision Code: latest_rev
    - C6- Issue purpose / Status: latest_rev_description
    
    Args:
        file_path: Path to CSV file
    
    Returns:
        DataFrame with ground truth data
    """
    # Read CSV - handle multi-line values
    df = pd.read_csv(file_path, quotechar='"', skipinitialspace=True)
    print(f"Loaded {len(df)} ground truth records from {file_path}")
    print(f"Original columns: {list(df.columns)}")
    
    # Extract filename from "Doc Path in Databricks"
    if "Doc Path in Databricks" in df.columns:
        df["pdf_filename"] = df["Doc Path in Databricks"].apply(
            lambda x: os.path.basename(str(x)) if pd.notna(x) else None
        )
    elif "doc_path" in [c.lower() for c in df.columns]:
        # Try normalized column name
        doc_path_col = [c for c in df.columns if "doc_path" in c.lower()][0]
        df["pdf_filename"] = df[doc_path_col].apply(
            lambda x: os.path.basename(str(x)) if pd.notna(x) else None
        )
    
    # Map columns to expected field names
    column_mapping = {}
    
    # Map document number to agnoc_gas_dwg_no
    if "C1 - Document number" in df.columns:
        column_mapping["C1 - Document number"] = "agnoc_gas_dwg_no"
    elif "Document Number" in df.columns:
        column_mapping["Document Number"] = "agnoc_gas_dwg_no"
    
    # Map title to drawing_title
    if "C2 -Title" in df.columns:
        column_mapping["C2 -Title"] = "drawing_title"
    elif "C2 - Title" in df.columns:
        column_mapping["C2 - Title"] = "drawing_title"
    
    # Map discipline code to unit (extract numeric part if present)
    if "C3- Discipline Code" in df.columns:
        column_mapping["C3- Discipline Code"] = "unit"
    elif "C3 - Discipline Code" in df.columns:
        column_mapping["C3 - Discipline Code"] = "unit"
    
    # Map revision code to latest_rev
    if "C5- Revision Code" in df.columns:
        column_mapping["C5- Revision Code"] = "latest_rev"
    elif "C5 - Revision Code" in df.columns:
        column_mapping["C5 - Revision Code"] = "latest_rev"
    
    # Map issue purpose/status to latest_rev_description
    if "C6- Issue purpose / Status" in df.columns:
        column_mapping["C6- Issue purpose / Status"] = "latest_rev_description"
    elif "C6 - Issue purpose / Status" in df.columns:
        column_mapping["C6 - Issue purpose / Status"] = "latest_rev_description"
    
    # Rename columns
    df = df.rename(columns=column_mapping)
    
    # Clean up title field (remove newlines, extra spaces)
    if "drawing_title" in df.columns:
        df["drawing_title"] = df["drawing_title"].apply(
            lambda x: " ".join(str(x).split()) if pd.notna(x) else None
        )
    
    # Extract unit from discipline code if it's in format like "00-40" or "13-60"
    # The unit might be the numeric part before the dash
    if "unit" in df.columns:
        def extract_unit(discipline_code):
            if pd.isna(discipline_code) or discipline_code == "":
                return None
            code_str = str(discipline_code).strip()
            # If format is like "00-40", extract first part or whole
            # For now, keep as is - may need adjustment based on actual data
            return code_str
        
        df["unit"] = df["unit"].apply(extract_unit)
    
    print(f"Mapped columns: {list(df.columns)}")
    print(f"Sample records:")
    # Note: display() is Databricks-specific, so we'll just print
    print(df[["pdf_filename"] + [c for c in column_mapping.values() if c in df.columns]].head())
    
    return df


def normalize_column_name(col: str) -> str:
    """
    Normalize column name for matching.
    
    Args:
        col: Column name to normalize
    
    Returns:
        Normalized column name
    """
    return col.strip().lower().replace(" ", "_").replace("-", "_")


def prepare_evaluation_data(
    predictions_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    join_key: str = "pdf_filename"
) -> pd.DataFrame:
    """
    Merge predictions and ground truth into evaluation dataset.
    
    Args:
        predictions_df: DataFrame with predictions
        ground_truth_df: DataFrame with ground truth
        join_key: Column name to join on
    
    Returns:
        Merged evaluation DataFrame
    """
    # Normalize column names in ground truth
    gt_cols = {col: normalize_column_name(col) for col in ground_truth_df.columns}
    ground_truth_df = ground_truth_df.rename(columns=gt_cols)
    
    # Find the join key in ground truth
    gt_join_key = None
    for col in ground_truth_df.columns:
        if "filename" in col or "file" in col or "pdf" in col:
            gt_join_key = col
            break
    
    if gt_join_key is None:
        gt_join_key = ground_truth_df.columns[0]
    
    print(f"Joining on: predictions.{join_key} <-> ground_truth.{gt_join_key}")
    
    # Rename columns to avoid conflicts
    pred_rename = {col: f"pred_{col}" for col in predictions_df.columns if col != join_key}
    gt_rename = {col: f"gt_{col}" for col in ground_truth_df.columns if col != gt_join_key}
    
    pred_df = predictions_df.rename(columns=pred_rename)
    gt_df = ground_truth_df.rename(columns=gt_rename)
    gt_df = gt_df.rename(columns={gt_join_key: join_key})
    
    # Merge
    eval_df = pd.merge(pred_df, gt_df, on=join_key, how="inner")
    print(f"Merged {len(eval_df)} records")
    
    return eval_df


def call_llm_judge(
    pdf_filename: str,
    field_name: str,
    predicted_value: Any,
    ground_truth_value: Any,
    fm_client: OpenAI,
    judge_model: str,
) -> Dict[str, Any]:
    """
    Call LLM to judge the extraction quality for a single field.
    
    Args:
        pdf_filename: Name of the PDF file
        field_name: Name of the field being evaluated
        predicted_value: Predicted value
        ground_truth_value: Ground truth value
        fm_client: OpenAI-compatible client for FMAPI
        judge_model: Model name for judging
    
    Returns:
        Dict with rating, score, and rationale
    """
    # Handle null/None values for display
    pred_display = str(predicted_value) if pd.notna(predicted_value) and predicted_value else "<null>"
    gt_display = str(ground_truth_value) if pd.notna(ground_truth_value) and ground_truth_value else "<null>"
    
    prompt = JUDGE_PROMPT.format(
        pdf_filename=pdf_filename,
        field_name=field_name,
        predicted_value=pred_display,
        ground_truth_value=gt_display,
    )
    
    try:
        completion = fm_client.chat.completions.create(
            model=judge_model,
            max_tokens=256,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        
        response_text = completion.choices[0].message.content
        
        # Handle list response format
        if isinstance(response_text, list):
            for block in response_text:
                if hasattr(block, "text"):
                    response_text = block.text
                    break
                elif isinstance(block, dict) and "text" in block:
                    response_text = block["text"]
                    break
        
        # Extract JSON from response
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            result = json.loads(json_match.group(0))
            return {
                "rating": result.get("rating", "UNKNOWN"),
                "score": float(result.get("score", 0.0)),
                "rationale": result.get("rationale", ""),
                "raw_response": response_text,
            }
        else:
            return {
                "rating": "ERROR",
                "score": 0.0,
                "rationale": f"Failed to parse JSON from response: {response_text[:200]}",
                "raw_response": response_text,
            }
            
    except Exception as e:
        return {
            "rating": "ERROR",
            "score": 0.0,
            "rationale": f"LLM call failed: {str(e)}",
            "raw_response": None,
        }


def evaluate_record_and_add_to_trace(
    row: pd.Series,
    fields: List[str],
    fm_client: OpenAI,
    judge_model: str,
    trace_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Evaluate a single record using LLM-as-judge and add assessments to existing OCR trace.
    
    Args:
        row: Evaluation record row
        fields: List of fields to evaluate
        fm_client: OpenAI-compatible client for FMAPI
        judge_model: Model name for judging
        trace_id: OCR trace ID to add assessments to
    
    Returns:
        Dict with all field assessments
    """
    pdf_filename = row.get("pdf_filename", "unknown")
    assessments = {}
    
    if not trace_id:
        print(f"  Warning: No trace_id for {pdf_filename}, skipping assessment logging")
        return assessments
    
    for field in fields:
        pred_col = f"pred_{field}"
        gt_col = f"gt_{field}"
        
        # Skip if columns don't exist
        if pred_col not in row.index or gt_col not in row.index:
            continue
        
        predicted = row[pred_col]
        ground_truth = row[gt_col]
        
        # Call the LLM judge
        assessment = call_llm_judge(
            pdf_filename=pdf_filename,
            field_name=field,
            predicted_value=predicted,
            ground_truth_value=ground_truth,
            fm_client=fm_client,
            judge_model=judge_model,
        )
        
        # Add assessment to the existing OCR trace
        add_assessment_to_trace(
            trace_id=trace_id,
            field_name=field,
            assessment=assessment,
            predicted_value=predicted,
            ground_truth_value=ground_truth,
            judge_model=judge_model,
        )
        
        assessments[field] = assessment
    
    return assessments


def run_llm_evaluation(
    eval_df: pd.DataFrame,
    fields: List[str],
    fm_client: OpenAI,
    judge_model: str,
) -> List[Dict[str, Any]]:
    """
    Run LLM-as-judge evaluation on all records.
    Adds assessments to existing OCR traces instead of creating new traces.
    
    Args:
        eval_df: Evaluation DataFrame with predictions and ground truth
        fields: List of fields to evaluate
        fm_client: OpenAI-compatible client for FMAPI
        judge_model: Model name for judging
    
    Returns:
        List of evaluation results
    """
    all_results = []
    
    print(f"\n{'='*80}")
    print(f"LLM-AS-JUDGE EVALUATION")
    print(f"{'='*80}")
    print(f"Records to evaluate: {len(eval_df)}")
    print(f"Fields: {fields}")
    print(f"Judge model: {judge_model}")
    print(f"Note: Assessments will be added to existing OCR traces (no new traces created)")
    print(f"{'='*80}\n")
    
    for idx, row in eval_df.iterrows():
        pdf_filename = row.get("pdf_filename", f"record_{idx}")
        ocr_trace_id = row.get("ocr_trace_id", None)
        ocr_trace_name = row.get("ocr_trace_name", None)
        
        print(f"\n[{idx+1}/{len(eval_df)}] Evaluating: {pdf_filename}")
        if ocr_trace_id:
            print(f"  Adding assessments to OCR trace: {ocr_trace_id[:20]}...")
        else:
            print(f"  Warning: No OCR trace found for {pdf_filename}, skipping")
            continue
        
        # Evaluate all fields and add assessments to existing trace
        assessments = evaluate_record_and_add_to_trace(
            row=row,
            fields=fields,
            fm_client=fm_client,
            judge_model=judge_model,
            trace_id=ocr_trace_id,
        )
        
        # Calculate aggregate score for this record
        scores = [a["score"] for a in assessments.values() if a["rating"] != "ERROR"]
        avg_score = np.mean(scores) if scores else 0.0
        
        # Count ratings
        rating_counts = {}
        for a in assessments.values():
            rating = a["rating"]
            rating_counts[rating] = rating_counts.get(rating, 0) + 1
        
        # Store result
        result = {
            "pdf_filename": pdf_filename,
            "ocr_trace_id": ocr_trace_id,
            "ocr_trace_name": ocr_trace_name,
            "avg_score": avg_score,
            "assessments": assessments,
            "rating_counts": rating_counts,
        }
        all_results.append(result)
        
        # Print summary
        print(f"  Score: {avg_score:.2%} | "
              f"Correct: {rating_counts.get('CORRECT', 0)} | "
              f"Partial: {rating_counts.get('PARTIALLY_CORRECT', 0)} | "
              f"Incorrect: {rating_counts.get('INCORRECT', 0)}")
    
    return all_results


def calculate_aggregate_metrics(
    results: List[Dict[str, Any]],
    fields: List[str],
) -> Dict[str, Any]:
    """
    Calculate aggregate metrics from evaluation results.
    
    Args:
        results: List of evaluation results
        fields: List of fields that were evaluated
    
    Returns:
        Dict with aggregate metrics
    """
    all_scores = []
    total_correct = 0
    total_partial = 0
    total_incorrect = 0
    total_both_null = 0
    total_error = 0
    
    field_scores = {field: [] for field in fields}
    
    for result in results:
        all_scores.append(result["avg_score"])
        
        counts = result["rating_counts"]
        total_correct += counts.get("CORRECT", 0)
        total_partial += counts.get("PARTIALLY_CORRECT", 0)
        total_incorrect += counts.get("INCORRECT", 0)
        total_both_null += counts.get("BOTH_NULL", 0)
        total_error += counts.get("ERROR", 0)
        
        for field, assessment in result["assessments"].items():
            if field in field_scores:
                field_scores[field].append(assessment["score"])
    
    # Calculate per-field averages
    field_avg_scores = {
        field: np.mean(scores) if scores else 0.0
        for field, scores in field_scores.items()
    }
    
    total_assessments = total_correct + total_partial + total_incorrect + total_both_null + total_error
    
    return {
        "overall_avg_score": np.mean(all_scores) if all_scores else 0.0,
        "records_evaluated": len(results),
        "total_assessments": total_assessments,
        "total_correct": total_correct,
        "total_partially_correct": total_partial,
        "total_incorrect": total_incorrect,
        "total_both_null": total_both_null,
        "total_errors": total_error,
        "accuracy_rate": total_correct / total_assessments if total_assessments > 0 else 0.0,
        "field_scores": field_avg_scores,
    }


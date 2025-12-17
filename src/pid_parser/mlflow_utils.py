"""
Utilities for MLflow trace management and evaluation.
"""

import pandas as pd
from typing import Dict, Optional, Any

import mlflow


def get_experiment_id_from_path(experiment_path: str) -> Optional[str]:
    """
    Get experiment ID from experiment path/name.
    
    Args:
        experiment_path: MLflow experiment name/path (e.g., "/Users/.../experiment_name")
    
    Returns:
        Experiment ID as string, or None if not found
    """
    try:
        experiment = mlflow.get_experiment_by_name(experiment_path)
        if experiment:
            return experiment.experiment_id
        else:
            print(f"Experiment not found: {experiment_path}")
            return None
    except Exception as e:
        print(f"Error getting experiment ID: {e}")
        return None


def fetch_all_traces_for_run(
    experiment_id: str,
    run_id: str,
    limit: int = 1000,
) -> pd.DataFrame:
    """
    Fetch all OCR traces for a specific run_id.
    
    Args:
        experiment_id: MLflow experiment ID (passed to locations parameter)
        run_id: MLflow run ID to filter by
        limit: Maximum number of traces to return
    
    Returns:
        DataFrame with all traces from the run
    """
    try:
        # Filter by run_id only
        filter_string = f"run = '{run_id}'"
        
        # mlflow.search_traces takes 'locations' which should be a list of experiment IDs
        traces_df = mlflow.search_traces(
            locations=[experiment_id],
            filter_string=filter_string,
            max_results=limit,
        )
        
        return traces_df
    except Exception as e:
        print(f"Error fetching traces: {e}")
        return pd.DataFrame()


def extract_trace_info_from_dataframe(traces_df: pd.DataFrame) -> Dict[str, Dict[str, str]]:
    """
    Extract trace_id and trace_name (pdf_filename) from traces DataFrame.
    
    Based on structure:
    - trace_id: ocr_traces_df.loc[idx]['trace_id']
    - pdf_filename: ocr_traces_df.loc[idx]["spans"][0]['attributes']['pdf_filename']
    - trace_name: ocr_traces_df.loc[idx]['name']
    
    Args:
        traces_df: DataFrame from mlflow.search_traces
    
    Returns:
        Dict mapping pdf_filename to {trace_id, trace_name}
        Format: {pdf_filename: {"trace_id": "...", "trace_name": "..."}}
    """
    trace_map = {}
    
    if traces_df is None or len(traces_df) == 0:
        return trace_map
    
    for idx in range(len(traces_df)):
        try:
            # Get trace_id - direct access as shown in user example
            trace_id = traces_df.loc[idx]['trace_id']
            
            # Get pdf_filename - direct access as shown: traces_df.loc[idx]["spans"][0]['attributes']['pdf_filename']
            spans = traces_df.loc[idx]["spans"]
            if spans and len(spans) > 0:
                attributes = spans[0]['attributes']
                
                # Extract pdf_filename - may need eval() if stored as string representation
                pdf_filename_raw = attributes.get('pdf_filename', None)
                if pdf_filename_raw:
                    try:
                        # Try eval() first (in case it's stored as a string representation)
                        pdf_filename = eval(pdf_filename_raw) if isinstance(pdf_filename_raw, str) else pdf_filename_raw
                    except:
                        # If eval fails, use the value directly
                        pdf_filename = pdf_filename_raw
                    
                    if pdf_filename:
                        # Get the trace name from spans[0]['name'] as shown in user example
                        trace_name = spans[0].get('name', '')
                        
                        trace_map[pdf_filename] = {
                            "trace_id": trace_id,
                            "trace_name": trace_name,
                        }
        except (KeyError, IndexError, TypeError):
            # Skip if structure doesn't match expected format
            continue
        except Exception as e:
            print(f"Error extracting trace info at index {idx}: {e}")
            continue
    
    return trace_map


def match_eval_records_to_traces(
    eval_df: pd.DataFrame,
    trace_info_map: Dict[str, Dict[str, str]],
) -> pd.DataFrame:
    """
    Match each evaluation record to its corresponding OCR trace using the trace info map.
    
    Adds 'ocr_trace_id' and 'ocr_trace_name' columns to eval_df.
    
    Args:
        eval_df: Evaluation DataFrame with 'pdf_filename' column
        trace_info_map: Dict mapping pdf_filename to {trace_id, trace_name}
    
    Returns:
        DataFrame with added ocr_trace_id and ocr_trace_name columns
    """
    trace_ids = []
    trace_names = []
    
    for _, row in eval_df.iterrows():
        pdf_filename = row.get("pdf_filename", "")
        
        # Look up in trace map
        trace_info = trace_info_map.get(pdf_filename, None)
        
        if trace_info:
            trace_ids.append(trace_info["trace_id"])
            trace_names.append(trace_info["trace_name"])
        else:
            trace_ids.append(None)
            trace_names.append(None)
    
    eval_df = eval_df.copy()
    eval_df["ocr_trace_id"] = trace_ids
    eval_df["ocr_trace_name"] = trace_names
    
    matched = sum(1 for t in trace_ids if t is not None)
    print(f"\nMatched {matched}/{len(eval_df)} records to OCR traces")
    
    return eval_df


def add_assessment_to_trace(
    trace_id: str,
    field_name: str,
    assessment: Dict[str, Any],
    predicted_value: Any,
    ground_truth_value: Any,
    judge_model: str,
) -> None:
    """
    Add an assessment to an existing OCR trace using mlflow.log_feedback.
    
    Args:
        trace_id: The OCR trace ID to add assessment to
        field_name: Field being evaluated
        assessment: Assessment dict with rating, score, rationale
        predicted_value: Predicted value
        ground_truth_value: Ground truth value
        judge_model: Name of the judge model
    """
    import pandas as pd
    
    try:
        from mlflow.entities import AssessmentSource, AssessmentSourceType
        
        # Use score as the value (float)
        assessment_value = float(assessment['score'])
        
        # Rationale is a separate parameter
        rationale = assessment.get("rationale", "")
        
        # Prepare metadata with additional details
        assessment_metadata = {
            "field": field_name,
            "rating": assessment["rating"],
            "predicted_value": str(predicted_value) if pd.notna(predicted_value) else "<null>",
            "ground_truth_value": str(ground_truth_value) if pd.notna(ground_truth_value) else "<null>",
        }
        
        # Create assessment source indicating it's from an LLM judge
        source = AssessmentSource(
            source_type=AssessmentSourceType.LLM_JUDGE,
            source_id=judge_model,
        )
        
        # Log feedback using the correct API signature
        feedback = mlflow.log_feedback(
            trace_id=trace_id,
            name=f"field_evaluation_{field_name}",
            value=assessment_value,
            source=source,
            rationale=rationale,
            metadata=assessment_metadata,
        )
        
        print(f"    ✓ Added assessment for {field_name}: {assessment['rating']} (score: {assessment['score']})")
        
    except ImportError:
        print(f"    ⚠ Could not import AssessmentSource")
        print(f"    Trying without source parameter...")
        try:
            # Fallback without source
            feedback = mlflow.log_feedback(
                trace_id=trace_id,
                name=f"field_evaluation_{field_name}",
                value=float(assessment['score']),
                rationale=assessment.get("rationale", ""),
                metadata={
                    "field": field_name,
                    "rating": assessment["rating"],
                    "predicted_value": str(predicted_value) if pd.notna(predicted_value) else "<null>",
                    "ground_truth_value": str(ground_truth_value) if pd.notna(ground_truth_value) else "<null>",
                },
            )
            print(f"    ✓ Added assessment for {field_name} (without source)")
        except Exception as e2:
            print(f"    ✗ Failed to log feedback: {e2}")
            import traceback
            traceback.print_exc()
    except Exception as e:
        print(f"    ✗ Error adding assessment for {field_name}: {e}")
        import traceback
        traceback.print_exc()


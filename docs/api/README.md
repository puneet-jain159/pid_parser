# API Documentation

This directory contains API reference documentation for the PID Parser project.

## Module Overview

The PID Parser is organized into modular Python packages in `src/pid_parser/`:

### Image Processing (`image_utils.py`)

**Purpose**: Image manipulation and PDF processing

**Key Functions**:
- `pdf_page_to_image()`: Convert PDF page to PIL Image
- `make_preview()`: Downscale image for API limits
- `image_to_base64()`: Convert image to base64 string
- `rotate_image()`: Rotate image by specified degrees
- `crop_from_normalized()`: Crop using normalized coordinates
- `resize_for_ocr()`: Resize crop for optimal OCR quality
- `get_pdf_page_count()`: Get number of pages in PDF

### API Utilities (`api_utils.py`)

**Purpose**: Handle LLM API responses

**Key Functions**:
- `extract_text_from_response()`: Safely extract text from API completion
- `extract_json_from_response()`: Extract JSON from response text (handles markdown code blocks)

### Prompts (`prompts.py`)

**Purpose**: Centralized prompt templates

**Constants**:
- `ORIENTATION_PROMPT`: Prompt for rotation detection
- `LAYOUT_PROMPT`: Prompt for title block region detection
- `OCR_PROMPT`: Prompt for metadata extraction
- `JUDGE_PROMPT`: Prompt template for LLM-as-judge evaluation

### OCR Pipeline (`ocr_pipeline.py`)

**Purpose**: Core OCR processing logic

**Key Functions**:
- `direction_to_rotation()`: Map text direction to rotation degrees
- `detect_page_rotation()`: Detect rotation needed for page (MLflow traced)
- `detect_title_block_region()`: Detect title block region (MLflow traced)
- `extract_metadata_ocr()`: Extract metadata from title block (MLflow traced)
- `process_engineering_drawing()`: Main processing pipeline (MLflow traced)
- `process_single_drawing_safe()`: Process single drawing with error handling
- `process_folder()`: Batch process all PDFs in a folder

### Delta Utilities (`delta_utils.py`)

**Purpose**: Delta table operations

**Key Functions**:
- `create_delta_table()`: Create Delta table from processing results

### MLflow Utilities (`mlflow_utils.py`)

**Purpose**: MLflow trace management

**Key Functions**:
- `get_experiment_id_from_path()`: Get experiment ID from path
- `fetch_all_traces_for_run()`: Fetch all traces for a run
- `extract_trace_info_from_dataframe()`: Extract trace info from DataFrame
- `match_eval_records_to_traces()`: Match evaluation records to OCR traces
- `add_assessment_to_trace()`: Add assessment to existing trace

### Evaluation (`evaluation.py`)

**Purpose**: Evaluation and metrics

**Key Functions**:
- `load_predictions_from_delta()`: Load predictions from Delta table
- `load_ground_truth()`: Load ground truth from CSV
- `prepare_evaluation_data()`: Merge predictions and ground truth
- `call_llm_judge()`: Call LLM to judge extraction quality
- `evaluate_record_and_add_to_trace()`: Evaluate single record
- `run_llm_evaluation()`: Run evaluation on all records
- `calculate_aggregate_metrics()`: Calculate aggregate metrics

## Configuration API

### Configuration Loading (`notebooks/config.py`)

**Key Functions**:
- `load_config()`: Load and validate configuration from JSON
- `get_config()`: Get validated Config instance
- `get_databricks_client()`: Get Databricks credentials from notebook context

**Configuration Classes**:
- `Config`: Main configuration container
- `ProjectConfig`: Project paths
- `ModelsConfig`: Model names
- `MLflowConfig`: MLflow settings
- `OCRPipelineConfig`: OCR pipeline settings
- `EvaluationConfig`: Evaluation settings

## Usage Examples

### Using OCR Pipeline

```python
from pid_parser.ocr_pipeline import process_engineering_drawing
from openai import OpenAI

fm_client = OpenAI(...)
result = process_engineering_drawing(
    pdf_path="path/to/drawing.pdf",
    fm_client=fm_client,
    ocr_model="databricks-claude-sonnet-4-5",
    layout_model="databricks-claude-3-7-sonnet",
    page_index=0,
)
```

### Using Configuration

```python
from config import get_config

cfg = get_config()
# Type-safe access
print(cfg.project.project_root)
print(cfg.models.claude_fmapi_model)
full_table = cfg.evaluation.predictions.get_full_table_name()
```

### Using Evaluation

```python
from pid_parser.evaluation import run_llm_evaluation

results = run_llm_evaluation(
    eval_df=eval_df,
    fields=["drawing_title", "agnoc_gas_dwg_no"],
    fm_client=fm_client,
    judge_model="databricks-claude-sonnet-4-5",
)
```

## Reference

For detailed function signatures and parameters, see the source code in `src/pid_parser/`. All functions include type hints and docstrings following Google style.


# Databricks Notebooks

This directory contains Databricks notebooks for the PID Parser project.

## Current Notebooks

- **`extract_metadata_notebook.py`** - Main OCR pipeline for extracting metadata from engineering drawings
  - Processes PDF files to extract title block and revision table metadata
  - Uses vision models via Databricks FMAPI
  - Stores results in Delta tables with MLflow tracing

- **`evaluate_extraction_notebook.py`** - LLM-as-judge evaluation system
  - Evaluates OCR extraction quality using Claude as a judge
  - Compares predictions against ground truth
  - Logs assessments to MLflow traces

## Configuration Files

**`config.json`** - JSON configuration file containing all settings  
**`config.py`** - Python module that loads and validates the JSON configuration

The configuration system uses:
- **JSON file** (`config.json`) for easy editing and version control
- **Dataclasses with validation** (`config.py`) for type safety and error checking

Before running any notebook, **update the values in `config.json`** to match your Databricks environment.

### Configuration Structure:

1. **Project Configuration** (`project`)
   - `project_root` - Path to the project in Databricks workspace

2. **Model Configuration** (`models`)
   - `claude_fmapi_model` - Model for OCR extraction
   - `layout_model` - Model for layout detection
   - `judge_model` - Model for LLM-as-judge evaluation

3. **MLflow Configuration** (`mlflow`)
   - `tracking_uri` - MLflow tracking URI (typically "databricks")
   - `experiment_name_ocr` - Experiment name for OCR pipeline
   - `experiment_name_evaluation` - Experiment name for evaluation
   - `ocr_experiment_name` - OCR experiment name (for trace lookup)
   - `ocr_run_id` - Run ID from OCR batch processing (can be null)
   - `run_name` - MLflow run name for batch processing

4. **OCR Pipeline Configuration** (`ocr_pipeline`)
   - `input_folder` - Folder containing PDF files
   - `output_dir` - Directory for cropped images
   - `delta_table` - Delta table configuration (catalog, schema, table_name)
   - `processing` - Processing options (dpi, force_rotation)
   - `test_files` - Test file paths for single file processing (optional)

5. **Evaluation Configuration** (`evaluation`)
   - `predictions` - Predictions table configuration (catalog, schema, table)
   - `ground_truth_file` - Path to ground truth CSV file
   - `pdf_folder` - PDF folder path for trace lookup
   - `fields_to_evaluate` - List of fields to evaluate

## Usage

These notebooks are designed to run in Databricks environment with:
- Databricks FMAPI access
- MLflow tracking enabled
- Unity Catalog access

## Setup

Before running the notebooks:

1. **Update `config.json`** with your environment-specific values:
   - Update all paths to match your Databricks workspace
   - Set your catalog and schema names
   - Configure model names if different
   - Set MLflow experiment paths
   - Update `ocr_run_id` with the actual run ID from OCR batch processing

2. Ensure the project code is available in the Databricks workspace

3. Install required dependencies (handled by `%pip install` commands in notebooks)

### Configuration Validation

The `config.py` module automatically validates the configuration when loaded:
- **Type checking**: Ensures all values are of the correct type
- **Value validation**: Checks ranges (e.g., DPI between 100-1200)
- **Required fields**: Ensures all required fields are present
- **Path validation**: Verifies paths start with `/Workspace/`

If validation fails, you'll get a clear error message indicating what needs to be fixed.

## Module Imports

The notebooks import functions from the `pid_parser` package modules:
- `pid_parser.ocr_pipeline` - Main OCR processing functions
- `pid_parser.evaluation` - Evaluation utilities
- `pid_parser.mlflow_utils` - MLflow trace management
- `pid_parser.delta_utils` - Delta table utilities

## Benefits of JSON + Dataclass Configuration

- **Type safety**: Dataclasses provide type hints and validation
- **Early error detection**: Configuration errors are caught at load time, not runtime
- **Easy editing**: JSON format is human-readable and easy to edit
- **Validation**: Automatic validation of values (ranges, formats, required fields)
- **IDE support**: Type hints enable better autocomplete and error detection
- **Single source of truth**: All configuration in one JSON file
- **Environment management**: Easy to maintain different configs for different environments
- **Version control**: Configuration changes are tracked in git

### Example Usage in Notebooks

```python
from config import get_config

# Load validated configuration
cfg = get_config()

# Access configuration values
print(cfg.project.project_root)
print(cfg.models.claude_fmapi_model)
print(cfg.ocr_pipeline.input_folder)

# Helper methods
full_table_name = cfg.evaluation.predictions.get_full_table_name()
```

## Local Development

For local development, extract reusable code into `src/pid_parser/` modules. The notebooks serve as orchestration layers that call these modules.

# Architecture Documentation

This directory contains high-level architecture documentation for the PID Parser project.

## System Overview

The PID Parser is an OCR pipeline that extracts metadata from engineering drawings using:
- PDF processing (PyMuPDF)
- Vision models via Databricks FMAPI (Claude)
- MLflow for tracing and evaluation
- Delta tables for data storage

## Architecture Pattern

The project follows a **modular architecture** with clear separation of concerns:

- **Notebooks** (`notebooks/`): Orchestration layers that coordinate the workflow
- **Core Modules** (`src/pid_parser/`): Reusable, testable functions
- **Configuration** (`notebooks/config.json` + `config.py`): Centralized, validated settings

## Components

### 1. OCR Pipeline (`extract_metadata_notebook.py`)

Orchestrates the metadata extraction workflow using functions from `src/pid_parser/`:

- **Image Processing** (`image_utils.py`): PDF to image conversion, rotation, cropping
- **Rotation Detection** (`ocr_pipeline.py`): Detects page orientation using vision model
- **Layout Detection** (`ocr_pipeline.py`): Identifies title block region
- **Metadata Extraction** (`ocr_pipeline.py`): Extracts structured metadata from title block
- **Data Storage** (`delta_utils.py`): Saves results to Delta tables

### 2. Evaluation System (`evaluate_extraction_notebook.py`)

Orchestrates the evaluation workflow:

- **Data Loading** (`evaluation.py`): Loads predictions and ground truth
- **Trace Management** (`mlflow_utils.py`): Links evaluation to OCR traces
- **LLM-as-Judge** (`evaluation.py`): Uses Claude to assess extraction quality
- **Metrics Calculation** (`evaluation.py`): Computes aggregate metrics
- **Assessment Logging** (`mlflow_utils.py`): Logs assessments to MLflow traces

### 3. Configuration System (`notebooks/config.json` + `config.py`)

- **JSON Configuration**: Human-readable settings in `config.json`
- **Dataclass Validation**: Type-safe configuration objects with validation
- **Error Detection**: Catches configuration errors at load time

## Data Flow

### OCR Pipeline Flow

1. **Input**: PDF file
2. **Image Conversion** (`image_utils.pdf_page_to_image`): PDF → High-res image
3. **Rotation Detection** (`ocr_pipeline.detect_page_rotation`): Full page → Rotation angle
4. **Image Rotation** (`image_utils.rotate_image`): Apply rotation correction
5. **Layout Detection** (`ocr_pipeline.detect_title_block_region`): Rotated image → Title block coordinates
6. **Crop & Resize** (`image_utils.crop_from_normalized`, `resize_for_ocr`): Extract title block
7. **OCR Extraction** (`ocr_pipeline.extract_metadata_ocr`): Title block → Structured metadata
8. **Storage** (`delta_utils.create_delta_table`): Results → Delta table

### Evaluation Flow

1. **Data Loading**: Load predictions from Delta table, ground truth from CSV
2. **Trace Lookup** (`mlflow_utils`): Find corresponding OCR traces
3. **Field Evaluation** (`evaluation.call_llm_judge`): Evaluate each field using LLM
4. **Assessment Logging** (`mlflow_utils.add_assessment_to_trace`): Log to MLflow traces
5. **Metrics Calculation** (`evaluation.calculate_aggregate_metrics`): Compute aggregate scores

## Module Dependencies

```
notebooks/
├── extract_metadata_notebook.py
│   ├── config.py (loads config.json)
│   ├── ocr_pipeline.py
│   ├── image_utils.py
│   ├── delta_utils.py
│   └── prompts.py
│
└── evaluate_extraction_notebook.py
    ├── config.py (loads config.json)
    ├── evaluation.py
    ├── mlflow_utils.py
    ├── api_utils.py
    └── prompts.py
```

## Configuration Architecture

```
config.json (JSON)
    ↓
config.py (loads & validates)
    ↓
Dataclass instances (type-safe)
    ↓
Notebooks (use validated config)
```

## Future Documents

- Component interaction diagrams
- Deployment architecture
- Integration architecture
- Performance optimization strategies


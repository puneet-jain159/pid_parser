# Technical Design Documents

This directory contains detailed technical design documents for the PID Parser project.

## Current Documents

- (Add documents as the project evolves)

## Recent Architecture Changes

### Modularization (Completed)

The codebase has been refactored from monolithic notebooks into a modular architecture:

**Before**: All code in notebooks  
**After**: Reusable modules in `src/pid_parser/` with notebooks as orchestration layers

**Benefits**:
- Code reusability across notebooks and scripts
- Easier unit testing of individual functions
- Better code organization and maintainability
- Type safety with type hints

### Configuration System (Completed)

**Implementation**: JSON + Dataclass validation

- **`config.json`**: Human-readable JSON configuration
- **`config.py`**: Dataclass-based loader with validation
- **Validation**: Type checking, range validation, required field checks

**Benefits**:
- Type safety at configuration load time
- Early error detection
- IDE autocomplete support
- Clear error messages for invalid config

## Module Design

### Image Processing (`image_utils.py`)
- **Purpose**: Low-level image manipulation
- **Functions**: PDF conversion, resizing, rotation, cropping
- **Dependencies**: PyMuPDF, PIL

### API Utilities (`api_utils.py`)
- **Purpose**: Handle LLM API responses
- **Functions**: Extract text, parse JSON from responses
- **Dependencies**: None (pure utility)

### Prompts (`prompts.py`)
- **Purpose**: Centralize prompt templates
- **Content**: Orientation, layout, OCR, and evaluation prompts
- **Dependencies**: None

### OCR Pipeline (`ocr_pipeline.py`)
- **Purpose**: Core OCR processing logic
- **Functions**: Rotation detection, layout detection, metadata extraction, batch processing
- **Dependencies**: `image_utils`, `api_utils`, `prompts`, MLflow

### Evaluation (`evaluation.py`)
- **Purpose**: Evaluation and metrics
- **Functions**: Data loading, LLM-as-judge, metrics calculation
- **Dependencies**: `api_utils`, `prompts`, `mlflow_utils`

### MLflow Utilities (`mlflow_utils.py`)
- **Purpose**: MLflow trace management
- **Functions**: Experiment lookup, trace extraction, assessment logging
- **Dependencies**: MLflow

### Delta Utilities (`delta_utils.py`)
- **Purpose**: Delta table operations
- **Functions**: Table creation with schema
- **Dependencies**: PySpark

## Template

When creating a new technical design document, include:
- Problem statement
- Proposed solution
- Architecture/design details
- Implementation considerations
- Testing strategy
- Future improvements


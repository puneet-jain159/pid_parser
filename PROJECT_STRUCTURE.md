# Project Structure

This document provides a detailed overview of the PID Parser project structure.

## Directory Tree

```
pid_parser/
│
├── .cursorrules                    # Cursor IDE rules (root level)
├── .gitignore                      # Git ignore patterns
├── README.md                       # Main project documentation
├── PROJECT_STRUCTURE.md            # This file
├── requirements.txt                # Python dependencies
├── pyproject.toml                  # Project configuration (setuptools)
│
├── src/                            # Source code package
│   └── pid_parser/                 # Main Python package
│       ├── __init__.py            # Package initialization
│       ├── image_utils.py          # Image processing utilities
│       ├── api_utils.py            # API response handling
│       ├── prompts.py              # Prompt templates
│       ├── ocr_pipeline.py         # Main OCR pipeline functions
│       ├── delta_utils.py          # Delta table utilities
│       ├── mlflow_utils.py         # MLflow trace management
│       └── evaluation.py           # Evaluation utilities
│
├── notebooks/                      # Databricks notebooks
│   ├── README.md                   # Notebook documentation
│   ├── config.json                 # JSON configuration file
│   ├── config.py                   # Configuration loader with validation
│   ├── extract_metadata_notebook.py  # Main OCR pipeline notebook
│   └── evaluate_extraction_notebook.py  # LLM-as-judge evaluation notebook
│
├── docs/                           # Documentation
│   ├── README.md                   # Documentation index
│   ├── technical_design/           # Technical design documents
│   │   └── README.md               # Technical design index
│   ├── architecture/               # Architecture documentation
│   │   └── README.md               # Architecture overview
│   └── api/                        # API reference
│       └── README.md               # API documentation index
│
├── tests/                          # Test suite
│   ├── __init__.py                 # Test package initialization
│   └── README.md                   # Testing documentation
│
├── data/                           # Data files
│   ├── raw/                        # Raw input data
│   │   └── .gitkeep                # Keep directory in git
│   └── processed/                  # Processed output data
│       └── .gitkeep                # Keep directory in git
│
├── cursor_rules/                   # Cursor IDE rules (alternative location)
│   └── .cursorrules                # Project-specific Cursor rules
│
└── [data files]                    # CSV/Excel files (evaluation data)
    ├── e2e_workflow_v2.csv
    └── Labelled Evaluation Data_11-12-25.xlsx - New Samples.csv
```

## Directory Descriptions

### `src/pid_parser/`
Core Python package containing modular, reusable code:

- **`image_utils.py`**: PDF to image conversion, image manipulation (resize, rotate, crop)
- **`api_utils.py`**: API response parsing and JSON extraction from LLM responses
- **`prompts.py`**: Prompt templates for orientation detection, layout detection, OCR, and evaluation
- **`ocr_pipeline.py`**: Main OCR pipeline functions (rotation detection, layout detection, metadata extraction, batch processing)
- **`delta_utils.py`**: Delta table creation and schema management
- **`mlflow_utils.py`**: MLflow trace management, experiment lookup, assessment logging
- **`evaluation.py`**: Evaluation utilities (data loading, LLM-as-judge, metrics calculation)

All functions are designed to be reusable and testable independently.

### `notebooks/`
Databricks notebooks and configuration:

- **`config.json`**: Centralized JSON configuration file with all settings
- **`config.py`**: Configuration loader with dataclass validation
- **`extract_metadata_notebook.py`**: Main OCR processing pipeline (orchestrates `ocr_pipeline.py` functions)
- **`evaluate_extraction_notebook.py`**: Evaluation using LLM-as-judge methodology (orchestrates `evaluation.py` functions)

The notebooks serve as orchestration layers that import and use functions from the `src/pid_parser/` modules.

### `docs/`
Documentation organized by type:
- **technical_design/**: Detailed technical specifications, algorithms, data flows
- **architecture/**: High-level system architecture, component interactions
- **api/**: API reference, function documentation, usage examples

### `tests/`
Unit tests and integration tests. Use pytest for testing framework.

### `data/`
Data storage:
- **raw/**: Input PDFs, evaluation datasets
- **processed/**: Extracted metadata, cropped images, results

### `cursor_rules/`
Project-specific rules for Cursor IDE. Also available as `.cursorrules` in root.

## File Purposes

### Configuration Files
- **requirements.txt**: Python package dependencies
- **pyproject.toml**: Modern Python project configuration (setuptools)
- **notebooks/config.json**: JSON configuration file for all notebook settings
- **notebooks/config.py**: Configuration loader with dataclass validation
- **.gitignore**: Files and directories to exclude from version control
- **.cursorrules**: Cursor IDE configuration and coding standards

### Documentation Files
- **README.md**: Main project overview and getting started guide
- **PROJECT_STRUCTURE.md**: This file - detailed structure documentation
- **docs/**: Various documentation organized by category

## Recent Updates

1. ✅ **Modularized Code**: All reusable functions extracted into `src/pid_parser/` modules
2. ✅ **Configuration System**: JSON-based configuration with dataclass validation
3. ✅ **Notebook Refactoring**: Notebooks now import from modules and use centralized config
4. ✅ **Type Safety**: Dataclasses provide type hints and validation

## Next Steps

1. **Add Tests**: Create unit tests for core functionality in `src/pid_parser/`
2. **Document APIs**: Add detailed API documentation in `docs/api/`
3. **Technical Design**: Document algorithms and design decisions in `docs/technical_design/`
4. **Integration Tests**: Add integration tests for the full pipeline

## Notes

- Databricks notebooks remain in `notebooks/` for Databricks environment execution
- For local development, extract code into `src/pid_parser/` modules
- Keep data files organized in `data/` directories
- Update documentation as the project evolves


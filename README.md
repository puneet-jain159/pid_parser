# PID Parser

Engineering Drawing OCR Pipeline for extracting metadata from P&ID (Piping and Instrumentation Diagram) and technical drawings using vision models via Databricks FMAPI.

## Overview

The PID Parser is a comprehensive system that:
- Converts PDF engineering drawings to images
- Detects and corrects page rotation
- Identifies title block and revision table regions
- Extracts metadata using vision models (Claude via Databricks FMAPI)
- Evaluates extraction quality using LLM-as-judge methodology
- Stores results in Delta tables with MLflow tracing

## Project Structure

```
pid_parser/
├── src/                    # Source code package
│   └── pid_parser/         # Main package
│       ├── __init__.py
│       ├── image_utils.py      # Image processing utilities
│       ├── api_utils.py        # API response handling
│       ├── prompts.py          # Prompt templates
│       ├── ocr_pipeline.py     # Main OCR pipeline functions
│       ├── delta_utils.py      # Delta table utilities
│       ├── mlflow_utils.py     # MLflow trace management
│       └── evaluation.py       # Evaluation utilities
├── notebooks/              # Databricks notebooks
│   ├── config.json             # JSON configuration file
│   ├── config.py               # Configuration loader with validation
│   ├── extract_metadata_notebook.py  # Main OCR pipeline
│   └── evaluate_extraction_notebook.py  # LLM-as-judge evaluation
├── docs/                   # Documentation
│   ├── technical_design/  # Detailed technical design docs
│   ├── architecture/       # Architecture documentation
│   └── api/                # API reference
├── tests/                  # Unit and integration tests
├── data/                   # Data files
│   ├── raw/                # Raw input data
│   └── processed/          # Processed output data
├── cursor_rules/           # Cursor IDE rules (alternative location)
├── .cursorrules            # Cursor IDE rules (root level)
├── requirements.txt        # Python dependencies
├── pyproject.toml          # Project configuration
└── README.md              # This file
```

## Features

### OCR Pipeline
- **PDF Processing**: High-resolution PDF to image conversion using PyMuPDF
- **Rotation Detection**: Automatic detection and correction of page orientation
- **Layout Detection**: Intelligent identification of title block and revision table regions
- **Metadata Extraction**: Structured extraction of:
  - Contractor name
  - Project title
  - Drawing title
  - Unit number
  - Plant name
  - Project numbers
  - Revision history

### Evaluation System
- **LLM-as-Judge**: Automated quality assessment using Claude
- **Ground Truth Comparison**: Compare predictions against labeled data
- **MLflow Integration**: Comprehensive tracing and logging
- **Metrics**: Accuracy, partial correctness, and error tracking

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd pid_parser

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Databricks Environment

The notebooks are designed to run in Databricks with:
- Databricks FMAPI access
- MLflow tracking enabled
- Unity Catalog access

**Setup Steps:**

1. Upload notebooks and `src/pid_parser/` package to Databricks workspace
2. **Update `notebooks/config.json`** with your environment-specific values:
   - Project root path
   - Input/output folders
   - Delta table catalog and schema
   - MLflow experiment paths
   - Model names
3. Run `extract_metadata_notebook.py` for OCR processing
4. Run `evaluate_extraction_notebook.py` for evaluation

### Configuration

The project uses a **JSON-based configuration system** with dataclass validation:

- **`notebooks/config.json`**: Human-readable JSON configuration file
- **`notebooks/config.py`**: Python module that loads and validates the JSON

The configuration system provides:
- **Type safety**: Dataclasses with type hints
- **Validation**: Automatic validation of values (ranges, formats, required fields)
- **Early error detection**: Configuration errors caught at load time
- **IDE support**: Autocomplete and type checking

See `notebooks/README.md` for detailed configuration documentation.

### Local Development

The codebase is modularized into reusable Python modules in `src/pid_parser/`:
- Import and use functions in your own scripts
- Write unit tests for individual modules
- Develop and test locally before deploying to Databricks

## Code Organization

The project follows a modular architecture:

### Core Modules (`src/pid_parser/`)

- **`image_utils.py`**: PDF to image conversion, image manipulation, cropping
- **`api_utils.py`**: API response parsing and JSON extraction
- **`prompts.py`**: Prompt templates for vision model interactions
- **`ocr_pipeline.py`**: Main OCR pipeline (rotation detection, layout detection, metadata extraction)
- **`delta_utils.py`**: Delta table creation and management
- **`mlflow_utils.py`**: MLflow trace management and evaluation logging
- **`evaluation.py`**: LLM-as-judge evaluation and metrics calculation

### Notebooks (`notebooks/`)

- **`extract_metadata_notebook.py`**: Orchestrates the OCR pipeline
- **`evaluate_extraction_notebook.py`**: Orchestrates the evaluation process
- **`config.json`**: Centralized configuration (JSON format)
- **`config.py`**: Configuration loader with validation

## Dependencies

- `pymupdf`: PDF processing
- `pillow`: Image manipulation
- `openai`: OpenAI-compatible API client
- `mlflow`: Experiment tracking and tracing
- `pandas`: Data manipulation
- `numpy`: Numerical operations

See `requirements.txt` for complete list.

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/pid_parser --cov-report=html
```

### Code Style

- Follow PEP 8
- Use type hints
- Maximum line length: 100 characters
- Use black for formatting

### Documentation

- Technical design docs: `docs/technical_design/`
- Architecture docs: `docs/architecture/`
- API reference: `docs/api/`

## Contributing

1. Create a feature branch
2. Make your changes
3. Add tests
4. Update documentation
5. Submit a pull request

## License

MIT License

## Contact

For questions or issues, please open an issue in the repository.

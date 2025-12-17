# PID Parser Documentation

This directory contains technical documentation for the PID Parser project.

## Structure

- **technical_design/**: Detailed technical design documents
  - System architecture
  - Component specifications
  - Data flow diagrams
  - Algorithm descriptions
  - Module design patterns

- **architecture/**: High-level architecture documentation
  - System overview
  - Component interactions
  - Modular architecture
  - Configuration system
  - Data flow diagrams
  - Deployment architecture
  - Integration points

- **api/**: API documentation
  - Function/class reference
  - Module overview
  - Usage examples
  - Integration guides
  - Configuration API

## Recent Updates

### Modular Architecture (✅ Completed)

The codebase has been refactored into a modular structure:

- **Core Modules** (`src/pid_parser/`): 7 reusable Python modules
  - `image_utils.py`: Image processing
  - `api_utils.py`: API response handling
  - `prompts.py`: Prompt templates
  - `ocr_pipeline.py`: Main OCR pipeline
  - `delta_utils.py`: Delta table utilities
  - `mlflow_utils.py`: MLflow trace management
  - `evaluation.py`: Evaluation utilities

- **Notebooks** (`notebooks/`): Orchestration layers
  - Import and use functions from modules
  - Focus on workflow coordination

### Configuration System (✅ Completed)

- **JSON Configuration** (`config.json`): Human-readable settings
- **Dataclass Validation** (`config.py`): Type-safe configuration with validation
- **Early Error Detection**: Configuration errors caught at load time

### Notebook Refactoring (✅ Completed)

- Renamed to descriptive names:
  - `extract_metadata_notebook.py` (was `wood_poc.py`)
  - `evaluate_extraction_notebook.py` (was `ocr_evaluation.py`)
- All configuration abstracted to `config.json`
- Notebooks use validated configuration objects

## Contributing

When adding new features or making significant changes:
1. Update relevant technical design docs
2. Add architecture diagrams if system structure changes
3. Update API documentation for new functions/classes
4. Update configuration schema if adding new settings
5. Keep module documentation in sync with code changes


"""
Configuration management for PID Parser notebooks.

This module loads configuration from config.json and provides validated dataclasses
for type-safe configuration access.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any


# ============================================================================
# Dataclass Definitions
# ============================================================================

@dataclass
class ProjectConfig:
    """Project path configuration."""
    project_root: str
    
    def __post_init__(self):
        """Validate project root."""
        if not self.project_root or not isinstance(self.project_root, str):
            raise ValueError("project_root must be a non-empty string")
        if not self.project_root.startswith("/Workspace/"):
            raise ValueError("project_root must start with /Workspace/")


@dataclass
class ModelsConfig:
    """Model configuration for different tasks."""
    claude_fmapi_model: str
    layout_model: str
    judge_model: str
    
    def __post_init__(self):
        """Validate model names."""
        if not all([self.claude_fmapi_model, self.layout_model, self.judge_model]):
            raise ValueError("All model names must be non-empty strings")


@dataclass
class DeltaTableConfig:
    """Delta table configuration."""
    table_name: str
    catalog: str
    schema: str
    
    def __post_init__(self):
        """Validate delta table config."""
        if not all([self.table_name, self.catalog, self.schema]):
            raise ValueError("table_name, catalog, and schema must be non-empty strings")
    
    def get_full_table_name(self) -> str:
        """Get the full table name in catalog.schema.table format."""
        return f"{self.catalog}.{self.schema}.{self.table_name}"


@dataclass
class ProcessingConfig:
    """Processing options for OCR pipeline."""
    dpi: int
    force_rotation: Optional[int] = None
    
    def __post_init__(self):
        """Validate processing options."""
        if self.dpi < 100 or self.dpi > 1200:
            raise ValueError("DPI must be between 100 and 1200")
        if self.force_rotation is not None and self.force_rotation not in [0, 90, 180, 270]:
            raise ValueError("force_rotation must be one of: 0, 90, 180, 270, or None")


@dataclass
class TestFilesConfig:
    """Test file paths for single file processing."""
    test_pdf: Optional[str] = None
    test_output: Optional[str] = None
    
    def __post_init__(self):
        """Validate test file paths."""
        if self.test_pdf and not isinstance(self.test_pdf, str):
            raise ValueError("test_pdf must be a string or None")
        if self.test_output and not isinstance(self.test_output, str):
            raise ValueError("test_output must be a string or None")


@dataclass
class OCRPipelineConfig:
    """OCR pipeline configuration."""
    input_folder: str
    output_dir: str
    delta_table: DeltaTableConfig
    processing: ProcessingConfig
    test_files: TestFilesConfig
    
    def __post_init__(self):
        """Validate OCR pipeline config."""
        if not self.input_folder or not isinstance(self.input_folder, str):
            raise ValueError("input_folder must be a non-empty string")
        if not self.output_dir or not isinstance(self.output_dir, str):
            raise ValueError("output_dir must be a non-empty string")
        if not self.input_folder.startswith("/Workspace/"):
            raise ValueError("input_folder must start with /Workspace/")
        if not self.output_dir.startswith("/Workspace/"):
            raise ValueError("output_dir must start with /Workspace/")


@dataclass
class PredictionsConfig:
    """Predictions table configuration."""
    catalog: str
    schema: str
    table: str
    
    def __post_init__(self):
        """Validate predictions config."""
        if not all([self.catalog, self.schema, self.table]):
            raise ValueError("catalog, schema, and table must be non-empty strings")
    
    def get_full_table_name(self) -> str:
        """Get the full table name in catalog.schema.table format."""
        return f"{self.catalog}.{self.schema}.{self.table}"


@dataclass
class EvaluationConfig:
    """Evaluation configuration."""
    predictions: PredictionsConfig
    ground_truth_file: str
    pdf_folder: str
    fields_to_evaluate: List[str]
    
    def __post_init__(self):
        """Validate evaluation config."""
        if not self.ground_truth_file or not isinstance(self.ground_truth_file, str):
            raise ValueError("ground_truth_file must be a non-empty string")
        if not self.pdf_folder or not isinstance(self.pdf_folder, str):
            raise ValueError("pdf_folder must be a non-empty string")
        if not isinstance(self.fields_to_evaluate, list) or len(self.fields_to_evaluate) == 0:
            raise ValueError("fields_to_evaluate must be a non-empty list")
        if not all(isinstance(field, str) for field in self.fields_to_evaluate):
            raise ValueError("All fields in fields_to_evaluate must be strings")


@dataclass
class MLflowConfig:
    """MLflow configuration."""
    tracking_uri: str
    experiment_name_ocr: str
    experiment_name_evaluation: str
    ocr_experiment_name: str
    ocr_run_id: Optional[str] = None
    run_name: str = "engineering_drawing_batch_ocr"
    
    def __post_init__(self):
        """Validate MLflow config."""
        if not self.tracking_uri or not isinstance(self.tracking_uri, str):
            raise ValueError("tracking_uri must be a non-empty string")
        if not self.experiment_name_ocr or not isinstance(self.experiment_name_ocr, str):
            raise ValueError("experiment_name_ocr must be a non-empty string")
        if not self.experiment_name_evaluation or not isinstance(self.experiment_name_evaluation, str):
            raise ValueError("experiment_name_evaluation must be a non-empty string")
        if not self.ocr_experiment_name or not isinstance(self.ocr_experiment_name, str):
            raise ValueError("ocr_experiment_name must be a non-empty string")
        if not self.run_name or not isinstance(self.run_name, str):
            raise ValueError("run_name must be a non-empty string")


@dataclass
class Config:
    """Main configuration dataclass containing all configuration sections."""
    project: ProjectConfig
    models: ModelsConfig
    mlflow: MLflowConfig
    ocr_pipeline: OCRPipelineConfig
    evaluation: EvaluationConfig
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """Create Config instance from dictionary."""
        return cls(
            project=ProjectConfig(**config_dict["project"]),
            models=ModelsConfig(**config_dict["models"]),
            mlflow=MLflowConfig(**config_dict["mlflow"]),
            ocr_pipeline=OCRPipelineConfig(
                input_folder=config_dict["ocr_pipeline"]["input_folder"],
                output_dir=config_dict["ocr_pipeline"]["output_dir"],
                delta_table=DeltaTableConfig(**config_dict["ocr_pipeline"]["delta_table"]),
                processing=ProcessingConfig(**config_dict["ocr_pipeline"]["processing"]),
                test_files=TestFilesConfig(**config_dict["ocr_pipeline"]["test_files"]),
            ),
            evaluation=EvaluationConfig(
                predictions=PredictionsConfig(**config_dict["evaluation"]["predictions"]),
                ground_truth_file=config_dict["evaluation"]["ground_truth_file"],
                pdf_folder=config_dict["evaluation"]["pdf_folder"],
                fields_to_evaluate=config_dict["evaluation"]["fields_to_evaluate"],
            ),
        )


# ============================================================================
# Configuration Loading
# ============================================================================

def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from JSON file.
    
    Args:
        config_path: Path to config.json file. If None, looks for config.json
                     in the same directory as this file.
    
    Returns:
        Validated Config instance
    
    Raises:
        FileNotFoundError: If config file doesn't exist
        json.JSONDecodeError: If config file is not valid JSON
        ValueError: If configuration validation fails
    """
    if config_path is None:
        # Get the directory where this file is located
        current_dir = Path(__file__).parent
        config_path = current_dir / "config.json"
    else:
        config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}\n"
            f"Please create config.json in the notebooks directory."
        )
    
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(
            f"Invalid JSON in configuration file: {e.msg}",
            e.doc,
            e.pos
        )
    
    try:
        config = Config.from_dict(config_dict)
        return config
    except KeyError as e:
        raise ValueError(
            f"Missing required configuration key: {e}\n"
            f"Please check your config.json file."
        )
    except TypeError as e:
        raise ValueError(
            f"Invalid configuration value type: {e}\n"
            f"Please check your config.json file."
        )


# ============================================================================
# Global Configuration Instance
# ============================================================================

# Load configuration on module import
# This will raise an error if config.json is missing or invalid
try:
    _config = load_config()
except Exception as e:
    # Store error for better error message when accessed
    _config_error = e
    _config = None


def get_config() -> Config:
    """
    Get the global configuration instance.
    
    Returns:
        Config instance
    
    Raises:
        RuntimeError: If configuration failed to load
    """
    if _config is None:
        raise RuntimeError(
            f"Configuration failed to load: {_config_error}\n"
            f"Please check your config.json file and ensure it's valid."
        )
    return _config


# ============================================================================
# Helper Functions for Backward Compatibility
# ============================================================================

def get_databricks_client(dbutils):
    """
    Get Databricks host and token from notebook context.
    
    Args:
        dbutils: Databricks utilities object from notebook context
    
    Returns:
        Tuple of (host, token)
    """
    host = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
    token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
    return host, token

"""Reusable scikit-learn model selection engine."""

from .config import ExperimentConfig, SearchConfig
from .data import DatasetBundle, load_csv_dataset, split_dataset
from .registry import EstimatorSpec, get_estimator_spec, list_estimators
from .selection import ModelSelectionResult, run_model_selection, run_model_suite

__all__ = [
    "DatasetBundle",
    "EstimatorSpec",
    "ExperimentConfig",
    "ModelSelectionResult",
    "SearchConfig",
    "get_estimator_spec",
    "list_estimators",
    "load_csv_dataset",
    "run_model_selection",
    "run_model_suite",
    "split_dataset",
]

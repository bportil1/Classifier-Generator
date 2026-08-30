from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

ScalingMode = Literal["auto", "none", "standard", "minmax"]


@dataclass(slots=True)
class SearchConfig:
    """Configuration for cross-validated model selection."""

    cv_folds: int = 5
    scoring: str = "accuracy"
    n_jobs: int = -1
    random_state: int = 42
    scaling: ScalingMode = "auto"
    refit: bool = True
    error_score: float = float("nan")

    def validate(self) -> None:
        if self.cv_folds < 2:
            raise ValueError("cv_folds must be >= 2")


@dataclass(slots=True)
class ExperimentConfig:
    """Configuration for a complete train/holdout experiment."""

    algorithms: list[str] = field(default_factory=lambda: ["rf", "hgbc", "lda", "qda", "ridge"])
    test_size: float = 0.2
    search: SearchConfig = field(default_factory=SearchConfig)
    output_dir: Path = Path("model_selector_results")

    def validate(self) -> None:
        if not 0.0 < self.test_size < 1.0:
            raise ValueError("test_size must be between 0 and 1")
        if not self.algorithms:
            raise ValueError("at least one algorithm must be selected")
        self.search.validate()

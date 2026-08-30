from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import time
from typing import Any, Iterable
import warnings

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from .config import SearchConfig
from .data import DatasetBundle
from .evaluation import classification_metrics
from .registry import EstimatorSpec, get_estimator_spec, prefixed_param_grid


@dataclass(slots=True)
class ModelSelectionResult:
    estimator_id: str
    estimator_name: str
    best_cv_score: float
    best_params: dict[str, Any]
    holdout_metrics: dict[str, float | None]
    search_seconds: float
    cv_folds_used: int
    fitted_model: BaseEstimator
    failed_candidates: int = 0

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record.pop("fitted_model")
        return record


def _effective_cv_folds(y_train: Any, requested: int) -> int:
    _, counts = np.unique(np.asarray(y_train), return_counts=True)
    folds = min(requested, int(counts.min()))
    if folds < 2:
        raise ValueError("not enough samples in the smallest class for stratified cross-validation")
    return folds


def _scaler_for(spec: EstimatorSpec, config: SearchConfig):
    if config.scaling == "none":
        return "passthrough"
    if config.scaling == "standard":
        return StandardScaler()
    if config.scaling == "minmax":
        return MinMaxScaler()
    if spec.preprocess == "standard":
        return StandardScaler()
    if spec.preprocess == "minmax":
        return MinMaxScaler()
    return "passthrough"


def build_pipeline(spec: EstimatorSpec, config: SearchConfig) -> Pipeline:
    return Pipeline([
        ("preprocess", _scaler_for(spec, config)),
        ("estimator", spec.factory(config.random_state)),
    ])


def run_model_selection(
    estimator: str | EstimatorSpec,
    dataset: DatasetBundle,
    config: SearchConfig | None = None,
) -> ModelSelectionResult:
    """Select hyperparameters using training-only CV, then evaluate once on the holdout."""
    config = config or SearchConfig()
    config.validate()
    spec = get_estimator_spec(estimator) if isinstance(estimator, str) else estimator
    folds = _effective_cv_folds(dataset.y_train, config.cv_folds)
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=config.random_state)
    pipeline = build_pipeline(spec, config)

    search = GridSearchCV(
        pipeline,
        prefixed_param_grid(spec),
        scoring=config.scoring,
        cv=cv,
        n_jobs=config.n_jobs,
        refit=config.refit,
        error_score=config.error_score,
        return_train_score=False,
    )

    started = time.perf_counter()
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        search.fit(dataset.X_train, dataset.y_train)
    elapsed = time.perf_counter() - started

    if not hasattr(search, "best_estimator_"):
        raise RuntimeError(f"no valid candidate was found for {spec.id}")

    scores = np.asarray(search.cv_results_["mean_test_score"], dtype=float)
    failed_candidates = int(np.isnan(scores).sum())
    best_params = {
        key.removeprefix("estimator__"): value
        for key, value in search.best_params_.items()
    }
    holdout = classification_metrics(search.best_estimator_, dataset.X_test, dataset.y_test)

    return ModelSelectionResult(
        estimator_id=spec.id,
        estimator_name=spec.name,
        best_cv_score=float(search.best_score_),
        best_params=best_params,
        holdout_metrics=holdout,
        search_seconds=float(elapsed),
        cv_folds_used=folds,
        fitted_model=search.best_estimator_,
        failed_candidates=failed_candidates,
    )


def run_model_suite(
    estimators: Iterable[str],
    dataset: DatasetBundle,
    config: SearchConfig | None = None,
) -> list[ModelSelectionResult]:
    results = [run_model_selection(estimator, dataset, config) for estimator in estimators]
    return sorted(results, key=lambda item: item.best_cv_score, reverse=True)

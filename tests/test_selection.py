import pandas as pd
from sklearn.datasets import load_breast_cancer, load_iris

from classifier_generator.config import SearchConfig
from classifier_generator.data import split_dataset
from classifier_generator.registry import EstimatorSpec
from classifier_generator.selection import run_model_selection
from sklearn.linear_model import LogisticRegression


def tiny_spec():
    return EstimatorSpec(
        id="tiny_logreg",
        name="Tiny Logistic Regression",
        factory=lambda seed: LogisticRegression(random_state=seed, max_iter=500),
        param_grid=[{"C": [0.1, 1.0]}],
        preprocess="standard",
    )


def test_binary_selection_runs_without_holdout_selection():
    raw = load_breast_cancer(as_frame=True)
    bundle = split_dataset(raw.data, raw.target, test_size=0.2, random_state=3)
    result = run_model_selection(tiny_spec(), bundle, SearchConfig(cv_folds=3, n_jobs=1, random_state=3))
    assert result.best_params["C"] in {0.1, 1.0}
    assert 0.0 <= result.best_cv_score <= 1.0
    assert 0.0 <= result.holdout_metrics["accuracy"] <= 1.0
    assert result.cv_folds_used == 3


def test_multiclass_metrics_run():
    raw = load_iris(as_frame=True)
    bundle = split_dataset(raw.data, raw.target, test_size=0.2, random_state=3)
    result = run_model_selection(tiny_spec(), bundle, SearchConfig(cv_folds=3, n_jobs=1, random_state=3))
    assert result.holdout_metrics["f1"] is not None
    assert result.holdout_metrics["roc_auc"] is not None


def test_binary_string_labels_are_supported():
    raw = load_breast_cancer(as_frame=True)
    labels = raw.target.map({0: "no", 1: "yes"})
    bundle = split_dataset(raw.data, labels, test_size=0.2, random_state=5)
    result = run_model_selection(tiny_spec(), bundle, SearchConfig(cv_folds=2, n_jobs=1, random_state=5))
    assert 0.0 <= result.holdout_metrics["precision"] <= 1.0

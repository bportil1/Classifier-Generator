from pathlib import Path
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression

from classifier_generator.config import SearchConfig
from classifier_generator.data import split_dataset
from classifier_generator.registry import EstimatorSpec
from classifier_generator.reporting import save_suite_results
from classifier_generator.selection import run_model_selection


def test_reporting_writes_json_csv_and_model(tmp_path: Path):
    raw = load_breast_cancer(as_frame=True)
    bundle = split_dataset(raw.data, raw.target, test_size=0.2, random_state=4)
    spec = EstimatorSpec("lr", "Logistic Regression", lambda seed: LogisticRegression(random_state=seed, max_iter=500), [{"C": [1.0]}], "standard")
    result = run_model_selection(spec, bundle, SearchConfig(cv_folds=2, n_jobs=1))
    save_suite_results([result], tmp_path)
    assert (tmp_path / "results.json").exists()
    assert (tmp_path / "results.csv").exists()
    assert (tmp_path / "models" / "lr.joblib").exists()

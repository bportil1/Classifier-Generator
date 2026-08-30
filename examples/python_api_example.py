"""Minimal API example using a built-in sklearn dataset."""
from sklearn.datasets import load_iris

from classifier_generator import SearchConfig, run_model_suite, split_dataset

raw = load_iris(as_frame=True)
data = split_dataset(raw.data, raw.target, test_size=0.2, random_state=42)
results = run_model_suite(
    ["lda", "qda", "gnb"],
    data,
    SearchConfig(cv_folds=3, n_jobs=1, random_state=42),
)

for result in results:
    print(
        result.estimator_id,
        f"cv={result.best_cv_score:.3f}",
        f"holdout={result.holdout_metrics['accuracy']:.3f}",
    )

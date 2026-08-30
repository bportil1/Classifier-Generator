# Classifier Generator — Refactored

A reusable scikit-learn **model selection and evaluation engine** extracted from the original Classifier Generator prototype.

The refactor keeps the original purpose—compare multiple classifiers and their hyperparameters—but removes the requirement that data loading, optimization, plotting, reporting, and UI behavior all live in the same script.

## Design

```text
CSV / DataFrame
      ↓
stratified train / holdout split
      ↓
training set only
      ↓
scikit-learn Pipeline
(preprocessing + estimator)
      ↓
Stratified CV + GridSearchCV
      ↓
selected candidate
      ↓
refit on all training data
      ↓
ONE final holdout evaluation
      ↓
JSON / CSV / serialized model
```

The holdout test set is never passed into the hyperparameter search.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

For tests:

```bash
pip install -e '.[test]'
pytest -q
```

## CLI

List classifiers:

```bash
classifier-generator list
```

Run selection:

```bash
classifier-generator run data.csv \
  --label label \
  --algorithms rf hgbc lda qda ridge \
  --cv 5 \
  --scoring accuracy \
  --output results/run_001
```

Multiple compatible CSV files can be concatenated by passing more than one path.

## Python API

```python
from classifier_generator import SearchConfig, load_csv_dataset, split_dataset, run_model_suite

X, y = load_csv_dataset(["data.csv"], label_column="label")
data = split_dataset(X, y, test_size=0.2, random_state=42)

results = run_model_suite(
    ["rf", "hgbc", "lda"],
    data,
    SearchConfig(cv_folds=5, scoring="accuracy", n_jobs=-1),
)
```

## Supported registry entries

The registry currently includes the original supervised classifier family: KNN, SVC, NuSVC, Gaussian Process, Decision Tree, Random Forest, HistGradientBoosting, AdaBoost, QDA, LDA, MLP, Ridge, Passive Aggressive, SGD, Extra Trees, Gaussian/Multinomial/Complement/Bernoulli Naive Bayes, and LinearSVC.

The old Isolation Forest entry is intentionally not treated as a supervised classifier because it is an anomaly detector rather than a drop-in classifier for this workflow.

## Important changes from the prototype

- Hyperparameters are selected using **training-only cross-validation**.
- Preprocessing is inside the sklearn `Pipeline`, so scaling is fitted separately inside each CV fold and then on the full training set for final refit.
- The holdout test set is used only for final evaluation.
- The LDA optimized path now actually selects LDA rather than calling QDA.
- One parallelism layer is used (`GridSearchCV.n_jobs`); individual parallel estimators are configured with `n_jobs=1` where applicable.
- No global warning suppression.
- Importing the package does not execute an experiment or open Tkinter.
- t-SNE is not part of model evaluation. Visualization can be layered on later without changing model selection semantics.
- Search spaces are declarative in `registry.py` rather than repeated optimization functions.
- Output is machine-readable JSON/CSV plus serialized sklearn pipelines.

## Legacy code

The three original Python files are retained under `legacy/` for provenance and behavioral comparison. They are not imported by the refactored package.

See `REFACTOR_NOTES.md` for details.

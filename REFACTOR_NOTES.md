# Refactor Notes

## Scope

This refactor targets the reusable supervised model-selection core of the supplied Classifier Generator repository. It is not yet a PAH or Modeling Lab integration.

## Problems addressed

### 1. Test-set leakage during KNN selection

The legacy KNN optimization directly predicted `X_test` for each candidate and selected the candidate with the highest `y_test` accuracy. That makes the nominal test set part of model selection.

The new engine performs all candidate selection using cross-validation on `X_train` only. The holdout is evaluated after selection.

### 2. Scaling semantics

The legacy `scale_data` method fitted a scaler to training data but only assigned transformed values back into `train_data`; the test data was not transformed consistently. It also passed test data as the positional `y` argument to `fit_transform`, which does not make the scaler learn a train/test mapping.

Scaling is now a Pipeline step. This also prevents CV-fold preprocessing leakage.

### 3. LDA dispatch bug

The legacy optimized dispatch contains:

```python
elif alg == 'lda':
    model, best_rec = generate_qda(...)
```

The registry constructs a real `LinearDiscriminantAnalysis` estimator for `lda`.

### 4. t-SNE comparability

The legacy code fits t-SNE independently to train and test sets. Those embeddings have unrelated coordinate systems and should not be treated as a shared 2-D feature space for classifier evaluation.

The refactored model-selection engine removes t-SNE from the evaluation path entirely. A later visualization layer can fit an explicit visualization workflow without changing the model-selection data.

### 5. Repeated optimization code

The original `classifier_helper.py` contains many estimator-specific nested loops with almost identical CV/select logic. Those are represented as declarative `EstimatorSpec` entries consumed by a single `GridSearchCV` implementation.

### 6. Nested parallelism

The original combines `ThreadPoolExecutor` with estimators/CV using `n_jobs=-1`. This can create substantial CPU oversubscription. The refactor delegates parallel candidate/CV work to GridSearchCV and keeps internally parallel estimators at one worker.

### 7. Import side effects

The original utility script calls `main()` unconditionally at module import. The refactored package has no such side effects and exposes an explicit CLI and Python API.

### 8. Summary-column state bug

The legacy `learning.__init__` assigns the result of `get_summary_column_labels(...)` to `self.summary_cols`, but that method does not return the generated list; it only mutates `self.summary_cols`. The assignment therefore replaces the list with `None`. The new reporting layer returns explicit result records and does not depend on this hidden mutable state.

## Compatibility decisions

The goal is semantic continuity, not line-for-line translation. Most original estimator families and their search-space intent are represented in the registry.

Two legacy KNN details are not silently reproduced: Mahalanobis distance with a data-dependent inverse covariance matrix, and custom inverse-distance weighting callables. These can be added later as explicit registered search variants if they are required for a study; silently rebuilding covariance-dependent parameters inside a static grid would make the new abstraction misleading.

Isolation Forest is left outside the supervised classifier registry because it is an anomaly-detection estimator with different target/evaluation semantics.

## Next architectural step

For Modeling Lab, this package should remain an independent engine. PAH should call a thin adapter around:

- estimator discovery (`list_estimators`)
- dataset/config construction
- `run_model_selection` / `run_model_suite`
- result/model loading

The engine itself should not import PAH.

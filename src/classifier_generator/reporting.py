from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import joblib
import pandas as pd

from .selection import ModelSelectionResult


def save_suite_results(results: Iterable[ModelSelectionResult], output_dir: str | Path) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    results = list(results)

    records = [result.to_record() for result in results]
    (output / "results.json").write_text(json.dumps(records, indent=2, default=str), encoding="utf-8")

    flat_rows = []
    for result in results:
        row = {
            "estimator_id": result.estimator_id,
            "estimator_name": result.estimator_name,
            "best_cv_score": result.best_cv_score,
            "best_params": json.dumps(result.best_params, default=str, sort_keys=True),
            "search_seconds": result.search_seconds,
            "cv_folds_used": result.cv_folds_used,
            "failed_candidates": result.failed_candidates,
        }
        row.update({f"holdout_{k}": v for k, v in result.holdout_metrics.items()})
        flat_rows.append(row)
    pd.DataFrame(flat_rows).to_csv(output / "results.csv", index=False)

    models_dir = output / "models"
    models_dir.mkdir(exist_ok=True)
    for result in results:
        joblib.dump(result.fitted_model, models_dir / f"{result.estimator_id}.joblib")

    return output

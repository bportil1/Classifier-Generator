from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import SearchConfig
from .data import load_csv_dataset, split_dataset
from .registry import list_estimators
from .reporting import save_suite_results
from .selection import run_model_suite


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="classifier-generator", description="Cross-validated scikit-learn model selector")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="List registered classifiers")

    run = sub.add_parser("run", help="Run model selection on one or more CSV files")
    run.add_argument("csv", nargs="+", help="CSV file(s) containing features and a label column")
    run.add_argument("--label", default="label", help="Label column name")
    run.add_argument("--algorithms", nargs="+", default=["rf", "hgbc", "lda", "qda", "ridge"])
    run.add_argument("--test-size", type=float, default=0.2)
    run.add_argument("--cv", type=int, default=5)
    run.add_argument("--scoring", default="accuracy")
    run.add_argument("--scaling", choices=["auto", "none", "standard", "minmax"], default="auto")
    run.add_argument("--n-jobs", type=int, default=-1)
    run.add_argument("--random-state", type=int, default=42)
    run.add_argument("--output", default="model_selector_results")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "list":
        for spec in list_estimators():
            print(f"{spec.id:8s} {spec.name}")
        return 0

    X, y = load_csv_dataset(args.csv, label_column=args.label)
    dataset = split_dataset(X, y, test_size=args.test_size, random_state=args.random_state, label_name=args.label)
    config = SearchConfig(
        cv_folds=args.cv,
        scoring=args.scoring,
        n_jobs=args.n_jobs,
        random_state=args.random_state,
        scaling=args.scaling,
    )
    results = run_model_suite(args.algorithms, dataset, config)
    output = save_suite_results(results, args.output)

    print(f"Results written to: {output}")
    for rank, result in enumerate(results, start=1):
        acc = result.holdout_metrics["accuracy"]
        print(f"{rank:2d}. {result.estimator_id:8s} CV={result.best_cv_score:.4f} holdout_accuracy={acc:.4f}")
    return 0

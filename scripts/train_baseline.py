from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from plastech.modeling import (
    DEFAULT_TARGET,
    random_forest_diagnostics,
    run_baseline_models,
    target_candidates,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train baseline regression models.")
    parser.add_argument("--dataset", default="processed_main")
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument(
        "--include-ridge",
        action="store_true",
        help="Also run a scaled Ridge model. This can warn on ill-conditioned data.",
    )
    parser.add_argument(
        "--show-targets",
        action="store_true",
        help="Print inspected target candidates before training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.show_targets:
        print("Target candidates:")
        print(target_candidates().to_string(index=False))
        print()

    try:
        if args.include_ridge:
            metrics, prepared, fitted_models = run_baseline_models(
                dataset_name=args.dataset,
                target_col=args.target,
                test_size=args.test_size,
                random_state=args.random_state,
                n_estimators=args.n_estimators,
                include_ridge=True,
            )
            feature_importance = None
        else:
            diagnostics = random_forest_diagnostics(
                dataset_name=args.dataset,
                target_col=args.target,
                test_size=args.test_size,
                random_state=args.random_state,
                n_estimators=args.n_estimators,
            )
            metrics = diagnostics.metrics
            prepared = diagnostics.prepared
            fitted_models = {}
            feature_importance = diagnostics.feature_importance
    except ModuleNotFoundError as exc:
        print(exc)
        raise SystemExit(1) from exc

    print(f"dataset: {args.dataset}")
    print(f"target: {prepared.target_col}")
    print(f"rows: {len(prepared.y)}")
    print(f"features: {len(prepared.feature_columns)}")
    print()
    print("Dropped columns:")
    for reason, columns in prepared.dropped_columns.items():
        print(f"- {reason}: {len(columns)}")
    print()
    print("Metrics:")
    print(metrics.to_string(index=False))

    random_forest = fitted_models.get("random_forest")
    if feature_importance is not None:
        print()
        print("Top RandomForest feature importances:")
        for _, row in feature_importance.head(20).iterrows():
            print(f"{row['display_name']}: {row['importance']:.6f}")
    elif random_forest is not None:
        importance = sorted(
            zip(prepared.feature_columns, random_forest.feature_importances_),
            key=lambda item: item[1],
            reverse=True,
        )
        print()
        print("Top RandomForest feature importances:")
        for column, score in importance[:20]:
            print(f"{column}: {score:.6f}")


if __name__ == "__main__":
    main()

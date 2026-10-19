from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "plastech-matplotlib"))

from plastech.inverse_design import (  # noqa: E402
    DEFAULT_DESIGN_BRIEFS,
    ROLE_BASED_DESIGN_BRIEFS,
    brief_specifications_frame,
    candidate_summary,
    parent_formulation_deltas,
    run_inverse_design_for_brief,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inverse design simulations.")
    parser.add_argument("--dataset", default="processed_main")
    parser.add_argument("--model", default="lightgbm", choices=["random_forest", "xgboost", "lightgbm"])
    parser.add_argument("--concept", default="all", help="Design concept name, or 'all'.")
    parser.add_argument(
        "--design-mode",
        default="legacy",
        choices=["legacy", "role_based"],
        help="Use the legacy all-in-one specification or the role-based specification.",
    )
    parser.add_argument("--n-estimators", type=int, default=800)
    parser.add_argument("--seed-count", type=int, default=30)
    parser.add_argument("--partners-per-seed", type=int, default=5)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--max-material-count",
        type=int,
        help="Optional upper limit for the provisional non-zero material count.",
    )
    parser.add_argument(
        "--require-identified-primary-materials",
        action="store_true",
        help="Exclude candidates whose primary PVC or plasticizer cannot be identified.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Optional directory to write recommended candidate CSV files.",
    )
    return parser.parse_args()


def selected_briefs(concept: str, design_mode: str):
    briefs = ROLE_BASED_DESIGN_BRIEFS if design_mode == "role_based" else DEFAULT_DESIGN_BRIEFS
    if concept == "all":
        return briefs.values()
    if concept not in briefs:
        available = ", ".join(briefs)
        raise KeyError(f"Unknown concept: {concept}. Available concepts: {available}")
    return [briefs[concept]]


def main() -> None:
    args = parse_args()
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    for brief in selected_briefs(args.concept, args.design_mode):
        print("=" * 80)
        print(f"concept: {brief.name}")
        print(f"description: {brief.description}")
        print()
        print("property roles:")
        print(brief_specifications_frame(brief).to_string(index=False))
        print()

        result = run_inverse_design_for_brief(
            brief,
            dataset_name=args.dataset,
            model_name=args.model,
            n_estimators=args.n_estimators,
            seed_count=args.seed_count,
            partners_per_seed=args.partners_per_seed,
            top_n=args.top_n,
            random_state=args.random_state,
            max_material_count=args.max_material_count,
            require_identified_primary_materials=args.require_identified_primary_materials,
        )
        print(f"library rows: {len(result.library):,}")
        print(f"virtual candidates: {len(result.virtual_candidates):,}")
        print(f"pareto candidates: {len(result.pareto_candidates):,}")
        print(f"recommended candidates: {len(result.recommended_candidates):,}")
        print()
        print(candidate_summary(result.recommended_candidates, result.forward_models, top_n=args.top_n).to_string(index=False))
        if "predicted_unmet_preferences" in result.recommended_candidates:
            unmet = result.recommended_candidates[["row_id", "predicted_meets_preferences", "predicted_unmet_preferences"]]
            print()
            print("preference status:")
            print(unmet.to_string(index=False))
        if "predicted_meets_requirements" in result.recommended_candidates:
            status = result.recommended_candidates[
                [
                    "row_id",
                    "predicted_meets_objectives",
                    "predicted_meets_requirements",
                    "predicted_unmet_management_targets",
                    "predicted_unmet_constraints",
                    "predicted_unmet_monitors",
                ]
            ]
            print()
            print("role-based status:")
            print(status.to_string(index=False))
        print()

        if args.output_dir is not None:
            safe_name = brief.name.replace("/", "_").replace(" ", "_")
            output_path = args.output_dir / f"{safe_name}_recommended_candidates.csv"
            result.recommended_candidates.to_csv(output_path, index=False, encoding="utf-8-sig")
            print(f"wrote: {output_path}")

            summary_path = args.output_dir / f"{safe_name}_candidate_summary.csv"
            candidate_summary(result.recommended_candidates, result.forward_models, top_n=args.top_n).to_csv(
                summary_path,
                index=False,
                encoding="utf-8-sig",
            )
            print(f"wrote: {summary_path}")

            delta_path = args.output_dir / f"{safe_name}_parent_formulation_deltas.csv"
            parent_formulation_deltas(
                result.recommended_candidates,
                result.library,
                dataset_name=args.dataset,
            ).to_csv(delta_path, index=False, encoding="utf-8-sig")
            print(f"wrote: {delta_path}")


if __name__ == "__main__":
    main()

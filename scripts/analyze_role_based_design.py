from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "plastech-matplotlib"))

from plastech.inverse_design import (  # noqa: E402
    DEFAULT_DESIGN_BRIEFS,
    ROLE_BASED_DESIGN_BRIEFS,
    DesignBrief,
    ForwardModelSet,
    add_design_brief_status,
    brief_specifications_frame,
    build_candidate_library,
    candidate_summary,
    fit_forward_models,
    generate_interpolated_candidates,
    parent_formulation_deltas,
    rank_seed_candidates,
    score_design_preferences,
    select_diverse_candidates,
)


PROPERTY_NAMES = {
    "S03010_030010": "硬度",
    "S06010_060030": "100%モジュラス",
    "S06020_060030": "引張強さ",
    "S06060_060030": "引張伸び",
    "S13010_130010": "脆化温度",
    "S01010_010010": "比重",
}


@dataclass(frozen=True)
class AnalysisResult:
    mode: str
    brief: DesignBrief
    models: ForwardModelSet
    library: pd.DataFrame
    candidates: pd.DataFrame
    recommended: pd.DataFrame
    historical_references: pd.DataFrame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare legacy and role-based inverse design.")
    parser.add_argument("--dataset", default="processed_main")
    parser.add_argument("--model", default="lightgbm", choices=["random_forest", "xgboost", "lightgbm"])
    parser.add_argument("--n-estimators", type=int, default=800)
    parser.add_argument("--seed-count", type=int, default=30)
    parser.add_argument("--partners-per-seed", type=int, default=5)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "role_based_design",
    )
    return parser.parse_args()


def unique_model_specifications(*briefs: DesignBrief):
    by_column = {}
    for brief in briefs:
        for specification in brief.all_specifications:
            by_column.setdefault(specification.column, specification)
    return tuple(by_column.values())


def subset_models(full: ForwardModelSet, brief: DesignBrief) -> ForwardModelSet:
    specifications = brief.all_specifications
    return ForwardModelSet(
        specifications=specifications,
        models={spec.column: full.models[spec.column] for spec in specifications},
        prepared={spec.column: full.prepared[spec.column] for spec in specifications},
    )


def formulation_material_count(candidates: pd.DataFrame, models: ForwardModelSet) -> pd.Series:
    feature_columns = sorted(
        {
            column
            for prepared in models.prepared.values()
            for column in prepared.feature_columns
        }
    )
    matrix = candidates.reindex(columns=feature_columns, fill_value=0.0)
    matrix = matrix.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return matrix.abs().gt(1e-12).sum(axis=1)


def run_mode(
    mode: str,
    brief: DesignBrief,
    full_models: ForwardModelSet,
    *,
    dataset_name: str,
    seed_count: int,
    partners_per_seed: int,
    top_n: int,
    random_state: int,
) -> AnalysisResult:
    models = subset_models(full_models, brief)
    library = build_candidate_library(
        models,
        dataset_name=dataset_name,
        required_observed=brief.required_specifications,
    )
    seeds = rank_seed_candidates(library, brief.main_goal)
    virtual = generate_interpolated_candidates(
        library,
        seeds,
        models,
        seed_count=seed_count,
        partners_per_seed=partners_per_seed,
        random_state=random_state,
        scoring_specifications=brief.objectives,
    )
    scored = score_design_preferences(virtual, brief.objectives)
    scored = add_design_brief_status(scored, brief)
    scored["formulation_material_count"] = formulation_material_count(scored, models)
    recommended = select_diverse_candidates(scored, top_n=top_n, prefer_pareto=True)
    recommended.insert(0, "rank", np.arange(1, len(recommended) + 1))
    recommended.insert(0, "mode", mode)

    historical = score_design_preferences(library, brief.objectives, value_prefix="observed")
    historical = add_design_brief_status(historical, brief, value_prefix="observed")
    historical["formulation_material_count"] = formulation_material_count(historical, models)
    historical = historical[
        historical["observed_meets_objectives_and_requirements"]
    ].copy()
    historical = select_diverse_candidates(historical, top_n=5, prefer_pareto=True)
    historical.insert(0, "rank", np.arange(1, len(historical) + 1))
    historical.insert(0, "mode", mode)

    return AnalysisResult(
        mode=mode,
        brief=brief,
        models=models,
        library=library,
        candidates=scored,
        recommended=recommended,
        historical_references=historical,
    )


def mode_metrics(result: AnalysisResult) -> dict[str, object]:
    candidates = result.candidates
    top5 = result.recommended.head(5)
    return {
        "mode": result.mode,
        "library_rows": len(result.library),
        "virtual_candidates": len(candidates),
        "objective_pass_candidates": int(candidates["predicted_meets_objectives"].sum()),
        "requirement_pass_candidates": int(candidates["predicted_meets_requirements"].sum()),
        "objective_and_requirement_pass_candidates": int(
            candidates["predicted_meets_objectives_and_requirements"].sum()
        ),
        "pareto_candidates": int(candidates["pareto_optimal"].sum()),
        "top5_objective_and_requirement_pass": int(
            top5["predicted_meets_objectives_and_requirements"].sum()
        ),
        "top5_material_series": int(
            top5[["primary_pvc", "primary_plasticizer"]].drop_duplicates().shape[0]
        ),
        "top5_material_count_min": int(top5["formulation_material_count"].min()),
        "top5_material_count_median": float(top5["formulation_material_count"].median()),
        "top5_material_count_max": int(top5["formulation_material_count"].max()),
        "best_design_score": float(result.recommended["design_score"].min()),
    }


def readable_candidate_table(result: AnalysisResult) -> pd.DataFrame:
    columns = [
        "mode",
        "rank",
        "row_id",
        "parent_a",
        "parent_b",
        "parent_a_fraction",
        "primary_pvc",
        "primary_plasticizer",
        "formulation_material_count",
        "design_score",
        "pareto_optimal",
        "predicted_meets_objectives",
        "predicted_meets_management_targets",
        "predicted_meets_constraints",
        "predicted_meets_requirements",
        "predicted_meets_objectives_and_requirements",
        "predicted_unmet_management_targets",
        "predicted_unmet_constraints",
        "predicted_unmet_monitors",
    ]
    columns.extend(f"predicted__{column}" for column in PROPERTY_NAMES)
    available = [column for column in columns if column in result.recommended]
    table = result.recommended.loc[:, available].copy()
    return table.rename(
        columns={f"predicted__{column}": f"予測_{name}" for column, name in PROPERTY_NAMES.items()}
    )


def eligible_role_based_candidates(
    result: AnalysisResult,
    *,
    max_material_count: int | None,
) -> pd.DataFrame:
    candidates = result.candidates.copy()
    primary_materials_identified = (
        ~candidates["primary_pvc"].str.startswith("(no ")
        & ~candidates["primary_plasticizer"].str.startswith("(no ")
    )
    mask = candidates["predicted_meets_objectives_and_requirements"] & primary_materials_identified
    if max_material_count is not None:
        mask &= candidates["formulation_material_count"].le(max_material_count)
    return candidates.loc[mask].copy()


def material_count_comparison(result: AnalysisResult) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    practical_shortlist = pd.DataFrame()
    for label, limit in (("制限なし", None), ("12材料以下", 12), ("10材料以下", 10), ("8材料以下", 8)):
        eligible = eligible_role_based_candidates(result, max_material_count=limit)
        if not eligible.empty:
            eligible = score_design_preferences(eligible, result.brief.objectives)
        selected = select_diverse_candidates(
            eligible,
            top_n=min(5, len(eligible)),
            prefer_pareto=True,
        ) if not eligible.empty else eligible
        rows.append(
            {
                "material_count_condition": label,
                "max_material_count": limit,
                "eligible_candidates": len(eligible),
                "material_series": int(
                    eligible[["primary_pvc", "primary_plasticizer"]]
                    .drop_duplicates()
                    .shape[0]
                ),
                "selected_candidates": len(selected),
                "selected_material_count_min": (
                    int(selected["formulation_material_count"].min())
                    if not selected.empty
                    else np.nan
                ),
                "selected_material_count_max": (
                    int(selected["formulation_material_count"].max())
                    if not selected.empty
                    else np.nan
                ),
                "selected_predicted_brittleness_min": (
                    float(selected["predicted__S13010_130010"].min())
                    if not selected.empty
                    else np.nan
                ),
                "selected_predicted_brittleness_max": (
                    float(selected["predicted__S13010_130010"].max())
                    if not selected.empty
                    else np.nan
                ),
            }
        )
        if limit == 10:
            practical_shortlist = selected.copy()

    practical_shortlist.insert(0, "rank", np.arange(1, len(practical_shortlist) + 1))
    practical_shortlist.insert(0, "mode", "role_based_max_10_materials")
    return pd.DataFrame(rows), practical_shortlist


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    legacy_brief = DEFAULT_DESIGN_BRIEFS["柔軟・低温用途"]
    role_based_brief = ROLE_BASED_DESIGN_BRIEFS["柔軟・低温用途"]
    specifications = unique_model_specifications(legacy_brief, role_based_brief)

    print(f"[1/4] fitting {len(specifications)} shared forward models", flush=True)
    full_models = fit_forward_models(
        specifications,
        dataset_name=args.dataset,
        model_name=args.model,
        n_estimators=args.n_estimators,
        random_state=args.random_state,
    )

    print("[2/4] running legacy four-property design", flush=True)
    legacy = run_mode(
        "legacy_4_properties",
        legacy_brief,
        full_models,
        dataset_name=args.dataset,
        seed_count=args.seed_count,
        partners_per_seed=args.partners_per_seed,
        top_n=args.top_n,
        random_state=args.random_state,
    )
    print("[3/4] running role-based design", flush=True)
    role_based = run_mode(
        "role_based",
        role_based_brief,
        full_models,
        dataset_name=args.dataset,
        seed_count=args.seed_count,
        partners_per_seed=args.partners_per_seed,
        top_n=args.top_n,
        random_state=args.random_state,
    )

    print("[4/4] writing comparison outputs", flush=True)
    metrics = pd.DataFrame([mode_metrics(legacy), mode_metrics(role_based)])
    metrics.to_csv(args.output_dir / "mode_comparison.csv", index=False, encoding="utf-8-sig")

    candidate_tables = pd.concat(
        [readable_candidate_table(legacy), readable_candidate_table(role_based)],
        ignore_index=True,
    )
    candidate_tables.to_csv(
        args.output_dir / "recommended_candidates.csv",
        index=False,
        encoding="utf-8-sig",
    )

    role_definitions = brief_specifications_frame(role_based_brief)
    role_definitions.to_csv(
        args.output_dir / "role_definitions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    complexity_comparison, practical_shortlist = material_count_comparison(role_based)
    complexity_comparison.to_csv(
        args.output_dir / "material_count_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )
    practical_result = AnalysisResult(
        mode="role_based_max_10_materials",
        brief=role_based.brief,
        models=role_based.models,
        library=role_based.library,
        candidates=role_based.candidates,
        recommended=practical_shortlist,
        historical_references=role_based.historical_references,
    )
    readable_candidate_table(practical_result).to_csv(
        args.output_dir / "practical_shortlist_max_10_materials.csv",
        index=False,
        encoding="utf-8-sig",
    )

    historical = candidate_summary(
        role_based.historical_references,
        role_based.models,
        top_n=5,
    )
    historical.insert(0, "rank", np.arange(1, len(historical) + 1))
    historical.to_csv(
        args.output_dir / "historical_reference_candidates.csv",
        index=False,
        encoding="utf-8-sig",
    )

    parent_formulation_deltas(
        practical_shortlist.head(5),
        role_based.library,
        dataset_name=args.dataset,
    ).to_csv(
        args.output_dir / "role_based_parent_formulation_deltas.csv",
        index=False,
        encoding="utf-8-sig",
    )

    overlap = set(legacy.recommended.head(5)["row_id"]) & set(
        role_based.recommended.head(5)["row_id"]
    )
    parameters = {
        "dataset": args.dataset,
        "model": args.model,
        "n_estimators": args.n_estimators,
        "seed_count": args.seed_count,
        "partners_per_seed": args.partners_per_seed,
        "top_n": args.top_n,
        "random_state": args.random_state,
        "top5_common_candidates": len(overlap),
        "top5_jaccard_overlap": len(overlap) / (10 - len(overlap)),
    }
    (args.output_dir / "analysis_parameters.json").write_text(
        json.dumps(parameters, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(metrics.to_string(index=False))
    print(f"wrote: {args.output_dir}")


if __name__ == "__main__":
    main()

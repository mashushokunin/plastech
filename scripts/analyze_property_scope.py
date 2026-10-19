from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "plastech-matplotlib"))

from plastech.data_catalog import read_dataset  # noqa: E402
from plastech.inverse_design import (  # noqa: E402
    DEFAULT_DESIGN_BRIEFS,
    DesignBrief,
    ForwardModelSet,
    TargetSpecification,
    add_preference_status,
    build_candidate_library,
    fit_forward_models,
    generate_interpolated_candidates,
    pareto_optimal_mask,
    rank_seed_candidates,
    score_design_preferences,
    select_diverse_candidates,
)
from plastech.modeling import evaluate_model_grouped  # noqa: E402
from plastech.target_selection import cross_validated_forward_predictions  # noqa: E402


PROPERTY_ORDER = (
    "S03010_030010",
    "S06010_060030",
    "S06020_060030",
    "S06060_060030",
    "S13010_130010",
    "S01010_010010",
)

PROPERTY_NAMES = {
    "S03010_030010": "硬度 HDA (0s)",
    "S06010_060030": "100%モジュラス",
    "S06020_060030": "引張強さ",
    "S06060_060030": "引張伸び",
    "S13010_130010": "脆化温度",
    "S01010_010010": "比重",
}


@dataclass(frozen=True)
class ScopeScenario:
    concept: str
    scope: int
    properties: tuple[str, ...]

    @property
    def key(self) -> str:
        return f"{self.concept}__{self.scope}props"


SCENARIOS = (
    ScopeScenario("柔軟・低温用途", 2, ("S03010_030010", "S13010_130010")),
    ScopeScenario(
        "柔軟・低温用途",
        3,
        ("S03010_030010", "S13010_130010", "S06060_060030"),
    ),
    ScopeScenario(
        "柔軟・低温用途",
        4,
        ("S03010_030010", "S13010_130010", "S06060_060030", "S01010_010010"),
    ),
    ScopeScenario("硬さ・剛性用途", 2, ("S03010_030010", "S06010_060030")),
    ScopeScenario(
        "硬さ・剛性用途",
        3,
        ("S03010_030010", "S06010_060030", "S06020_060030"),
    ),
    ScopeScenario(
        "硬さ・剛性用途",
        4,
        ("S03010_030010", "S06010_060030", "S06020_060030", "S01010_010010"),
    ),
    ScopeScenario("バランス型軟質PVC", 2, ("S03010_030010", "S06010_060030")),
    ScopeScenario(
        "バランス型軟質PVC",
        3,
        ("S03010_030010", "S06010_060030", "S13010_130010"),
    ),
    ScopeScenario(
        "バランス型軟質PVC",
        4,
        ("S03010_030010", "S06010_060030", "S13010_130010", "S01010_010010"),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two-, three-, and four-property inverse design scopes.")
    parser.add_argument("--dataset", default="processed_main")
    parser.add_argument("--model", default="lightgbm", choices=["random_forest", "xgboost", "lightgbm"])
    parser.add_argument("--n-estimators", type=int, default=800)
    parser.add_argument("--evaluation-n-estimators", type=int, default=800)
    parser.add_argument("--grouped-n-estimators", type=int, default=400)
    parser.add_argument("--seed-count", type=int, default=30)
    parser.add_argument("--partners-per-seed", type=int, default=5)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "property_scope")
    return parser.parse_args()


def all_specifications() -> dict[str, TargetSpecification]:
    specs: dict[str, TargetSpecification] = {}
    for brief in DEFAULT_DESIGN_BRIEFS.values():
        for spec in brief.preferences:
            specs.setdefault(spec.column, spec)
    return specs


def scenario_brief(scenario: ScopeScenario) -> DesignBrief:
    source = DEFAULT_DESIGN_BRIEFS[scenario.concept]
    source_by_column = {spec.column: spec for spec in source.preferences}
    return DesignBrief(
        name=f"{scenario.concept}（{scenario.scope}物性）",
        description=source.description,
        preferences=tuple(source_by_column[column] for column in scenario.properties),
    )


def subset_forward_models(
    full: ForwardModelSet,
    specifications: Sequence[TargetSpecification],
) -> ForwardModelSet:
    specs = tuple(specifications)
    return ForwardModelSet(
        specifications=specs,
        models={spec.column: full.models[spec.column] for spec in specs},
        prepared={spec.column: full.prepared[spec.column] for spec in specs},
    )


def observed_pass(values: pd.Series, spec: TargetSpecification) -> pd.Series:
    if spec.objective == "target":
        return values.sub(spec.target).abs().le(spec.tolerance + 1e-12)
    if spec.objective == "lower_bound":
        return values.ge(spec.target)
    if spec.objective == "upper_bound":
        return values.le(spec.target)
    raise ValueError(f"Unknown objective: {spec.objective}")


def historical_funnel(
    source: pd.DataFrame,
    scenario: ScopeScenario,
    brief: DesignBrief,
) -> pd.DataFrame:
    numeric = {
        spec.column: pd.to_numeric(source[spec.column], errors="coerce").replace(0, np.nan)
        for spec in brief.preferences
    }
    rows: list[dict[str, object]] = []
    complete = pd.Series(True, index=source.index)
    passes = pd.Series(True, index=source.index)
    included: list[str] = []
    for step, spec in enumerate(brief.preferences, start=1):
        values = numeric[spec.column]
        complete &= values.notna()
        passes &= observed_pass(values, spec)
        included.append(PROPERTY_NAMES[spec.column])
        rows.append(
            {
                "concept": scenario.concept,
                "scope": scenario.scope,
                "step": step,
                "added_property": PROPERTY_NAMES[spec.column],
                "included_properties": " + ".join(included),
                "complete_rows": int(complete.sum()),
                "passing_rows": int((complete & passes).sum()),
                "pass_rate_among_complete": float((complete & passes).sum() / complete.sum()),
            }
        )
    return pd.DataFrame(rows)


def add_all_predictions(candidates: pd.DataFrame, full_models: ForwardModelSet) -> pd.DataFrame:
    result = candidates.copy()
    for column in PROPERTY_ORDER:
        prepared = full_models.prepared[column]
        matrix = result.reindex(columns=prepared.feature_columns, fill_value=0.0)
        matrix = matrix.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        result[f"predicted__{column}"] = full_models.models[column].predict(matrix)
    return result


def p90_screening_pass(
    candidates: pd.DataFrame,
    specifications: Sequence[TargetSpecification],
    p90_by_property: dict[str, float],
) -> pd.Series:
    result = pd.Series(True, index=candidates.index)
    for spec in specifications:
        values = pd.to_numeric(candidates[f"predicted__{spec.column}"], errors="coerce")
        error = p90_by_property[spec.column]
        if spec.objective == "target":
            passes = values.sub(error).ge(spec.target - spec.tolerance) & values.add(error).le(
                spec.target + spec.tolerance
            )
        elif spec.objective == "lower_bound":
            passes = values.sub(error).ge(spec.target)
        elif spec.objective == "upper_bound":
            passes = values.add(error).le(spec.target)
        else:
            raise ValueError(f"Unknown objective: {spec.objective}")
        result &= passes.fillna(False)
    return result


def target_region_metrics(
    predictions: pd.DataFrame,
    specification: TargetSpecification,
) -> dict[str, float | int | str]:
    actual = predictions["actual"]
    near_boundary = actual.sub(specification.target).abs().le(specification.tolerance)
    local = predictions.loc[near_boundary]
    return {
        "target_col": specification.column,
        "property": PROPERTY_NAMES[specification.column],
        "target": specification.target,
        "tolerance": specification.tolerance,
        "objective": specification.objective,
        "local_rows": len(local),
        "local_MAE": float(local["abs_error"].mean()),
        "local_p90_abs_error": float(local["abs_error"].quantile(0.9)),
        "local_bias": float(local["error"].mean()),
    }


def scenario_result(
    scenario: ScopeScenario,
    brief: DesignBrief,
    full_models: ForwardModelSet,
    p90_by_property: dict[str, float],
    *,
    dataset_name: str,
    seed_count: int,
    partners_per_seed: int,
    top_n: int,
    random_state: int,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    models = subset_forward_models(full_models, brief.preferences)
    library = build_candidate_library(models, dataset_name=dataset_name)
    seeds = rank_seed_candidates(library, brief.main_goal)
    virtual = generate_interpolated_candidates(
        library,
        seeds,
        models,
        seed_count=seed_count,
        partners_per_seed=partners_per_seed,
        random_state=random_state,
    )
    scored = score_design_preferences(virtual, brief.preferences)
    recommended = select_diverse_candidates(scored, top_n=top_n, prefer_pareto=True)
    recommended = add_preference_status(recommended, brief.preferences)
    recommended = add_all_predictions(recommended, full_models)
    recommended["p90_screening_meets_all"] = p90_screening_pass(
        recommended,
        brief.preferences,
        p90_by_property,
    )
    recommended.insert(0, "rank", np.arange(1, len(recommended) + 1))
    recommended.insert(0, "scope", scenario.scope)
    recommended.insert(0, "concept", scenario.concept)
    recommended.insert(0, "scenario", scenario.key)

    all_status = add_preference_status(scored, brief.preferences)
    all_p90 = p90_screening_pass(all_status, brief.preferences, p90_by_property)
    metrics = {
        "scenario": scenario.key,
        "concept": scenario.concept,
        "scope": scenario.scope,
        "properties": " + ".join(PROPERTY_NAMES[column] for column in scenario.properties),
        "library_rows": len(library),
        "virtual_candidates": len(scored),
        "virtual_meets_all": int(all_status["predicted_meets_preferences"].sum()),
        "virtual_meets_all_rate": float(all_status["predicted_meets_preferences"].mean()),
        "pareto_candidates": int(scored["pareto_optimal"].sum()),
        "recommended_top5_meets_all": int(recommended.head(5)["predicted_meets_preferences"].sum()),
        "recommended_top5_p90_screening_meets_all": int(
            recommended.head(5)["p90_screening_meets_all"].sum()
        ),
        "all_virtual_p90_screening_meets_all": int(all_p90.sum()),
        "top10_material_series": int(
            recommended.head(10)[["primary_pvc", "primary_plasticizer"]].drop_duplicates().shape[0]
        ),
        "best_design_score": float(recommended["design_score"].min()),
    }

    historical = seeds.head(10).copy()
    historical.insert(0, "rank", np.arange(1, len(historical) + 1))
    historical.insert(0, "scope", scenario.scope)
    historical.insert(0, "concept", scenario.concept)
    historical.insert(0, "scenario", scenario.key)
    return metrics, recommended, historical


def candidate_overlap(candidates: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for concept in candidates["concept"].drop_duplicates():
        subset = candidates[candidates["concept"] == concept]
        for left_scope, right_scope in ((2, 3), (3, 4), (2, 4)):
            for top_n in (5, 10):
                left = set(subset[subset["scope"] == left_scope].head(top_n)["row_id"])
                right = set(subset[subset["scope"] == right_scope].head(top_n)["row_id"])
                union = left | right
                rows.append(
                    {
                        "concept": concept,
                        "left_scope": left_scope,
                        "right_scope": right_scope,
                        "top_n": top_n,
                        "common_candidates": len(left & right),
                        "union_candidates": len(union),
                        "jaccard_overlap": len(left & right) / len(union) if union else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = read_dataset(args.dataset)

    specs_by_column = all_specifications()
    # The same property may have a different target by concept, but its forward model is shared.
    model_specs = tuple(specs_by_column[column] for column in PROPERTY_ORDER)

    print("[1/5] fitting six forward models", flush=True)
    full_models = fit_forward_models(
        model_specs,
        dataset_name=args.dataset,
        model_name=args.model,
        n_estimators=args.n_estimators,
        random_state=args.random_state,
    )

    print("[2/5] calculating OOF errors", flush=True)
    oof_rows: list[pd.DataFrame] = []
    target_region_rows: list[dict[str, object]] = []
    p90_by_property: dict[str, float] = {}
    all_target_specs = {
        (brief.name, spec.column): spec
        for brief in DEFAULT_DESIGN_BRIEFS.values()
        for spec in brief.preferences
    }
    for index, column in enumerate(PROPERTY_ORDER, start=1):
        print(f"  OOF {index}/{len(PROPERTY_ORDER)}: {PROPERTY_NAMES[column]}", flush=True)
        cv = cross_validated_forward_predictions(
            column,
            dataset_name=args.dataset,
            model_name=args.model,
            n_splits=5,
            n_estimators=args.evaluation_n_estimators,
            random_state=args.random_state,
        )
        metrics = cv.metrics.copy()
        metrics["property"] = PROPERTY_NAMES[column]
        oof_rows.append(metrics)
        p90_by_property[column] = float(metrics.iloc[0]["p90_abs_error"])
        for (concept, spec_column), spec in all_target_specs.items():
            if spec_column != column:
                continue
            row = target_region_metrics(cv.predictions, spec)
            row["concept"] = concept
            target_region_rows.append(row)

    oof_metrics = pd.concat(oof_rows, ignore_index=True)
    oof_metrics.to_csv(args.output_dir / "oof_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(target_region_rows).to_csv(
        args.output_dir / "target_region_metrics.csv", index=False, encoding="utf-8-sig"
    )

    print("[3/5] calculating grouped generalization errors", flush=True)
    grouped_rows: list[dict[str, object]] = []
    for index, column in enumerate(PROPERTY_ORDER, start=1):
        print(f"  grouped {index}/{len(PROPERTY_ORDER)}: {PROPERTY_NAMES[column]}", flush=True)
        grouped = evaluate_model_grouped(
            args.model,
            dataset_name=args.dataset,
            target_col=column,
            group_by="primary_pvc_plus_primary_plasticizer",
            random_state=args.random_state,
            n_splits=5,
            n_estimators=args.grouped_n_estimators,
        )
        grouped_rows.append(
            {
                "target_col": column,
                "property": PROPERTY_NAMES[column],
                "rows": len(grouped.prepared.y),
                "groups": len(grouped.group_summary),
                "mean_MAE": grouped.metrics["MAE"].mean(),
                "mean_RMSE": grouped.metrics["RMSE"].mean(),
                "mean_R2": grouped.metrics["R2"].mean(),
                "std_R2": grouped.metrics["R2"].std(),
                "worst_split_R2": grouped.metrics["R2"].min(),
                "max_abs_error": grouped.metrics["max_abs_error"].max(),
            }
        )
    pd.DataFrame(grouped_rows).to_csv(
        args.output_dir / "grouped_metrics.csv", index=False, encoding="utf-8-sig"
    )

    print("[4/5] calculating data funnels and correlations", flush=True)
    numeric = pd.DataFrame(
        {
            PROPERTY_NAMES[column]: pd.to_numeric(source[column], errors="coerce").replace(0, np.nan)
            for column in PROPERTY_ORDER
        }
    )
    numeric.corr(method="spearman", min_periods=100).to_csv(
        args.output_dir / "property_spearman_correlation.csv",
        encoding="utf-8-sig",
    )
    valid_counts = pd.DataFrame(
        {
            "property": numeric.columns,
            "valid_rows": [int(numeric[column].notna().sum()) for column in numeric.columns],
        }
    )
    valid_counts.to_csv(args.output_dir / "property_valid_counts.csv", index=False, encoding="utf-8-sig")

    funnel_frames = []
    for scenario in SCENARIOS:
        brief = scenario_brief(scenario)
        funnel_frames.append(historical_funnel(source, scenario, brief))
    pd.concat(funnel_frames, ignore_index=True).to_csv(
        args.output_dir / "historical_funnel.csv", index=False, encoding="utf-8-sig"
    )

    print("[5/5] running nine inverse-design scenarios", flush=True)
    scenario_metrics: list[dict[str, object]] = []
    recommended_frames: list[pd.DataFrame] = []
    historical_frames: list[pd.DataFrame] = []
    for index, scenario in enumerate(SCENARIOS, start=1):
        print(f"  scenario {index}/{len(SCENARIOS)}: {scenario.key}", flush=True)
        metrics, recommended, historical = scenario_result(
            scenario,
            scenario_brief(scenario),
            full_models,
            p90_by_property,
            dataset_name=args.dataset,
            seed_count=args.seed_count,
            partners_per_seed=args.partners_per_seed,
            top_n=args.top_n,
            random_state=args.random_state,
        )
        scenario_metrics.append(metrics)
        recommended_frames.append(recommended)
        historical_frames.append(historical)

    scenario_metrics_frame = pd.DataFrame(scenario_metrics)
    candidates = pd.concat(recommended_frames, ignore_index=True)
    historical = pd.concat(historical_frames, ignore_index=True)
    scenario_metrics_frame.to_csv(
        args.output_dir / "scenario_metrics.csv", index=False, encoding="utf-8-sig"
    )
    candidates.to_csv(
        args.output_dir / "scenario_top_candidates.csv", index=False, encoding="utf-8-sig"
    )
    historical.to_csv(
        args.output_dir / "scenario_top_historical_seeds.csv", index=False, encoding="utf-8-sig"
    )
    candidate_overlap(candidates).to_csv(
        args.output_dir / "candidate_overlap.csv", index=False, encoding="utf-8-sig"
    )

    write_json(
        args.output_dir / "analysis_parameters.json",
        {
            "dataset": args.dataset,
            "model": args.model,
            "n_estimators": args.n_estimators,
            "evaluation_n_estimators": args.evaluation_n_estimators,
            "grouped_n_estimators": args.grouped_n_estimators,
            "seed_count": args.seed_count,
            "partners_per_seed": args.partners_per_seed,
            "top_n": args.top_n,
            "random_state": args.random_state,
        },
    )
    print(f"wrote analysis outputs to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()

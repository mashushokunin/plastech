from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd

from plastech.inverse_design import ForwardModelSet, TargetSpecification
from plastech.modeling import column_metadata


@dataclass(frozen=True)
class CandidateConstraints:
    """企業側の判断で差し替える、物性以外の暫定処方制約。"""

    max_material_count: int | None = None
    min_total_formulation_amount: float | None = None
    max_total_formulation_amount: float | None = None
    require_identified_primary_materials: bool = True
    forbidden_columns: tuple[str, ...] = ()
    forbidden_material_display_names: tuple[str, ...] = ()

    @classmethod
    def from_json(cls, path: str | Path) -> "CandidateConstraints":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            max_material_count=payload.get("max_material_count"),
            min_total_formulation_amount=payload.get("min_total_formulation_amount"),
            max_total_formulation_amount=payload.get("max_total_formulation_amount"),
            require_identified_primary_materials=payload.get(
                "require_identified_primary_materials", True
            ),
            forbidden_columns=tuple(payload.get("forbidden_columns", ())),
            forbidden_material_display_names=tuple(
                payload.get("forbidden_material_display_names", ())
            ),
        )

    def __post_init__(self) -> None:
        if self.max_material_count is not None and self.max_material_count <= 0:
            raise ValueError("max_material_count must be positive.")
        if (
            self.min_total_formulation_amount is not None
            and self.max_total_formulation_amount is not None
            and self.min_total_formulation_amount > self.max_total_formulation_amount
        ):
            raise ValueError("min_total_formulation_amount must not exceed the maximum.")


def _feature_columns(models: ForwardModelSet) -> list[str]:
    return sorted(
        {
            column
            for prepared in models.prepared.values()
            for column in prepared.feature_columns
        }
    )


def apply_candidate_constraints(
    candidates: pd.DataFrame,
    models: ForwardModelSet,
    constraints: CandidateConstraints,
    *,
    dataset_name: str = "processed_main",
) -> pd.DataFrame:
    """候補へ実務制約の判定列を追加する。候補を削除せず理由を残す。"""
    result = candidates.copy()
    features = _feature_columns(models)
    matrix = result.reindex(columns=features, fill_value=0.0)
    matrix = matrix.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    result["formulation_material_count"] = matrix.abs().gt(1e-12).sum(axis=1)
    result["formulation_total_amount"] = matrix.sum(axis=1)

    checks: list[tuple[str, pd.Series, str]] = []
    if constraints.max_material_count is not None:
        checks.append(
            (
                "material_count",
                result["formulation_material_count"].le(constraints.max_material_count),
                f"材料数>{constraints.max_material_count}",
            )
        )
    if constraints.min_total_formulation_amount is not None:
        checks.append(
            (
                "total_amount_min",
                result["formulation_total_amount"].ge(
                    constraints.min_total_formulation_amount
                ),
                f"総配合量<{constraints.min_total_formulation_amount:g}",
            )
        )
    if constraints.max_total_formulation_amount is not None:
        checks.append(
            (
                "total_amount_max",
                result["formulation_total_amount"].le(
                    constraints.max_total_formulation_amount
                ),
                f"総配合量>{constraints.max_total_formulation_amount:g}",
            )
        )
    if constraints.require_identified_primary_materials:
        identified = (
            ~result["primary_pvc"].astype(str).str.startswith("(no ")
            & ~result["primary_plasticizer"].astype(str).str.startswith("(no ")
        )
        checks.append(("identified_primary_materials", identified, "主材料を特定できない"))

    forbidden = set(constraints.forbidden_columns)
    if constraints.forbidden_material_display_names:
        metadata = column_metadata(dataset_name)
        forbidden.update(
            metadata.loc[
                metadata["display_name"].isin(constraints.forbidden_material_display_names),
                "column",
            ]
        )
    forbidden_present = pd.Series(False, index=result.index)
    for column in sorted(forbidden & set(matrix.columns)):
        forbidden_present |= matrix[column].abs().gt(1e-12)
    if forbidden:
        checks.append(("forbidden_materials", ~forbidden_present, "使用禁止原料を含む"))

    violations = pd.Series("", index=result.index, dtype=object)
    for key, passes, message in checks:
        passes = passes.fillna(False)
        result[f"practical_pass__{key}"] = passes
        violations = violations.mask(
            ~passes,
            violations.where(violations.eq(""), violations + ", ") + message,
        )
    result["practical_constraint_pass"] = True if not checks else pd.concat(
        [result[f"practical_pass__{key}"] for key, _, _ in checks], axis=1
    ).all(axis=1)
    result["practical_constraint_violations"] = violations
    return result


def add_prediction_intervals(
    candidates: pd.DataFrame,
    specifications: Sequence[TargetSpecification],
    p90_errors: dict[str, float],
    *,
    value_prefix: str = "predicted",
) -> pd.DataFrame:
    """OOF p90絶対誤差による参考区間と保守判定を追加する。"""
    result = candidates.copy()
    robust_columns: list[str] = []
    for spec in specifications:
        if spec.column not in p90_errors:
            raise KeyError(f"Missing p90 error for {spec.column}")
        values = pd.to_numeric(result[f"{value_prefix}__{spec.column}"], errors="coerce")
        error = float(p90_errors[spec.column])
        lower = values - error
        upper = values + error
        result[f"p90_lower__{spec.column}"] = lower
        result[f"p90_upper__{spec.column}"] = upper
        if spec.objective == "target":
            passes = lower.ge(spec.target - spec.tolerance) & upper.le(
                spec.target + spec.tolerance
            )
        elif spec.objective in {"lower_bound", "maximize"}:
            passes = lower.ge(spec.target)
        elif spec.objective in {"upper_bound", "minimize"}:
            passes = upper.le(spec.target)
        else:
            raise ValueError(f"Unknown objective: {spec.objective}")
        column = f"p90_robust_pass__{spec.column}"
        result[column] = passes.fillna(False)
        robust_columns.append(column)
    result["p90_robust_pass_all"] = result[robust_columns].all(axis=1)
    return result


def load_prediction_errors(path: str | Path) -> dict[str, float]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return {column: float(value) for column, value in payload["errors"].items()}

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd

from plastech.data_catalog import read_dataset
from plastech.modeling import (
    PreparedRegressionData,
    column_metadata,
    make_regression_model,
    prepare_regression_data,
)


Objective = Literal["target", "lower_bound", "upper_bound"]


@dataclass(frozen=True)
class TargetSpecification:
    """逆解析で使う1物性の目標値と判定方法。"""

    column: str
    name: str
    target: float
    tolerance: float
    objective: Objective = "target"
    weight: float = 1.0

    def description(self) -> str:
        if self.objective == "lower_bound":
            return f">= {self.target:g}"
        if self.objective == "upper_bound":
            return f"<= {self.target:g}"
        return f"{self.target:g} ± {self.tolerance:g}"


# 用途仕様が未確定のため、データ分布の中心付近に置いた最初の探索条件。
# 製品仕様が決まったらNotebook側でこの値を置き換える。
DEFAULT_INVERSE_TARGETS: tuple[TargetSpecification, ...] = (
    TargetSpecification("S03010_030010", "硬度 HDA (0s)", 78.0, 3.0),
    TargetSpecification("S06010_060030", "100%モジュラス (JIS2 200mm/min)", 7.5, 1.5),
    TargetSpecification(
        "S13010_130010",
        "脆化温度 (JIS K6723)",
        -33.0,
        3.0,
        objective="upper_bound",
    ),
    TargetSpecification("S01010_010010", "比重 (23℃)", 1.34, 0.05, weight=0.7),
)


@dataclass(frozen=True)
class ForwardModelSet:
    """各物性の順解析モデル。逆解析では候補の採点器として使う。"""

    specifications: tuple[TargetSpecification, ...]
    models: dict[str, Any]
    prepared: dict[str, PreparedRegressionData]


def specifications_frame(
    specifications: Sequence[TargetSpecification] = DEFAULT_INVERSE_TARGETS,
) -> pd.DataFrame:
    """Notebook表示用の目標物性表を返す。"""
    return pd.DataFrame(
        {
            "column": [spec.column for spec in specifications],
            "property": [spec.name for spec in specifications],
            "objective": [spec.objective for spec in specifications],
            "target": [spec.target for spec in specifications],
            "tolerance": [spec.tolerance for spec in specifications],
            "weight": [spec.weight for spec in specifications],
            "requirement": [spec.description() for spec in specifications],
        }
    )


def fit_forward_models(
    specifications: Sequence[TargetSpecification] = DEFAULT_INVERSE_TARGETS,
    *,
    dataset_name: str = "processed_main",
    model_name: str = "random_forest",
    n_estimators: int = 400,
    random_state: int = 42,
) -> ForwardModelSet:
    """物性ごとに「配合 -> 物性」のモデルを全データで学習する。

    ここで作るのは逆向きの回帰器ではない。同じ配合から複数物性を予測する
    順解析器であり、後段の候補探索で目標との差を採点するために使う。
    """
    specs = tuple(specifications)
    if not specs:
        raise ValueError("At least one target specification is required.")

    models: dict[str, Any] = {}
    prepared_by_target: dict[str, PreparedRegressionData] = {}
    for spec in specs:
        prepared = prepare_regression_data(dataset_name=dataset_name, target_col=spec.column)
        model = make_regression_model(
            model_name,
            random_state=random_state,
            n_estimators=n_estimators,
        )
        model.fit(prepared.X, prepared.y)
        models[spec.column] = model
        prepared_by_target[spec.column] = prepared

    return ForwardModelSet(
        specifications=specs,
        models=models,
        prepared=prepared_by_target,
    )


def _primary_material_label(
    X: pd.DataFrame,
    metadata: pd.DataFrame,
    base_group: str,
) -> pd.Series:
    meta = metadata.set_index("column")
    columns = [
        column
        for column in X.columns
        if column in meta.index and meta.at[column, "base_group"] == base_group
    ]
    if not columns:
        return pd.Series("(no columns)", index=X.index)

    labels = meta.loc[columns, "display_name"].to_dict()

    def choose(row: pd.Series) -> str:
        present = row[row > 0]
        if present.empty:
            return f"(no {base_group})"
        return labels[present.idxmax()]

    return X[columns].apply(choose, axis=1)


def build_candidate_library(
    forward_models: ForwardModelSet,
    *,
    dataset_name: str = "processed_main",
    require_all_observed: bool = True,
) -> pd.DataFrame:
    """探索の起点となる既存配合ライブラリを作る。

    ``require_all_observed=True`` の場合、全目標物性を実測済みの過去配合だけを
    残す。これにより最初の候補ランキングを実測値で検証できる。
    """
    source = read_dataset(dataset_name)
    id_column = "Unnamed: 0" if "Unnamed: 0" in source.columns else source.columns[0]

    # prepare_regression_data と同じ列集合を使い、各モデルに渡せる数値行列にする。
    feature_columns = sorted(
        {
            column
            for prepared in forward_models.prepared.values()
            for column in prepared.feature_columns
        }
    )
    source_features = source.reindex(columns=feature_columns).apply(pd.to_numeric, errors="coerce")
    source_features = source_features.fillna(0.0).astype(float)

    # 先頭2行は原料・物性の説明行であり、配合データではない。
    is_data_row = source[id_column].notna()
    valid_mask = is_data_row.copy()
    observed: dict[str, pd.Series] = {}
    for spec in forward_models.specifications:
        values = pd.to_numeric(source[spec.column], errors="coerce")
        observed[spec.column] = values
        if require_all_observed:
            valid_mask &= values.notna() & values.ne(0)

    metadata = column_metadata(dataset_name)
    library = source_features.loc[valid_mask].copy()
    library.insert(0, "row_id", source.loc[valid_mask, id_column].astype(str).values)
    library.insert(1, "candidate_type", "historical")
    library.insert(
        2,
        "primary_pvc",
        _primary_material_label(library[feature_columns], metadata, "PVC").values,
    )
    library.insert(
        3,
        "primary_plasticizer",
        _primary_material_label(library[feature_columns], metadata, "可塑剤オイル類").values,
    )
    for spec in forward_models.specifications:
        library[f"observed__{spec.column}"] = observed[spec.column].loc[valid_mask].values

    return library.reset_index(drop=True)


def _objective_error(values: pd.Series, spec: TargetSpecification) -> pd.Series:
    """許容幅を1とする無次元の目標誤差を計算する。小さいほど良い。"""
    if spec.tolerance <= 0:
        raise ValueError(f"tolerance must be positive: {spec.column}")
    if spec.objective == "target":
        return (values - spec.target).abs() / spec.tolerance
    if spec.objective == "lower_bound":
        return (spec.target - values).clip(lower=0) / spec.tolerance
    if spec.objective == "upper_bound":
        return (values - spec.target).clip(lower=0) / spec.tolerance
    raise ValueError(f"Unknown objective: {spec.objective}")


def _objective_pass(values: pd.Series, spec: TargetSpecification) -> pd.Series:
    """仕様を満たすかを判定する。境界条件は予測誤差幅で緩和しない。"""
    if spec.objective == "target":
        return (values - spec.target).abs().le(spec.tolerance)
    if spec.objective == "lower_bound":
        return values.ge(spec.target)
    if spec.objective == "upper_bound":
        return values.le(spec.target)
    raise ValueError(f"Unknown objective: {spec.objective}")


def add_predictions_and_scores(
    candidates: pd.DataFrame,
    forward_models: ForwardModelSet,
) -> pd.DataFrame:
    """候補に順解析予測と、実測・予測それぞれの目標スコアを追加する。"""
    result = candidates.copy()
    weighted_prediction_error = pd.Series(0.0, index=result.index)
    weighted_observed_error = pd.Series(0.0, index=result.index)
    total_weight = sum(spec.weight for spec in forward_models.specifications)

    for spec in forward_models.specifications:
        prepared = forward_models.prepared[spec.column]
        X = result.reindex(columns=prepared.feature_columns, fill_value=0.0)
        X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        prediction = pd.Series(forward_models.models[spec.column].predict(X), index=result.index)
        predicted_col = f"predicted__{spec.column}"
        result[predicted_col] = prediction
        prediction_error = _objective_error(prediction, spec)
        result[f"prediction_error__{spec.column}"] = prediction_error
        weighted_prediction_error += spec.weight * prediction_error

        observed_col = f"observed__{spec.column}"
        if observed_col in result:
            observed = pd.to_numeric(result[observed_col], errors="coerce")
            observed_error = _objective_error(observed, spec)
            result[f"observed_error__{spec.column}"] = observed_error
            weighted_observed_error += spec.weight * observed_error.fillna(0.0)

    result["predicted_score"] = weighted_prediction_error / total_weight
    predicted_passes = pd.concat(
        [
            _objective_pass(result[f"predicted__{spec.column}"], spec)
            for spec in forward_models.specifications
        ],
        axis=1,
    )
    result["predicted_meets_all"] = predicted_passes.all(axis=1)
    if all(f"observed__{spec.column}" in result for spec in forward_models.specifications):
        result["observed_score"] = weighted_observed_error / total_weight
        observed_passes = pd.concat(
            [
                _objective_pass(result[f"observed__{spec.column}"], spec)
                for spec in forward_models.specifications
            ],
            axis=1,
        )
        result["observed_meets_all"] = observed_passes.all(axis=1)
    return result


def rank_historical_candidates(
    library: pd.DataFrame,
    forward_models: ForwardModelSet,
) -> pd.DataFrame:
    """実測物性で過去配合を順位付けし、順解析値も並べて確認可能にする。"""
    scored = add_predictions_and_scores(library, forward_models)
    return scored.sort_values(["observed_score", "predicted_score"], kind="stable").reset_index(drop=True)


def generate_interpolated_candidates(
    library: pd.DataFrame,
    historical_ranking: pd.DataFrame,
    forward_models: ForwardModelSet,
    *,
    seed_count: int = 20,
    partners_per_seed: int = 3,
    fractions: tuple[float, ...] = (0.25, 0.5, 0.75),
    random_state: int = 42,
) -> pd.DataFrame:
    """同一の主PVC・主可塑剤系列で、既存配合を凸結合した仮想候補を生成する。

    これは原料可否、総配合量、加工性、コスト、法規を保証しない探索候補である。
    実験へ進める前に必ず配合担当者が処方を確認すること。
    """
    if not fractions or any(not 0.0 < fraction < 1.0 for fraction in fractions):
        raise ValueError("fractions must be between 0 and 1.")

    feature_columns = sorted(
        {
            column
            for prepared in forward_models.prepared.values()
            for column in prepared.feature_columns
        }
    )
    rng = np.random.default_rng(random_state)
    records: list[dict[str, Any]] = []
    seeds = historical_ranking.head(seed_count)

    for _, seed in seeds.iterrows():
        peers = library[
            (library["primary_pvc"] == seed["primary_pvc"])
            & (library["primary_plasticizer"] == seed["primary_plasticizer"])
            & (library["row_id"] != seed["row_id"])
        ]
        if peers.empty:
            continue
        partner_count = min(partners_per_seed, len(peers))
        partner_positions = rng.choice(len(peers), size=partner_count, replace=False)
        for partner_position in partner_positions:
            partner = peers.iloc[int(partner_position)]
            for fraction in fractions:
                values = (
                    fraction * seed[feature_columns].astype(float)
                    + (1.0 - fraction) * partner[feature_columns].astype(float)
                )
                record: dict[str, Any] = values.to_dict()
                record.update(
                    {
                        "row_id": f"virtual::{seed['row_id']}::{partner['row_id']}::{fraction:.2f}",
                        "candidate_type": "interpolated_virtual",
                        "parent_a": seed["row_id"],
                        "parent_b": partner["row_id"],
                        "parent_a_fraction": fraction,
                        "primary_pvc": seed["primary_pvc"],
                        "primary_plasticizer": seed["primary_plasticizer"],
                    }
                )
                records.append(record)

    if not records:
        raise ValueError("No compatible parent pairs were found for interpolation.")
    return add_predictions_and_scores(pd.DataFrame(records), forward_models).sort_values(
        "predicted_score", kind="stable"
    ).reset_index(drop=True)


def candidate_summary(
    candidates: pd.DataFrame,
    forward_models: ForwardModelSet,
    *,
    top_n: int = 20,
) -> pd.DataFrame:
    """配合の全列を隠し、候補確認に必要な列だけを返す。"""
    columns = [
        column
        for column in (
            "candidate_type",
            "row_id",
            "parent_a",
            "parent_b",
            "parent_a_fraction",
            "primary_pvc",
            "primary_plasticizer",
            "observed_score",
            "observed_meets_all",
            "predicted_score",
            "predicted_meets_all",
        )
        if column in candidates
    ]
    for spec in forward_models.specifications:
        for prefix in ("observed", "predicted"):
            column = f"{prefix}__{spec.column}"
            if column in candidates:
                columns.append(column)
    return candidates.loc[:, columns].head(top_n).copy()

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


Objective = Literal["target", "lower_bound", "upper_bound", "minimize", "maximize"]


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
        if self.objective == "minimize":
            return f"minimize (reference <= {self.target:g})"
        if self.objective == "maximize":
            return f"maximize (reference >= {self.target:g})"
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


@dataclass(frozen=True)
class DesignBrief:
    """用途イメージと、用途内で役割分けした物性目標の組み合わせ。

    ``preferences`` は既存Notebookとの互換性のため名称を維持しているが、
    新しい設計では候補順位とPareto判定に使う「最適化物性」を表す。
    管理目標と制約は候補選定時の優先条件、監視物性は予測・表示だけに使う。
    """

    name: str
    description: str
    preferences: tuple[TargetSpecification, ...]
    management_targets: tuple[TargetSpecification, ...] = ()
    constraints: tuple[TargetSpecification, ...] = ()
    monitors: tuple[TargetSpecification, ...] = ()

    def __post_init__(self) -> None:
        if not self.preferences:
            raise ValueError("DesignBrief.preferences must not be empty.")
        columns = [spec.column for spec in self.all_specifications]
        duplicates = sorted({column for column in columns if columns.count(column) > 1})
        if duplicates:
            raise ValueError(
                "A property must have only one role in a DesignBrief: "
                f"{duplicates}"
            )

    @property
    def objectives(self) -> tuple[TargetSpecification, ...]:
        """候補順位とPareto判定に使う最適化物性。"""
        return self.preferences

    @property
    def required_specifications(self) -> tuple[TargetSpecification, ...]:
        """探索ライブラリと候補優先順位に使う必須測定物性。"""
        return self.objectives + self.management_targets + self.constraints

    @property
    def all_specifications(self) -> tuple[TargetSpecification, ...]:
        """予測する全物性。監視物性も含む。"""
        return self.required_specifications + self.monitors

    @property
    def main_goal(self) -> TargetSpecification:
        return self.objectives[0]


@dataclass(frozen=True)
class InverseDesignResult:
    """1つの材料像に対する逆解析結果一式。"""

    brief: DesignBrief
    forward_models: ForwardModelSet
    library: pd.DataFrame
    seed_ranking: pd.DataFrame
    virtual_candidates: pd.DataFrame
    scored_candidates: pd.DataFrame
    pareto_candidates: pd.DataFrame
    recommended_candidates: pd.DataFrame


DEFAULT_DESIGN_BRIEFS: dict[str, DesignBrief] = {
    "柔軟・低温用途": DesignBrief(
        name="柔軟・低温用途",
        description="柔らかく、低温で脆くなりにくい材料を狙う。",
        preferences=(
            TargetSpecification("S03010_030010", "硬度 HDA (0s)", 70.0, 4.0, weight=1.4),
            TargetSpecification(
                "S06060_060030",
                "引張伸び (JIS2 200mm/min)",
                370.0,
                40.0,
                objective="lower_bound",
            ),
            TargetSpecification(
                "S13010_130010",
                "脆化温度 (JIS K6723)",
                -41.0,
                4.0,
                objective="upper_bound",
            ),
            TargetSpecification("S01010_010010", "比重 (23℃)", 1.34, 0.06, weight=0.5),
        ),
    ),
    "硬さ・剛性用途": DesignBrief(
        name="硬さ・剛性用途",
        description="硬度と剛性を優先する材料を狙う。",
        preferences=(
            TargetSpecification("S03010_030010", "硬度 HDA (0s)", 89.0, 3.0, weight=1.4),
            TargetSpecification(
                "S06010_060030",
                "100%モジュラス (JIS2 200mm/min)",
                11.0,
                2.0,
                objective="lower_bound",
            ),
            TargetSpecification(
                "S06020_060030",
                "引張強さ (JIS2 200mm/min)",
                20.0,
                3.0,
                objective="lower_bound",
                weight=0.8,
            ),
            TargetSpecification("S01010_010010", "比重 (23℃)", 1.38, 0.06, weight=0.4),
        ),
    ),
    "バランス型軟質PVC": DesignBrief(
        name="バランス型軟質PVC",
        description="硬度、剛性、低温特性、比重のバランスを狙う。",
        preferences=(
            TargetSpecification("S03010_030010", "硬度 HDA (0s)", 78.0, 3.0, weight=1.3),
            TargetSpecification("S06010_060030", "100%モジュラス (JIS2 200mm/min)", 7.5, 1.5),
            TargetSpecification(
                "S13010_130010",
                "脆化温度 (JIS K6723)",
                -33.0,
                3.0,
                objective="upper_bound",
            ),
            TargetSpecification("S01010_010010", "比重 (23℃)", 1.34, 0.05, weight=0.5),
        ),
    ),
}


# 2026-08の物性数比較を受けて導入した役割分離版。
# 既存の4物性方式は DEFAULT_DESIGN_BRIEFS に残し、新旧比較を可能にする。
ROLE_BASED_DESIGN_BRIEFS: dict[str, DesignBrief] = {
    "柔軟・低温用途": DesignBrief(
        name="柔軟・低温用途（役割分離）",
        description=(
            "脆化温度を最適化し、硬度と引張伸びを必要条件として確認する。"
            "比重、100%モジュラス、引張強さは初期順位を変えずに監視する。"
        ),
        preferences=(
            TargetSpecification(
                "S13010_130010",
                "脆化温度 (JIS K6723)",
                -41.0,
                4.0,
                objective="minimize",
            ),
        ),
        management_targets=(
            TargetSpecification("S03010_030010", "硬度 HDA (0s)", 70.0, 4.0),
        ),
        constraints=(
            TargetSpecification(
                "S06060_060030",
                "引張伸び (JIS2 200mm/min)",
                370.0,
                40.0,
                objective="lower_bound",
            ),
        ),
        monitors=(
            # 監視物性の目標値は前回の探索条件を参考値として保持するが、
            # 候補順位・Pareto判定・必要条件の合否には使わない。
            TargetSpecification("S01010_010010", "比重 (23℃)", 1.34, 0.06),
            TargetSpecification(
                "S06010_060030",
                "100%モジュラス (JIS2 200mm/min)",
                7.5,
                1.5,
            ),
            TargetSpecification(
                "S06020_060030",
                "引張強さ (JIS2 200mm/min)",
                20.0,
                3.0,
                objective="lower_bound",
            ),
        ),
    ),
}


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


def brief_specifications_frame(brief: DesignBrief) -> pd.DataFrame:
    """用途内の物性役割と探索条件を表示用の表にする。"""
    frames: list[pd.DataFrame] = []
    for role, specifications in (
        ("objective", brief.objectives),
        ("management_target", brief.management_targets),
        ("constraint", brief.constraints),
        ("monitor", brief.monitors),
    ):
        if not specifications:
            continue
        frame = specifications_frame(specifications)
        frame.insert(0, "role", role)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


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
    required_observed: Sequence[TargetSpecification] | None = None,
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
    required_columns = {
        spec.column
        for spec in (
            tuple(required_observed)
            if required_observed is not None
            else forward_models.specifications
        )
    }
    available_columns = {spec.column for spec in forward_models.specifications}
    missing_required = required_columns - available_columns
    if missing_required:
        raise ValueError(
            "required_observed contains properties without a fitted model: "
            f"{sorted(missing_required)}"
        )

    observed: dict[str, pd.Series] = {}
    for spec in forward_models.specifications:
        values = pd.to_numeric(source[spec.column], errors="coerce")
        observed[spec.column] = values
        if require_all_observed and spec.column in required_columns:
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
    if spec.objective == "minimize":
        return (values - spec.target) / spec.tolerance
    if spec.objective == "maximize":
        return (spec.target - values) / spec.tolerance
    raise ValueError(f"Unknown objective: {spec.objective}")


def _objective_pass(values: pd.Series, spec: TargetSpecification) -> pd.Series:
    """仕様を満たすかを判定する。境界条件は予測誤差幅で緩和しない。"""
    if spec.objective == "target":
        # 1.34 - 0.05 のような十進小数の境界が、二進浮動小数点の丸めで
        # 0.050000000000000044 となり不合格になることだけを防ぐ。
        return (values - spec.target).abs().le(spec.tolerance + 1e-12)
    if spec.objective == "lower_bound":
        return values.ge(spec.target)
    if spec.objective == "upper_bound":
        return values.le(spec.target)
    if spec.objective == "minimize":
        return values.le(spec.target)
    if spec.objective == "maximize":
        return values.ge(spec.target)
    raise ValueError(f"Unknown objective: {spec.objective}")


def _objective_margin(values: pd.Series, spec: TargetSpecification) -> pd.Series:
    """仕様境界までの余裕を物性単位で返す。正なら合格、負なら未達。"""
    if spec.objective == "target":
        return spec.tolerance - (values - spec.target).abs()
    if spec.objective == "lower_bound":
        return values - spec.target
    if spec.objective == "upper_bound":
        return spec.target - values
    if spec.objective == "minimize":
        return spec.target - values
    if spec.objective == "maximize":
        return values - spec.target
    raise ValueError(f"Unknown objective: {spec.objective}")


def add_predictions_and_scores(
    candidates: pd.DataFrame,
    forward_models: ForwardModelSet,
    *,
    scoring_specifications: Sequence[TargetSpecification] | None = None,
) -> pd.DataFrame:
    """候補に順解析予測と、実測・予測それぞれの目標スコアを追加する。"""
    result = candidates.copy()
    weighted_prediction_error = pd.Series(0.0, index=result.index)
    weighted_observed_error = pd.Series(0.0, index=result.index)
    scoring_specs = tuple(scoring_specifications or forward_models.specifications)
    scoring_by_column = {spec.column: spec for spec in scoring_specs}
    total_weight = sum(spec.weight for spec in scoring_specs)
    if total_weight <= 0:
        raise ValueError("The total scoring weight must be positive.")

    for spec in forward_models.specifications:
        prepared = forward_models.prepared[spec.column]
        X = result.reindex(columns=prepared.feature_columns, fill_value=0.0)
        X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        prediction = pd.Series(forward_models.models[spec.column].predict(X), index=result.index)
        predicted_col = f"predicted__{spec.column}"
        result[predicted_col] = prediction
        score_spec = scoring_by_column.get(spec.column)
        prediction_error = _objective_error(prediction, score_spec or spec)
        result[f"prediction_error__{spec.column}"] = prediction_error
        if score_spec is not None:
            weighted_prediction_error += score_spec.weight * prediction_error

        observed_col = f"observed__{spec.column}"
        if observed_col in result:
            observed = pd.to_numeric(result[observed_col], errors="coerce")
            observed_error = _objective_error(observed, score_spec or spec)
            result[f"observed_error__{spec.column}"] = observed_error
            if score_spec is not None:
                weighted_observed_error += score_spec.weight * observed_error.fillna(0.0)

    result["predicted_score"] = weighted_prediction_error / total_weight
    predicted_passes = pd.concat(
        [
            _objective_pass(result[f"predicted__{spec.column}"], spec)
            for spec in scoring_specs
        ],
        axis=1,
    )
    result["predicted_meets_all"] = predicted_passes.all(axis=1)
    if all(f"observed__{spec.column}" in result for spec in scoring_specs):
        result["observed_score"] = weighted_observed_error / total_weight
        observed_passes = pd.concat(
            [
                _objective_pass(result[f"observed__{spec.column}"], spec)
                for spec in scoring_specs
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


def rank_seed_candidates(
    library: pd.DataFrame,
    main_goal: TargetSpecification,
) -> pd.DataFrame:
    """主目標の実測値が近い過去配合を、仮想候補生成の起点として並べる。"""
    observed_col = f"observed__{main_goal.column}"
    if observed_col not in library:
        raise KeyError(f"{observed_col} is required in library.")

    result = library.copy()
    observed = pd.to_numeric(result[observed_col], errors="coerce")
    result["main_goal_error"] = objective_error(observed, main_goal)
    return result.sort_values("main_goal_error", kind="stable").reset_index(drop=True)


def generate_interpolated_candidates(
    library: pd.DataFrame,
    historical_ranking: pd.DataFrame,
    forward_models: ForwardModelSet,
    *,
    seed_count: int = 20,
    partners_per_seed: int = 3,
    fractions: tuple[float, ...] = (0.25, 0.5, 0.75),
    random_state: int = 42,
    scoring_specifications: Sequence[TargetSpecification] | None = None,
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
    return add_predictions_and_scores(
        pd.DataFrame(records),
        forward_models,
        scoring_specifications=scoring_specifications,
    ).sort_values(
        "predicted_score", kind="stable"
    ).reset_index(drop=True)


def objective_error(values: pd.Series, specification: TargetSpecification) -> pd.Series:
    """目標からの無次元誤差を返す。小さいほど好ましい。"""
    return _objective_error(values, specification)


def pareto_optimal_mask(error_matrix: np.ndarray) -> np.ndarray:
    """小さいほど良い誤差行列に対し、他候補に支配されない行だけを残す。"""
    errors = np.asarray(error_matrix, dtype=float)
    if errors.ndim != 2:
        raise ValueError("error_matrix must be a 2D array.")

    mask = np.ones(len(errors), dtype=bool)
    for index, error in enumerate(errors):
        dominates_index = np.all(errors <= error, axis=1) & np.any(errors < error, axis=1)
        if dominates_index.any():
            mask[index] = False
    return mask


def score_design_preferences(
    candidates: pd.DataFrame,
    preferences: Sequence[TargetSpecification],
    *,
    value_prefix: str = "predicted",
) -> pd.DataFrame:
    """用途別の主目標・副目標で候補を採点し、Pareto判定も追加する。"""
    specs = tuple(preferences)
    if not specs:
        raise ValueError("preferences must not be empty.")

    result = candidates.copy()
    error_columns: list[str] = []
    for specification in specs:
        value_col = f"{value_prefix}__{specification.column}"
        if value_col not in result:
            raise KeyError(f"{value_col} is required in candidates.")
        error_col = f"preference_error__{specification.column}"
        result[error_col] = objective_error(
            pd.to_numeric(result[value_col], errors="coerce"),
            specification,
        )
        error_columns.append(error_col)

    total_weight = sum(spec.weight for spec in specs)
    weighted_error = sum(spec.weight * result[f"preference_error__{spec.column}"] for spec in specs)
    result["design_score"] = weighted_error / total_weight
    result["pareto_optimal"] = pareto_optimal_mask(result[error_columns].to_numpy())
    return result.sort_values("design_score", kind="stable").reset_index(drop=True)


def add_preference_status(
    candidates: pd.DataFrame,
    preferences: Sequence[TargetSpecification],
    *,
    value_prefix: str = "predicted",
) -> pd.DataFrame:
    """候補ごとに、用途仕様の合否・未達物性・境界余裕を追加する。"""
    specs = tuple(preferences)
    if not specs:
        raise ValueError("preferences must not be empty.")

    result = candidates.copy()
    pass_columns: list[str] = []
    for spec in specs:
        value_col = f"{value_prefix}__{spec.column}"
        if value_col not in result:
            raise KeyError(f"{value_col} is required in candidates.")
        values = pd.to_numeric(result[value_col], errors="coerce")
        pass_col = f"{value_prefix}_meets__{spec.column}"
        margin_col = f"{value_prefix}_margin__{spec.column}"
        result[pass_col] = _objective_pass(values, spec).fillna(False)
        result[margin_col] = _objective_margin(values, spec)
        pass_columns.append(pass_col)

    result[f"{value_prefix}_meets_preferences"] = result[pass_columns].all(axis=1)

    def unmet(row: pd.Series) -> str:
        names = [spec.name for spec, column in zip(specs, pass_columns) if not bool(row[column])]
        return ", ".join(names)

    result[f"{value_prefix}_unmet_preferences"] = result.apply(unmet, axis=1)
    return result


def _add_role_status(
    candidates: pd.DataFrame,
    specifications: Sequence[TargetSpecification],
    *,
    role: str,
    value_prefix: str = "predicted",
) -> pd.DataFrame:
    """指定した役割の合否、未達物性、未達件数を候補表へ追加する。"""
    specs = tuple(specifications)
    result = candidates.copy()
    if not specs:
        result[f"{value_prefix}_meets_{role}"] = True
        result[f"{value_prefix}_unmet_{role}"] = ""
        result[f"{value_prefix}_{role}_violation_count"] = 0
        return result

    pass_columns: list[str] = []
    for spec in specs:
        value_col = f"{value_prefix}__{spec.column}"
        if value_col not in result:
            raise KeyError(f"{value_col} is required in candidates.")
        values = pd.to_numeric(result[value_col], errors="coerce")
        pass_col = f"{value_prefix}_{role}_meets__{spec.column}"
        margin_col = f"{value_prefix}_{role}_margin__{spec.column}"
        result[pass_col] = _objective_pass(values, spec).fillna(False)
        result[margin_col] = _objective_margin(values, spec)
        pass_columns.append(pass_col)

    result[f"{value_prefix}_meets_{role}"] = result[pass_columns].all(axis=1)
    result[f"{value_prefix}_{role}_violation_count"] = (~result[pass_columns]).sum(axis=1)

    def unmet(row: pd.Series) -> str:
        names = [spec.name for spec, column in zip(specs, pass_columns) if not bool(row[column])]
        return ", ".join(names)

    result[f"{value_prefix}_unmet_{role}"] = result.apply(unmet, axis=1)
    return result


def add_design_brief_status(
    candidates: pd.DataFrame,
    brief: DesignBrief,
    *,
    value_prefix: str = "predicted",
) -> pd.DataFrame:
    """役割別の判定を追加する。監視物性は参考判定であり選定には使わない。"""
    result = candidates.copy()
    for role, specifications in (
        ("objectives", brief.objectives),
        ("management_targets", brief.management_targets),
        ("constraints", brief.constraints),
        ("monitors", brief.monitors),
    ):
        result = _add_role_status(
            result,
            specifications,
            role=role,
            value_prefix=value_prefix,
        )

    result[f"{value_prefix}_meets_requirements"] = (
        result[f"{value_prefix}_meets_management_targets"]
        & result[f"{value_prefix}_meets_constraints"]
    )
    result[f"{value_prefix}_requirement_violation_count"] = (
        result[f"{value_prefix}_management_targets_violation_count"]
        + result[f"{value_prefix}_constraints_violation_count"]
    )
    result[f"{value_prefix}_meets_objectives_and_requirements"] = (
        result[f"{value_prefix}_meets_objectives"]
        & result[f"{value_prefix}_meets_requirements"]
    )
    return result


def add_formulation_material_count(
    candidates: pd.DataFrame,
    forward_models: ForwardModelSet,
) -> pd.DataFrame:
    """各候補で配合量が0より大きい説明変数の数を暫定材料数として追加する。"""
    feature_columns = sorted(
        {
            column
            for prepared in forward_models.prepared.values()
            for column in prepared.feature_columns
        }
    )
    result = candidates.copy()
    matrix = result.reindex(columns=feature_columns, fill_value=0.0)
    matrix = matrix.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    result["formulation_material_count"] = matrix.abs().gt(1e-12).sum(axis=1)
    return result


def select_diverse_candidates(
    candidates: pd.DataFrame,
    *,
    top_n: int = 5,
    prefer_pareto: bool = True,
) -> pd.DataFrame:
    """材料系列が重複しすぎないように、上位候補から実験候補を選ぶ。"""
    if top_n <= 0:
        raise ValueError("top_n must be positive.")
    required = {"primary_pvc", "primary_plasticizer", "design_score"}
    missing = required - set(candidates.columns)
    if missing:
        raise KeyError(f"candidates is missing required columns: {sorted(missing)}")

    sort_columns: list[str] = []
    ascending: list[bool] = []
    if "predicted_meets_requirements" in candidates:
        sort_columns.append("predicted_meets_requirements")
        ascending.append(False)
    if "predicted_requirement_violation_count" in candidates:
        sort_columns.append("predicted_requirement_violation_count")
        ascending.append(True)
    sort_columns.append("design_score")
    ascending.append(True)

    ranked_all = candidates.sort_values(sort_columns, ascending=ascending, kind="stable").copy()
    ranked_all["material_series"] = (
        ranked_all["primary_pvc"] + " | " + ranked_all["primary_plasticizer"]
    )

    source = ranked_all
    if prefer_pareto and "pareto_optimal" in candidates:
        pareto = ranked_all[ranked_all["pareto_optimal"]].copy()
        if not pareto.empty:
            feasible_pareto = pareto
            if "predicted_meets_requirements" in pareto:
                feasible_pareto = pareto[pareto["predicted_meets_requirements"]]
                feasible_all = ranked_all[ranked_all["predicted_meets_requirements"]]
                if not feasible_pareto.empty:
                    source = feasible_pareto
                elif not feasible_all.empty:
                    source = feasible_all
                else:
                    source = pareto
            else:
                source = pareto

    result = source.copy()
    diverse = result.drop_duplicates("material_series")
    if len(diverse) < top_n:
        fallback = result
        diverse = pd.concat([diverse, fallback], ignore_index=True).drop_duplicates("row_id")

    # Pareto候補がtop_n未満の場合は、全候補から未選択の材料系列を優先して補う。
    # 以前はPareto集合内だけで補充していたため、2物性などPareto候補が少ない
    # 条件では要求件数に達しないことがあった。
    if len(diverse) < top_n and len(source) < len(ranked_all):
        selected_series = set(diverse["material_series"])
        new_series = ranked_all[~ranked_all["material_series"].isin(selected_series)]
        new_series = new_series.drop_duplicates("material_series")
        diverse = pd.concat([diverse, new_series], ignore_index=True).drop_duplicates("row_id")
    if len(diverse) < top_n:
        diverse = pd.concat([diverse, ranked_all], ignore_index=True).drop_duplicates("row_id")
    return diverse.head(top_n).reset_index(drop=True)


def _candidate_metadata_columns() -> set[str]:
    return {
        "row_id",
        "candidate_type",
        "parent_a",
        "parent_b",
        "parent_a_fraction",
        "primary_pvc",
        "primary_plasticizer",
        "main_goal_error",
        "observed_score",
        "observed_meets_all",
        "predicted_score",
        "predicted_meets_all",
        "design_score",
        "pareto_optimal",
        "material_series",
        "predicted_meets_preferences",
        "predicted_unmet_preferences",
        "observed_meets_preferences",
        "observed_unmet_preferences",
    }


def _infer_formulation_columns(
    candidates: pd.DataFrame,
    library: pd.DataFrame,
) -> list[str]:
    metadata_columns = _candidate_metadata_columns()
    columns: list[str] = []
    for column in candidates.columns:
        if column not in library.columns:
            continue
        if column in metadata_columns or any(
            column.startswith(prefix)
            for prefix in (
                "observed__",
                "predicted__",
                "prediction_error__",
                "observed_error__",
                "preference_error__",
                "predicted_meets__",
                "predicted_margin__",
                "observed_meets__",
                "observed_margin__",
            )
        ):
            continue
        if pd.api.types.is_numeric_dtype(pd.to_numeric(library[column], errors="coerce")):
            columns.append(column)
    return columns


def parent_formulation_deltas(
    candidates: pd.DataFrame,
    library: pd.DataFrame,
    *,
    feature_columns: Sequence[str] | None = None,
    top_n_per_candidate: int = 30,
    min_abs_delta: float = 1e-9,
    dataset_name: str = "processed_main",
) -> pd.DataFrame:
    """推薦候補と親配合A/Bの配合差分を、原料別の縦持ち表で返す。"""
    if top_n_per_candidate <= 0:
        raise ValueError("top_n_per_candidate must be positive.")
    required = {"row_id", "parent_a", "parent_b"}
    missing = required - set(candidates.columns)
    if missing:
        raise KeyError(f"candidates is missing required columns: {sorted(missing)}")
    if "row_id" not in library:
        raise KeyError("library is missing required column: row_id")

    columns = list(feature_columns) if feature_columns is not None else _infer_formulation_columns(candidates, library)
    metadata = column_metadata(dataset_name).set_index("column")
    library_by_id = library.set_index("row_id", drop=False)
    rows: list[dict[str, Any]] = []

    for rank, (_, candidate) in enumerate(candidates.reset_index(drop=True).iterrows(), start=1):
        parent_a_id = candidate["parent_a"]
        parent_b_id = candidate["parent_b"]
        if parent_a_id not in library_by_id.index or parent_b_id not in library_by_id.index:
            continue

        parent_a = library_by_id.loc[parent_a_id]
        parent_b = library_by_id.loc[parent_b_id]
        candidate_values = pd.to_numeric(candidate.reindex(columns), errors="coerce").fillna(0.0)
        parent_a_values = pd.to_numeric(parent_a.reindex(columns), errors="coerce").fillna(0.0)
        parent_b_values = pd.to_numeric(parent_b.reindex(columns), errors="coerce").fillna(0.0)

        deltas = pd.DataFrame(
            {
                "column": columns,
                "candidate_amount": candidate_values.to_numpy(dtype=float),
                "parent_a_amount": parent_a_values.to_numpy(dtype=float),
                "parent_b_amount": parent_b_values.to_numpy(dtype=float),
            }
        )
        deltas["delta_from_parent_a"] = deltas["candidate_amount"] - deltas["parent_a_amount"]
        deltas["delta_from_parent_b"] = deltas["candidate_amount"] - deltas["parent_b_amount"]
        deltas["max_abs_delta"] = deltas[["delta_from_parent_a", "delta_from_parent_b"]].abs().max(axis=1)
        deltas = deltas[deltas["max_abs_delta"] > min_abs_delta]
        deltas = deltas.sort_values("max_abs_delta", ascending=False).head(top_n_per_candidate)

        for _, delta in deltas.iterrows():
            column = delta["column"]
            meta = metadata.loc[column] if column in metadata.index else {}
            rows.append(
                {
                    "candidate_rank": rank,
                    "candidate_row_id": candidate["row_id"],
                    "parent_a": parent_a_id,
                    "parent_b": parent_b_id,
                    "parent_a_fraction": candidate.get("parent_a_fraction"),
                    "column": column,
                    "base_group": meta.get("base_group") if hasattr(meta, "get") else None,
                    "category": meta.get("category") if hasattr(meta, "get") else None,
                    "label": meta.get("label") if hasattr(meta, "get") else None,
                    "display_name": meta.get("display_name") if hasattr(meta, "get") else column,
                    "candidate_amount": delta["candidate_amount"],
                    "parent_a_amount": delta["parent_a_amount"],
                    "parent_b_amount": delta["parent_b_amount"],
                    "delta_from_parent_a": delta["delta_from_parent_a"],
                    "delta_from_parent_b": delta["delta_from_parent_b"],
                }
            )

    return pd.DataFrame(rows)


def run_inverse_design_for_brief(
    brief: DesignBrief,
    *,
    dataset_name: str = "processed_main",
    model_name: str = "lightgbm",
    n_estimators: int = 800,
    seed_count: int = 30,
    partners_per_seed: int = 5,
    fractions: tuple[float, ...] = (0.25, 0.5, 0.75),
    top_n: int = 5,
    random_state: int = 42,
    max_material_count: int | None = None,
    require_identified_primary_materials: bool = False,
) -> InverseDesignResult:
    """材料像から順解析モデル学習、候補生成、Pareto候補選定まで一括実行する。"""
    forward_models = fit_forward_models(
        brief.all_specifications,
        dataset_name=dataset_name,
        model_name=model_name,
        n_estimators=n_estimators,
        random_state=random_state,
    )
    library = build_candidate_library(
        forward_models,
        dataset_name=dataset_name,
        required_observed=brief.required_specifications,
    )
    seed_ranking = rank_seed_candidates(library, brief.main_goal)
    virtual_candidates = generate_interpolated_candidates(
        library,
        seed_ranking,
        forward_models,
        seed_count=seed_count,
        partners_per_seed=partners_per_seed,
        fractions=fractions,
        random_state=random_state,
        scoring_specifications=brief.objectives,
    )
    scored_candidates = score_design_preferences(virtual_candidates, brief.objectives)
    scored_candidates = add_design_brief_status(scored_candidates, brief)
    scored_candidates = add_formulation_material_count(scored_candidates, forward_models)
    pareto_candidates = scored_candidates[scored_candidates["pareto_optimal"]].copy()
    eligible_candidates = scored_candidates
    if max_material_count is not None:
        if max_material_count <= 0:
            raise ValueError("max_material_count must be positive.")
        eligible_candidates = eligible_candidates[
            eligible_candidates["formulation_material_count"].le(max_material_count)
        ]
    if require_identified_primary_materials:
        eligible_candidates = eligible_candidates[
            ~eligible_candidates["primary_pvc"].str.startswith("(no ")
            & ~eligible_candidates["primary_plasticizer"].str.startswith("(no ")
        ]
    if eligible_candidates.empty:
        raise ValueError("No candidates remain after applying practical filters.")
    recommended_candidates = select_diverse_candidates(
        eligible_candidates,
        top_n=top_n,
        prefer_pareto=True,
    )
    return InverseDesignResult(
        brief=brief,
        forward_models=forward_models,
        library=library,
        seed_ranking=seed_ranking,
        virtual_candidates=virtual_candidates,
        scored_candidates=scored_candidates,
        pareto_candidates=pareto_candidates.reset_index(drop=True),
        recommended_candidates=recommended_candidates,
    )


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
            "design_score",
            "pareto_optimal",
            "predicted_meets_preferences",
            "predicted_unmet_preferences",
            "predicted_meets_objectives",
            "predicted_unmet_objectives",
            "predicted_meets_management_targets",
            "predicted_unmet_management_targets",
            "predicted_meets_constraints",
            "predicted_unmet_constraints",
            "predicted_meets_requirements",
            "predicted_requirement_violation_count",
            "predicted_meets_objectives_and_requirements",
            "predicted_meets_monitors",
            "predicted_unmet_monitors",
            "formulation_material_count",
        )
        if column in candidates
    ]
    for spec in forward_models.specifications:
        for prefix in ("observed", "predicted"):
            column = f"{prefix}__{spec.column}"
            if column in candidates:
                columns.append(column)
    return candidates.loc[:, columns].head(top_n).copy()

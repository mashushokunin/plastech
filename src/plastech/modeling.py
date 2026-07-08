from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Any

import pandas as pd

from plastech.data_catalog import read_dataset


# このファイルは、Notebookから共通利用する機械学習処理をまとめたもの。
# Notebook側は「結果を見る・考える」場所にして、前処理・評価・モデル比較の
# 実装はここに集約する。

# 最初に詳しく見る物性。S06020_060030 は「引張強さ JIS2 200mm/min」。
DEFAULT_TARGET = "S06020_060030"

# 現時点の仮の合格ライン。MAEや最大外れ誤差は、実務許容値が決まってから
# 別途ルール化する。
EVALUATION_RULES: dict[str, float] = {
    "random_split_r2_min": 0.75,
    "exact_composition_r2_min": 0.70,
    "primary_pvc_plus_primary_plasticizer_r2_min": 0.50,
}

# 最初にモデル化を試す候補物性。nonzero_count は「0ではない測定値がある行数」の目安。
# 0は未測定・未入力の可能性が高いため、目標変数では基本的に除外して扱う。
TARGET_CANDIDATES: list[dict[str, Any]] = [
    {"column": "S01010_010010", "name": "比重", "detail": "比重_23ﾟC", "nonzero_count": 5088},
    {"column": "S06020_060030", "name": "引張強さ", "detail": "JIS2 200mm/min", "nonzero_count": 3930},
    {"column": "S06060_060030", "name": "引張伸び", "detail": "JIS2 200mm/min", "nonzero_count": 3892},
    {"column": "S06010_060030", "name": "100%モジュラス", "detail": "JIS2 200mm/min", "nonzero_count": 3838},
    {"column": "S05010_050010_1", "name": "熱安定性", "detail": "180ﾟC", "nonzero_count": 3770},
    {"column": "S13010_130010", "name": "脆化温度", "detail": "JIS K6723", "nonzero_count": 3392},
    {"column": "S03010_030010", "name": "硬度 HDA", "detail": "0s", "nonzero_count": 3046},
]


@dataclass(frozen=True)
class PreparedRegressionData:
    """モデルに渡す直前の X, y と、前処理で落とした列の情報。"""

    X: pd.DataFrame
    y: pd.Series
    target_col: str
    feature_columns: list[str]
    dropped_columns: dict[str, list[str]]
    row_ids: pd.Series


@dataclass(frozen=True)
class RegressionDiagnostics:
    """1つのモデル診断に必要な評価表・予測表・重要特徴量。"""

    metrics: pd.DataFrame
    predictions: pd.DataFrame
    feature_importance: pd.DataFrame
    prepared: PreparedRegressionData


@dataclass(frozen=True)
class GroupedEvaluation:
    """グループをまたがない評価の結果。汎化性能を見るために使う。"""

    metrics: pd.DataFrame
    group_summary: pd.DataFrame
    group_labels: pd.Series
    prepared: PreparedRegressionData


@dataclass(frozen=True)
class ModelEvaluation:
    """単一モデルを通常ランダム分割で評価した結果。"""

    metrics: pd.DataFrame
    predictions: pd.DataFrame
    feature_importance: pd.DataFrame
    prepared: PreparedRegressionData
    model: Any


def target_candidates() -> pd.DataFrame:
    """最初に試す目標変数候補を表として返す。"""
    return pd.DataFrame(TARGET_CANDIDATES)


def evaluation_rules() -> pd.DataFrame:
    """現在の仮評価ルールを、Notebookで見やすい表として返す。"""
    return pd.DataFrame(
        [
            {
                "rule": "通常ランダム分割 R2",
                "metric_key": "random_split_r2",
                "threshold": EVALUATION_RULES["random_split_r2_min"],
                "direction": ">=",
                "purpose": "似た配合を含む補間性能を確認する",
            },
            {
                "rule": "完全一致配合グループ R2",
                "metric_key": "exact_composition_r2",
                "threshold": EVALUATION_RULES["exact_composition_r2_min"],
                "direction": ">=",
                "purpose": "完全重複配合による評価漏洩がないか確認する",
            },
            {
                "rule": "主PVC種 + 主可塑剤グループ R2",
                "metric_key": "primary_pvc_plus_primary_plasticizer_r2",
                "threshold": EVALUATION_RULES["primary_pvc_plus_primary_plasticizer_r2_min"],
                "direction": ">=",
                "purpose": "未知の主要材料系列への汎化性能を確認する",
            },
            {
                "rule": "MAE",
                "metric_key": "mae",
                "threshold": None,
                "direction": "実務閾値を別途設定",
                "purpose": "物性単位で実務上許容できる平均誤差か確認する",
            },
            {
                "rule": "最大外れ誤差",
                "metric_key": "max_abs_error",
                "threshold": None,
                "direction": "実務閾値を別途設定",
                "purpose": "危険な大外れがないか確認する",
            },
        ]
    )


def column_metadata(dataset_name: str = "processed_main") -> pd.DataFrame:
    """processed_main の先頭2行に入っている分類名・原料名を列メタデータにする。"""
    df = read_dataset(dataset_name, nrows=2)
    rows: list[dict[str, Any]] = []
    for column in df.columns:
        category = df[column].iloc[0] if len(df) > 0 else None
        label = df[column].iloc[1] if len(df) > 1 else None
        rows.append(
            {
                "column": column,
                "base_group": column.split(".")[0],
                "category": category,
                "label": label,
                "display_name": readable_column_name(column, category, label),
            }
        )
    return pd.DataFrame(rows)


def readable_column_name(column: str, category: Any = None, label: Any = None) -> str:
    """機械的な列名を、人間が読める「分類 / 原料名 (列名)」にする。"""
    category_text = "" if pd.isna(category) else str(category)
    label_text = "" if pd.isna(label) else str(label)
    if category_text and label_text and category_text != label_text:
        return f"{category_text} / {label_text} ({column})"
    if label_text:
        return f"{label_text} ({column})"
    if category_text:
        return f"{category_text} ({column})"
    return column


def prepare_regression_data(
    dataset_name: str = "processed_main",
    target_col: str = DEFAULT_TARGET,
    *,
    drop_zero_target: bool = True,
    id_columns: tuple[str, ...] = ("Unnamed: 0",),
) -> PreparedRegressionData:
    """回帰モデル用の X, y を作る。

    S で始まる列は物性値なので、説明変数 X から除外する。
    これにより、他の実測物性を使って目標物性を予測してしまう
    データ漏洩を避ける。
    """
    df = read_dataset(dataset_name)
    if target_col not in df.columns:
        raise KeyError(f"target_col is not in {dataset_name}: {target_col}")

    # 目標変数は数値化する。数値化できない値は欠損として扱う。
    target = pd.to_numeric(df[target_col], errors="coerce")
    valid_target = target.notna()
    if drop_zero_target:
        # このデータでは、物性値の0は「本当に0」ではなく未測定の可能性が高い。
        valid_target &= target != 0

    feature_source = df.loc[valid_target].copy()
    y = target.loc[valid_target].astype(float)
    row_ids = (
        feature_source["Unnamed: 0"].astype(str)
        if "Unnamed: 0" in feature_source.columns
        else pd.Series(feature_source.index.astype(str), index=feature_source.index)
    )

    # 物性列とID列は説明変数から除外する。
    excluded_targets = [column for column in df.columns if column.startswith("S")]
    excluded_ids = [column for column in id_columns if column in df.columns]
    excluded = set(excluded_targets + excluded_ids)

    candidate_columns = [column for column in df.columns if column not in excluded]
    X = feature_source[candidate_columns].apply(pd.to_numeric, errors="coerce")

    # 学習に使えない列を落とす。全欠損列や定数列はモデルに情報を与えない。
    all_missing = X.columns[X.isna().all()].tolist()
    X = X.drop(columns=all_missing)

    constant_columns = X.columns[X.nunique(dropna=True) <= 1].tolist()
    X = X.drop(columns=constant_columns)

    # この配合データでは、空欄は「その原料が入っていない」とみなし、0で埋める。
    X = X.fillna(0.0).astype(float)

    return PreparedRegressionData(
        X=X,
        y=y,
        target_col=target_col,
        feature_columns=X.columns.tolist(),
        dropped_columns={
            "id_columns": excluded_ids,
            "other_property_columns": excluded_targets,
            "all_missing_columns": all_missing,
            "constant_columns": constant_columns,
        },
        row_ids=row_ids,
    )


def feature_missing_zero_summary(
    dataset_name: str = "processed_main",
    target_col: str = DEFAULT_TARGET,
) -> pd.DataFrame:
    """説明変数ごとの欠損率・0率を確認する。

    配合データでは0が多いのが自然だが、0が未入力を意味していないか確認するために使う。
    """
    df = read_dataset(dataset_name)
    prepared = prepare_regression_data(dataset_name=dataset_name, target_col=target_col)
    target = pd.to_numeric(df[target_col], errors="coerce")
    valid_target = target.notna() & (target != 0)
    numeric = df.loc[valid_target, prepared.feature_columns].apply(pd.to_numeric, errors="coerce")
    filled = numeric.fillna(0.0)
    metadata = column_metadata(dataset_name).set_index("column")

    rows: list[dict[str, Any]] = []
    for column in prepared.feature_columns:
        series = numeric[column]
        filled_series = filled[column]
        rows.append(
            {
                "column": column,
                "base_group": metadata.at[column, "base_group"],
                "category": metadata.at[column, "category"],
                "label": metadata.at[column, "label"],
                "display_name": metadata.at[column, "display_name"],
                "missing_count": int(series.isna().sum()),
                "missing_rate": float(series.isna().mean()),
                "zero_count": int((filled_series == 0).sum()),
                "zero_rate": float((filled_series == 0).mean()),
                "nonzero_count": int((filled_series != 0).sum()),
                "nonzero_rate": float((filled_series != 0).mean()),
                "unique_count": int(filled_series.nunique(dropna=True)),
            }
        )
    return pd.DataFrame(rows).sort_values(["zero_rate", "nonzero_count"], ascending=[False, True])


def important_feature_review(
    diagnostics: RegressionDiagnostics,
    *,
    top_n: int = 30,
) -> pd.DataFrame:
    """重要特徴量を材料知識でレビューしやすい表にする。

    相関や方向ヒントは因果関係ではない。材料知識と照合するための手がかり。
    """
    X = diagnostics.prepared.X
    y = diagnostics.prepared.y
    rows: list[dict[str, Any]] = []

    for _, feature in diagnostics.feature_importance.head(top_n).iterrows():
        column = feature["column"]
        values = X[column]
        present = values != 0
        # 単純相関で「値が大きいと目標値が上がる/下がる」の方向を粗く見る。
        # 木系モデルの非線形・相互作用はここでは表現しきれない。
        if values.nunique(dropna=True) > 1:
            corr = values.corr(y)
        else:
            corr = float("nan")
        mean_present = y[present].mean() if present.any() else float("nan")
        mean_absent = y[~present].mean() if (~present).any() else float("nan")
        direction = "判定保留"
        if pd.notna(corr):
            if corr > 0.1:
                direction = "増えると目標値が高い傾向"
            elif corr < -0.1:
                direction = "増えると目標値が低い傾向"
            else:
                direction = "単純相関は弱い"
        rows.append(
            {
                "column": column,
                "base_group": feature.get("base_group"),
                "category": feature.get("category"),
                "label": feature.get("label"),
                "display_name": feature.get("display_name"),
                "importance": feature["importance"],
                "nonzero_count": int(present.sum()),
                "nonzero_rate": float(present.mean()),
                "pearson_corr_with_target": corr,
                "mean_target_when_present": mean_present,
                "mean_target_when_absent": mean_absent,
                "data_direction_hint": direction,
                "domain_review_point": "材料知識と照合する",
            }
        )

    return pd.DataFrame(rows)


def _primary_material_label(
    X: pd.DataFrame,
    metadata: pd.DataFrame,
    base_group: str,
) -> pd.Series:
    """PVCや可塑剤など、指定した材料グループ内で最も配合量が多い原料名を返す。"""
    meta = metadata.set_index("column")
    columns = [
        column
        for column in X.columns
        if column in meta.index and meta.at[column, "base_group"] == base_group
    ]
    if not columns:
        return pd.Series("(no columns)", index=X.index)

    values = X[columns]
    labels = meta.loc[columns, "display_name"].to_dict()

    def choose(row: pd.Series) -> str:
        positive = row[row > 0]
        if positive.empty:
            return f"(no {base_group})"
        column = positive.sort_values(ascending=False).index[0]
        return labels[column]

    return values.apply(choose, axis=1)


def group_labels(
    prepared: PreparedRegressionData,
    group_by: str,
    *,
    dataset_name: str = "processed_main",
) -> pd.Series:
    """評価用のグループラベルを作る。

    同じグループが学習側と評価側にまたがらないようにし、
    未知の配合系列への汎化性能を見る。
    """
    metadata = column_metadata(dataset_name)

    if group_by == "exact_composition":
        # Xが完全に同じ行を同じグループにする。
        labels = pd.util.hash_pandas_object(prepared.X, index=False).astype(str)
        labels.index = prepared.X.index
        return labels
    if group_by == "primary_pvc":
        # 最も多く入っているPVC種でグループ化する。
        return _primary_material_label(prepared.X, metadata, "PVC")
    if group_by == "primary_plasticizer":
        # 最も多く入っている可塑剤でグループ化する。
        return _primary_material_label(prepared.X, metadata, "可塑剤オイル類")
    if group_by == "primary_pvc_plus_primary_plasticizer":
        # 実務上の配合系列に近い「主PVC + 主可塑剤」の組み合わせ。
        pvc = _primary_material_label(prepared.X, metadata, "PVC")
        plasticizer = _primary_material_label(prepared.X, metadata, "可塑剤オイル類")
        return pvc + " + " + plasticizer

    raise ValueError(
        "group_by must be one of: exact_composition, primary_pvc, "
        "primary_plasticizer, primary_pvc_plus_primary_plasticizer"
    )


def grouped_random_forest_evaluation(
    dataset_name: str = "processed_main",
    target_col: str = DEFAULT_TARGET,
    *,
    group_by: str,
    test_size: float = 0.2,
    random_state: int = 42,
    n_splits: int = 5,
    n_estimators: int = 200,
) -> GroupedEvaluation:
    """RandomForestをグループ単位で評価する。

    古いNotebookから使っているRandomForest専用関数。新しいモデル比較では
    evaluate_model_grouped() を使う。
    """
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        from sklearn.model_selection import GroupShuffleSplit
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "scikit-learn is required for grouped evaluation. Install dependencies with "
            "`python3 -m pip install -r requirements.txt`."
        ) from exc

    prepared = prepare_regression_data(dataset_name=dataset_name, target_col=target_col)
    labels = group_labels(prepared, group_by, dataset_name=dataset_name)
    group_counts = labels.value_counts()

    # GroupShuffleSplit は、同じグループが train/test に分かれないように分割する。
    splitter = GroupShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
    rows: list[dict[str, Any]] = []
    for split, (train_idx, test_idx) in enumerate(
        splitter.split(prepared.X, prepared.y, groups=labels),
        start=1,
    ):
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
        )
        X_train = prepared.X.iloc[train_idx]
        X_test = prepared.X.iloc[test_idx]
        y_train = prepared.y.iloc[train_idx]
        y_test = prepared.y.iloc[test_idx]
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        test_groups = labels.iloc[test_idx]
        rows.append(
            {
                "group_by": group_by,
                "split": split,
                "train_rows": len(train_idx),
                "test_rows": len(test_idx),
                "train_groups": labels.iloc[train_idx].nunique(),
                "test_groups": test_groups.nunique(),
                "largest_test_group_size": int(test_groups.value_counts().max()),
                "MAE": mean_absolute_error(y_test, pred),
                "RMSE": sqrt(mean_squared_error(y_test, pred)),
                "R2": r2_score(y_test, pred),
            }
        )

    group_summary = pd.DataFrame(
        {
            "group": group_counts.index,
            "rows": group_counts.values,
        }
    )
    group_summary["row_rate"] = group_summary["rows"] / len(labels)

    return GroupedEvaluation(
        metrics=pd.DataFrame(rows),
        group_summary=group_summary,
        group_labels=labels,
        prepared=prepared,
    )


def make_regression_model(
    model_name: str,
    *,
    random_state: int = 42,
    n_estimators: int = 300,
) -> Any:
    """モデル名から回帰モデルを作る。

    Notebook側でモデルごとの細かい初期化を書かなくてよいようにする。
    """
    if model_name == "random_forest":
        from sklearn.ensemble import RandomForestRegressor

        # RandomForestは最初の安定したベースライン。スケール調整なしでも扱いやすい。
        return RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
        )
    if model_name == "xgboost":
        try:
            from xgboost import XGBRegressor
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "xgboost is required. Install dependencies with "
                "`python3 -m pip install -r requirements.txt`."
            ) from exc

        # XGBoostは、未知の主PVC+主可塑剤グループでRandomForestより良い傾向が出た候補。
        return XGBRegressor(
            n_estimators=n_estimators,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=3.0,
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=-1,
        )

    if model_name == "lightgbm":
        try:
            from lightgbm import LGBMRegressor
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "lightgbm is required. Install dependencies with "
                "`python3 -m pip install -r requirements.txt`."
            ) from exc

        # 疎な配合データ向けの初期設定。最終的な採用可否は、Notebookで
        # ランダム分割と材料系列を分けた評価の両方で確認する。
        return LGBMRegressor(
            objective="regression",
            n_estimators=n_estimators,
            learning_rate=0.03,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=random_state,
            # 交差検証では複数モデルを続けて学習するため、プロセス数を抑える。
            n_jobs=1,
            verbosity=-1,
        )

    raise ValueError("model_name must be one of: random_forest, xgboost, lightgbm")


def evaluate_model_random_split(
    model_name: str,
    dataset_name: str = "processed_main",
    target_col: str = DEFAULT_TARGET,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    n_estimators: int = 300,
) -> ModelEvaluation:
    """通常のランダム train/test 分割で1つのモデルを評価する。

    似た配合が train/test に混ざる可能性があるため、補間性能を見る評価として扱う。
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split

    prepared = prepare_regression_data(dataset_name=dataset_name, target_col=target_col)
    X_train, X_test, y_train, y_test = train_test_split(
        prepared.X,
        prepared.y,
        test_size=test_size,
        random_state=random_state,
    )
    model = make_regression_model(
        model_name,
        random_state=random_state,
        n_estimators=n_estimators,
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    # 外れ配合を後で確認できるように、予測値・実測値・誤差を行ID付きで残す。
    predictions = pd.DataFrame(
        {
            "row_id": prepared.row_ids.loc[X_test.index].values,
            "actual": y_test.values,
            "predicted": pred,
            "error": pred - y_test.values,
            "abs_error": abs(pred - y_test.values),
        },
        index=y_test.index,
    ).sort_values("abs_error", ascending=False)

    # MAE/RMSE/R2に加えて、実務上問題になりやすい最大外れ誤差も保存する。
    metrics = pd.DataFrame(
        [
            {
                "model": model_name,
                "evaluation": "random_split",
                "rows": len(y_test),
                "MAE": mean_absolute_error(y_test, pred),
                "RMSE": sqrt(mean_squared_error(y_test, pred)),
                "R2": r2_score(y_test, pred),
                "max_abs_error": predictions["abs_error"].max(),
            }
        ]
    )

    # 木系モデルの feature_importances_ を原料名付きで見られる表にする。
    metadata = column_metadata(dataset_name).set_index("column")
    if hasattr(model, "feature_importances_"):
        raw_importance = model.feature_importances_
    else:
        raw_importance = [0.0] * len(prepared.feature_columns)
    importance = pd.DataFrame(
        {
            "column": prepared.feature_columns,
            "importance": raw_importance,
        }
    )
    importance["base_group"] = importance["column"].map(metadata["base_group"])
    importance["category"] = importance["column"].map(metadata["category"])
    importance["label"] = importance["column"].map(metadata["label"])
    importance["display_name"] = importance["column"].map(metadata["display_name"])
    importance = importance.sort_values("importance", ascending=False)

    return ModelEvaluation(
        metrics=metrics,
        predictions=predictions,
        feature_importance=importance,
        prepared=prepared,
        model=model,
    )


def evaluate_model_grouped(
    model_name: str,
    dataset_name: str = "processed_main",
    target_col: str = DEFAULT_TARGET,
    *,
    group_by: str,
    test_size: float = 0.2,
    random_state: int = 42,
    n_splits: int = 5,
    n_estimators: int = 200,
) -> GroupedEvaluation:
    """グループをまたがないように1つのモデルを評価する。

    通常分割より厳しく、未知のPVC系・可塑剤系などへの汎化性能を見る。
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import GroupShuffleSplit

    prepared = prepare_regression_data(dataset_name=dataset_name, target_col=target_col)
    labels = group_labels(prepared, group_by, dataset_name=dataset_name)
    group_counts = labels.value_counts()

    splitter = GroupShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
    rows: list[dict[str, Any]] = []
    for split, (train_idx, test_idx) in enumerate(
        splitter.split(prepared.X, prepared.y, groups=labels),
        start=1,
    ):
        # 分割ごとに新しいモデルを作り直す。前のsplitの学習結果を持ち越さないため。
        model = make_regression_model(
            model_name,
            random_state=random_state,
            n_estimators=n_estimators,
        )
        X_train = prepared.X.iloc[train_idx]
        X_test = prepared.X.iloc[test_idx]
        y_train = prepared.y.iloc[train_idx]
        y_test = prepared.y.iloc[test_idx]
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        predictions = pd.Series(abs(pred - y_test.values), index=y_test.index)
        test_groups = labels.iloc[test_idx]
        rows.append(
            {
                "model": model_name,
                "group_by": group_by,
                "split": split,
                "train_rows": len(train_idx),
                "test_rows": len(test_idx),
                "train_groups": labels.iloc[train_idx].nunique(),
                "test_groups": test_groups.nunique(),
                "largest_test_group_size": int(test_groups.value_counts().max()),
                "MAE": mean_absolute_error(y_test, pred),
                "RMSE": sqrt(mean_squared_error(y_test, pred)),
                "R2": r2_score(y_test, pred),
                "max_abs_error": predictions.max(),
            }
        )

    group_summary = pd.DataFrame(
        {
            "group": group_counts.index,
            "rows": group_counts.values,
        }
    )
    group_summary["row_rate"] = group_summary["rows"] / len(labels)

    return GroupedEvaluation(
        metrics=pd.DataFrame(rows),
        group_summary=group_summary,
        group_labels=labels,
        prepared=prepared,
    )


def compare_models_with_rules(
    model_names: tuple[str, ...] = ("random_forest", "xgboost"),
    dataset_name: str = "processed_main",
    target_col: str = DEFAULT_TARGET,
    *,
    random_state: int = 42,
    random_n_estimators: int = 300,
    grouped_n_estimators: int = 200,
    n_splits: int = 5,
) -> pd.DataFrame:
    """固定した評価ルールで複数モデルを横並び比較する。

    通常分割、完全一致配合、主PVC、主可塑剤、主PVC+主可塑剤を同じ表にまとめる。
    """
    groupings = {
        "exact_composition": "完全一致配合",
        "primary_pvc": "主PVC種",
        "primary_plasticizer": "主可塑剤",
        "primary_pvc_plus_primary_plasticizer": "主PVC種 + 主可塑剤",
    }

    rows: list[dict[str, Any]] = []
    for model_name in model_names:
        # まず通常ランダム分割で、似た配合を含む補間性能を見る。
        random_eval = evaluate_model_random_split(
            model_name,
            dataset_name=dataset_name,
            target_col=target_col,
            random_state=random_state,
            n_estimators=random_n_estimators,
        )
        random_metrics = random_eval.metrics.iloc[0]
        rows.append(
            {
                "model": model_name,
                "evaluation": "通常ランダム分割",
                "evaluation_key": "random_split",
                "mean_MAE": random_metrics["MAE"],
                "mean_RMSE": random_metrics["RMSE"],
                "mean_R2": random_metrics["R2"],
                "max_abs_error": random_metrics["max_abs_error"],
                "splits": 1,
            }
        )

        # 次にグループ評価で、未知系列への汎化性能を見る。
        for group_by, label in groupings.items():
            grouped = evaluate_model_grouped(
                model_name,
                dataset_name=dataset_name,
                target_col=target_col,
                group_by=group_by,
                random_state=random_state,
                n_splits=n_splits,
                n_estimators=grouped_n_estimators,
            )
            metrics = grouped.metrics
            rows.append(
                {
                    "model": model_name,
                    "evaluation": label,
                    "evaluation_key": group_by,
                    "mean_MAE": metrics["MAE"].mean(),
                    "mean_RMSE": metrics["RMSE"].mean(),
                    "mean_R2": metrics["R2"].mean(),
                    "std_R2": metrics["R2"].std(),
                    "max_abs_error": metrics["max_abs_error"].max(),
                    "splits": len(metrics),
                    "groups": len(grouped.group_summary),
                    "max_group_size": int(grouped.group_summary["rows"].max()),
                }
            )

    result = pd.DataFrame(rows)
    return add_rule_judgements(result)


def add_rule_judgements(comparison: pd.DataFrame) -> pd.DataFrame:
    """比較表に、固定ルールの合否を付ける。

    MAEと最大外れ誤差はまだ実務閾値がないため、ここではR2系ルールだけ判定する。
    """
    result = comparison.copy()
    result["rule_threshold"] = None
    result["rule_pass"] = None
    result["rule_note"] = ""

    random_mask = result["evaluation_key"] == "random_split"
    result.loc[random_mask, "rule_threshold"] = EVALUATION_RULES["random_split_r2_min"]
    result.loc[random_mask, "rule_pass"] = (
        result.loc[random_mask, "mean_R2"] >= EVALUATION_RULES["random_split_r2_min"]
    )
    result.loc[random_mask, "rule_note"] = "通常ランダム分割 R2 >= 0.75"

    exact_mask = result["evaluation_key"] == "exact_composition"
    result.loc[exact_mask, "rule_threshold"] = EVALUATION_RULES["exact_composition_r2_min"]
    result.loc[exact_mask, "rule_pass"] = (
        result.loc[exact_mask, "mean_R2"] >= EVALUATION_RULES["exact_composition_r2_min"]
    )
    result.loc[exact_mask, "rule_note"] = "完全一致配合 R2 >= 0.70"

    combo_mask = result["evaluation_key"] == "primary_pvc_plus_primary_plasticizer"
    result.loc[combo_mask, "rule_threshold"] = EVALUATION_RULES[
        "primary_pvc_plus_primary_plasticizer_r2_min"
    ]
    result.loc[combo_mask, "rule_pass"] = (
        result.loc[combo_mask, "mean_R2"]
        >= EVALUATION_RULES["primary_pvc_plus_primary_plasticizer_r2_min"]
    )
    result.loc[combo_mask, "rule_note"] = "主PVC + 主可塑剤 R2 >= 0.50"

    no_threshold = result["rule_note"] == ""
    result.loc[no_threshold, "rule_note"] = "参考評価"
    return result


def random_forest_diagnostics(
    dataset_name: str = "processed_main",
    target_col: str = DEFAULT_TARGET,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    n_estimators: int = 300,
) -> RegressionDiagnostics:
    """RandomForestの初期診断をまとめて行う。

    dummy_mean と比較し、予測表・外れ配合・重要特徴量をNotebookで使いやすい形で返す。
    """
    try:
        from sklearn.dummy import DummyRegressor
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        from sklearn.model_selection import train_test_split
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "scikit-learn is required for model training. Install dependencies with "
            "`python3 -m pip install -r requirements.txt`."
        ) from exc

    prepared = prepare_regression_data(dataset_name=dataset_name, target_col=target_col)
    X_train, X_test, y_train, y_test = train_test_split(
        prepared.X,
        prepared.y,
        test_size=test_size,
        random_state=random_state,
    )
    row_ids_test = prepared.row_ids.loc[X_test.index]

    # dummy_mean は「常に平均値を予測するだけ」の最低限の比較基準。
    models = {
        "dummy_mean": DummyRegressor(strategy="mean"),
        "random_forest": RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
        ),
    }

    metric_rows: list[dict[str, float | str]] = []
    predictions: pd.DataFrame | None = None
    random_forest: Any | None = None
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        metric_rows.append(
            {
                "model": name,
                "MAE": mean_absolute_error(y_test, pred),
                "RMSE": sqrt(mean_squared_error(y_test, pred)),
                "R2": r2_score(y_test, pred),
            }
        )
        if name == "random_forest":
            # 外れが大きい順に並べ、後で配合を確認しやすくする。
            random_forest = model
            predictions = pd.DataFrame(
                {
                    "row_id": row_ids_test.values,
                    "actual": y_test.values,
                    "predicted": pred,
                    "error": pred - y_test.values,
                    "abs_error": abs(pred - y_test.values),
                },
                index=y_test.index,
            ).sort_values("abs_error", ascending=False)

    if random_forest is None or predictions is None:
        raise RuntimeError("RandomForest model was not trained.")

    # 重要特徴量を、可塑剤オイル類.29 のような列名ではなく原料名付きで表示する。
    metadata = column_metadata(dataset_name).set_index("column")
    importance = pd.DataFrame(
        {
            "column": prepared.feature_columns,
            "importance": random_forest.feature_importances_,
        }
    )
    importance["base_group"] = importance["column"].map(metadata["base_group"])
    importance["category"] = importance["column"].map(metadata["category"])
    importance["label"] = importance["column"].map(metadata["label"])
    importance["display_name"] = importance["column"].map(metadata["display_name"])
    importance = importance.sort_values("importance", ascending=False)

    return RegressionDiagnostics(
        metrics=pd.DataFrame(metric_rows).sort_values("MAE"),
        predictions=predictions,
        feature_importance=importance,
        prepared=prepared,
    )


def run_baseline_models(
    dataset_name: str = "processed_main",
    target_col: str = DEFAULT_TARGET,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    n_estimators: int = 300,
    include_ridge: bool = False,
) -> tuple[pd.DataFrame, PreparedRegressionData, dict[str, Any]]:
    """ベースラインモデルをまとめて学習する。

    train_baseline.py から使うための簡易関数。通常は random_forest_diagnostics() や
    evaluate_model_random_split() を使う。
    """
    try:
        from sklearn.dummy import DummyRegressor
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import Ridge
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "scikit-learn is required for model training. Install dependencies with "
            "`python3 -m pip install -r requirements.txt`."
        ) from exc

    prepared = prepare_regression_data(dataset_name=dataset_name, target_col=target_col)
    X_train, X_test, y_train, y_test = train_test_split(
        prepared.X,
        prepared.y,
        test_size=test_size,
        random_state=random_state,
    )

    # Ridgeは任意。特徴量が多く相関も強いため、警告が出る場合がある。
    models = {
        "dummy_mean": DummyRegressor(strategy="mean"),
        "random_forest": RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
        ),
    }
    if include_ridge:
        models["ridge_scaled"] = make_pipeline(StandardScaler(), Ridge(alpha=1.0))

    rows: list[dict[str, float | str]] = []
    fitted_models: dict[str, Any] = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        fitted_models[name] = model
        pred = model.predict(X_test)
        rows.append(
            {
                "model": name,
                "MAE": mean_absolute_error(y_test, pred),
                "RMSE": sqrt(mean_squared_error(y_test, pred)),
                "R2": r2_score(y_test, pred),
            }
        )

    return pd.DataFrame(rows).sort_values("MAE"), prepared, fitted_models

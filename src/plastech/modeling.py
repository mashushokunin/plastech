from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Any

import pandas as pd

from plastech.data_catalog import read_dataset


DEFAULT_TARGET = "S06020_060030"

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
    X: pd.DataFrame
    y: pd.Series
    target_col: str
    feature_columns: list[str]
    dropped_columns: dict[str, list[str]]
    row_ids: pd.Series


@dataclass(frozen=True)
class RegressionDiagnostics:
    metrics: pd.DataFrame
    predictions: pd.DataFrame
    feature_importance: pd.DataFrame
    prepared: PreparedRegressionData


def target_candidates() -> pd.DataFrame:
    """Return the first target candidates found during data inspection."""
    return pd.DataFrame(TARGET_CANDIDATES)


def column_metadata(dataset_name: str = "processed_main") -> pd.DataFrame:
    """Return readable labels stored in the first two rows of the dataset."""
    df = read_dataset(dataset_name, nrows=2)
    rows: list[dict[str, Any]] = []
    for column in df.columns:
        category = df[column].iloc[0] if len(df) > 0 else None
        label = df[column].iloc[1] if len(df) > 1 else None
        rows.append(
            {
                "column": column,
                "category": category,
                "label": label,
                "display_name": readable_column_name(column, category, label),
            }
        )
    return pd.DataFrame(rows)


def readable_column_name(column: str, category: Any = None, label: Any = None) -> str:
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
    """Prepare a simple numeric regression table.

    The processed data keeps property columns with an ``S`` prefix. Those columns
    are excluded from X so the model cannot use another measured property to
    predict the target property.
    """
    df = read_dataset(dataset_name)
    if target_col not in df.columns:
        raise KeyError(f"target_col is not in {dataset_name}: {target_col}")

    target = pd.to_numeric(df[target_col], errors="coerce")
    valid_target = target.notna()
    if drop_zero_target:
        valid_target &= target != 0

    feature_source = df.loc[valid_target].copy()
    y = target.loc[valid_target].astype(float)
    row_ids = (
        feature_source["Unnamed: 0"].astype(str)
        if "Unnamed: 0" in feature_source.columns
        else pd.Series(feature_source.index.astype(str), index=feature_source.index)
    )

    excluded_targets = [column for column in df.columns if column.startswith("S")]
    excluded_ids = [column for column in id_columns if column in df.columns]
    excluded = set(excluded_targets + excluded_ids)

    candidate_columns = [column for column in df.columns if column not in excluded]
    X = feature_source[candidate_columns].apply(pd.to_numeric, errors="coerce")

    all_missing = X.columns[X.isna().all()].tolist()
    X = X.drop(columns=all_missing)

    constant_columns = X.columns[X.nunique(dropna=True) <= 1].tolist()
    X = X.drop(columns=constant_columns)

    # In this data, empty material amount cells are safer to treat as absent.
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


def random_forest_diagnostics(
    dataset_name: str = "processed_main",
    target_col: str = DEFAULT_TARGET,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    n_estimators: int = 300,
) -> RegressionDiagnostics:
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

    metadata = column_metadata(dataset_name).set_index("column")
    importance = pd.DataFrame(
        {
            "column": prepared.feature_columns,
            "importance": random_forest.feature_importances_,
        }
    )
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
    """Train baseline regression models and return their test metrics."""
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

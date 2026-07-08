from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Any, Sequence

import numpy as np
import pandas as pd

from plastech.modeling import PreparedRegressionData, make_regression_model, prepare_regression_data


@dataclass(frozen=True)
class CrossValidatedPredictions:
    """目標値選定に使う、交差検証のout-of-fold予測。"""

    model_name: str
    target_col: str
    prepared: PreparedRegressionData
    predictions: pd.DataFrame
    metrics: pd.DataFrame


def cross_validated_forward_predictions(
    target_col: str,
    *,
    dataset_name: str = "processed_main",
    model_name: str = "lightgbm",
    n_splits: int = 5,
    n_estimators: int = 800,
    random_state: int = 42,
) -> CrossValidatedPredictions:
    """各行を学習に含めない予測値（OOF予測）を作る。

    目標値を決めるとき、学習に使った行への予測誤差では楽観的になる。
    OOF予測を使うことで、物性値の近傍ごとの誤差とデータ密度を確認する。
    """
    try:
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        from sklearn.model_selection import KFold
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "scikit-learn is required. Install dependencies with "
            "`python3 -m pip install -r requirements.txt`."
        ) from exc

    prepared = prepare_regression_data(dataset_name=dataset_name, target_col=target_col)
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof_prediction = np.full(len(prepared.y), np.nan)
    fold_numbers = np.zeros(len(prepared.y), dtype=int)

    for fold, (train_idx, test_idx) in enumerate(splitter.split(prepared.X), start=1):
        model = make_regression_model(
            model_name,
            random_state=random_state + fold,
            n_estimators=n_estimators,
        )
        model.fit(prepared.X.iloc[train_idx], prepared.y.iloc[train_idx])
        oof_prediction[test_idx] = model.predict(prepared.X.iloc[test_idx])
        fold_numbers[test_idx] = fold

    actual = prepared.y.reset_index(drop=True)
    prediction = pd.Series(oof_prediction)
    predictions = pd.DataFrame(
        {
            "row_id": prepared.row_ids.reset_index(drop=True),
            "fold": fold_numbers,
            "actual": actual,
            "predicted": prediction,
        }
    )
    predictions["error"] = predictions["predicted"] - predictions["actual"]
    predictions["abs_error"] = predictions["error"].abs()
    metrics = pd.DataFrame(
        [
            {
                "model": model_name,
                "target_col": target_col,
                "rows": len(predictions),
                "n_splits": n_splits,
                "MAE": mean_absolute_error(actual, prediction),
                "RMSE": sqrt(mean_squared_error(actual, prediction)),
                "R2": r2_score(actual, prediction),
                "p90_abs_error": predictions["abs_error"].quantile(0.9),
            }
        ]
    )
    return CrossValidatedPredictions(
        model_name=model_name,
        target_col=target_col,
        prepared=prepared,
        predictions=predictions,
        metrics=metrics,
    )


def target_value_profile(
    cross_validated: CrossValidatedPredictions,
    *,
    target_tolerance: float,
    target_values: Sequence[float] | None = None,
    quantiles: Sequence[float] = tuple(np.linspace(0.1, 0.9, 17)),
    rounding: int | None = None,
    min_support: int = 100,
) -> pd.DataFrame:
    """候補となる目標値ごとのデータ密度と局所OOF誤差を比較する。

    ``target_tolerance`` は目標値の前後をどこまで同じ用途領域とみなすかの幅。
    スコアが小さいほど、周辺の実測例が多く、かつOOF誤差が小さい。
    """
    if target_tolerance <= 0:
        raise ValueError("target_tolerance must be positive.")
    if min_support <= 0:
        raise ValueError("min_support must be positive.")

    predictions = cross_validated.predictions
    actual = predictions["actual"]
    if target_values is None:
        values = actual.quantile(list(quantiles)).to_numpy(dtype=float)
    else:
        values = np.asarray(target_values, dtype=float)
    if rounding is not None:
        values = np.round(values, rounding)
    values = np.unique(values)

    global_mae = predictions["abs_error"].mean()
    rows: list[dict[str, Any]] = []
    for value in values:
        near = predictions[actual.sub(value).abs().le(target_tolerance)]
        support = len(near)
        if support == 0:
            continue
        local_mae = near["abs_error"].mean()
        # 少数の偶然に誤差が小さい領域を選ばないため、必要標本数に満たない分を加点する。
        support_penalty = global_mae * sqrt(min_support / support)
        rows.append(
            {
                "target_value": float(value),
                "window": target_tolerance,
                "support": support,
                "support_rate": support / len(predictions),
                "local_MAE": local_mae,
                "local_RMSE": sqrt((near["error"] ** 2).mean()),
                "local_p90_abs_error": near["abs_error"].quantile(0.9),
                "local_bias": near["error"].mean(),
                "eligible": support >= min_support,
                "reliability_score": local_mae + support_penalty,
            }
        )

    profile = pd.DataFrame(rows)
    if profile.empty:
        return profile
    # 必要標本数を満たす目標値を先にし、その中で局所誤差とサンプル数を両立させる。
    return profile.sort_values(
        ["eligible", "reliability_score", "local_p90_abs_error", "support"],
        ascending=[False, True, True, False],
        kind="stable",
    ).reset_index(drop=True)


def compare_target_value(
    profile: pd.DataFrame,
    current_target: float,
) -> pd.DataFrame:
    """現在の目標値と、データに基づく候補値を同じ表で比較する。"""
    if profile.empty:
        return profile
    result = profile.copy()
    result["is_current_target"] = np.isclose(result["target_value"], current_target)
    return result

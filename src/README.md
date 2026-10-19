# src ディレクトリの説明

`src` は、Notebook や `scripts` から共通利用する Python コードを置く場所です。

このプロジェクトでは `src/plastech` を Python パッケージとして扱います。基本的には、各 `.py` ファイルを直接実行するのではなく、Notebook か `scripts/*.py` から import して使います。

## 実行時の注意

`src/plastech/modeling.py` などを直接 `python3 modeling.py` のように実行すると、`plastech` パッケージが見つからず、次のようなエラーになることがあります。

```text
ModuleNotFoundError: No module named 'plastech'
```

これはコードの中で `from plastech.data_catalog import read_dataset` のように、パッケージとして import しているためです。

通常はプロジェクトルートから、用意済みスクリプトを実行します。

```bash
python3 scripts/train_baseline.py
python3 scripts/run_inverse_design.py --concept all
```

Python から直接 import して確認したい場合は、`PYTHONPATH=src` を付けます。

```bash
PYTHONPATH=src python3 -c "from plastech.modeling import target_candidates; print(target_candidates())"
```

## ファイル構成

```text
src/
└── plastech/
    ├── __init__.py
    ├── data_catalog.py
    ├── modeling.py
    ├── target_selection.py
    └── inverse_design.py
```

## `plastech/__init__.py`

`plastech` を Python パッケージとして認識させるためのファイルです。

現時点では大きな処理は持たず、パッケージの入口として機能します。

## `plastech/data_catalog.py`

データセットの読み込みを共通化するためのファイルです。

実データのファイル名を Notebook やモデルコードに直接書かず、`config/data_catalog.local.json` に定義した論理名で読み込みます。

主な役割:

- プロジェクトルートを探す
- `config/data_catalog.local.json` を読み込む
- `processed_main` などの論理名からCSVのパスを取得する
- pandas でCSVを読み込む

主な関数:

- `project_root()`
- `load_catalog()`
- `dataset_info()`
- `dataset_path()`
- `list_datasets()`
- `read_dataset()`

使用例:

```python
from plastech.data_catalog import read_dataset

df = read_dataset("processed_main")
```

## `plastech/modeling.py`

順方向モデル、つまり「配合情報 -> 物性値」を予測するモデルを作るための共通コードです。

逆解析では、まず順方向モデルを学習し、そのモデルで候補配合の物性を予測します。そのため、このファイルは逆解析学習機の土台です。

主な役割:

- 物性列を説明変数から除外して、データ漏洩を避ける
- 0の物性値を未測定扱いとして学習対象から外す
- Random Forest、XGBoost、LightGBM のモデルを作る
- 通常ランダム分割でモデルを評価する
- 主PVCや主可塑剤などのグループをまたがない評価をする
- 特徴量重要度を原料名付きで確認できる形にする

主なクラス:

- `PreparedRegressionData`
- `RegressionDiagnostics`
- `GroupedEvaluation`
- `ModelEvaluation`

主な関数:

- `prepare_regression_data()`
- `make_regression_model()`
- `evaluate_model_random_split()`
- `evaluate_model_grouped()`
- `compare_models_with_rules()`
- `random_forest_diagnostics()`
- `run_baseline_models()`
- `column_metadata()`
- `important_feature_review()`

関連スクリプト:

```bash
python3 scripts/train_baseline.py
```

## `plastech/target_selection.py`

逆解析で使う目標値を検討するためのファイルです。

単に「狙いたい値」を決めるだけでなく、その値の近くに過去データがどれくらいあるか、OOF予測でどれくらい誤差が出るかを見ます。

主な役割:

- out-of-fold 予測を作る
- 学習に使っていない行への予測誤差を確認する
- 候補目標値ごとのデータ密度と局所誤差を集計する
- 現在の目標値と、データ上扱いやすい候補値を比較する

主なクラス:

- `CrossValidatedPredictions`

主な関数:

- `cross_validated_forward_predictions()`
- `target_value_profile()`
- `compare_target_value()`

## `plastech/inverse_design.py`

逆解析の中心となるファイルです。

ここでの逆解析は、物性値から直接1つの配合を回帰する方式ではありません。順方向モデルを使って候補配合の物性を予測し、目標物性に近い候補を探す方式です。

主な役割:

- 用途別の目標物性を定義する
- 物性ごとに順方向モデルを学習する
- 既存配合ライブラリを作る
- 主目標に近い既存配合を起点として選ぶ
- 主PVC・主可塑剤が同じ配合どうしを補間して仮想候補を作る
- 候補に予測物性とスコアを付ける
- Pareto最適候補を抽出する
- 材料系列が偏りすぎないように推薦候補を選ぶ
- 推薦候補ごとに未達物性や親配合との差分を確認する

主なクラス:

- `TargetSpecification`
- `ForwardModelSet`
- `DesignBrief`
- `InverseDesignResult`

主な関数:

- `fit_forward_models()`
- `build_candidate_library()`
- `rank_seed_candidates()`
- `generate_interpolated_candidates()`
- `score_design_preferences()`
- `add_preference_status()`
- `add_design_brief_status()`
- `add_formulation_material_count()`
- `select_diverse_candidates()`
- `parent_formulation_deltas()`
- `run_inverse_design_for_brief()`
- `candidate_summary()`

`DesignBrief`は物性を次の役割に分けられる。

- `preferences` / `objectives`: 候補順位とPareto判定に使用
- `management_targets`: 狙い範囲を満たす候補を優先
- `constraints`: 最低条件を満たす候補を優先
- `monitors`: 予測と表示のみ。順位・Pareto・必要条件には不使用

既存の4物性一括方式は`DEFAULT_DESIGN_BRIEFS`、役割分離方式は
`ROLE_BASED_DESIGN_BRIEFS`に分けてあり、新旧比較を再現できる。

関連スクリプト:

```bash
python3 scripts/run_inverse_design.py --concept all
```

CSV出力まで行う例:

```bash
python3 scripts/run_inverse_design.py \
  --concept all \
  --n-estimators 800 \
  --seed-count 30 \
  --partners-per-seed 5 \
  --top-n 5 \
  --output-dir outputs/inverse_design
```

## 現在の処理の流れ

```text
processed_main
  ↓
data_catalog.py で読み込み
  ↓
modeling.py で順方向モデル用データを作成
  ↓
modeling.py / inverse_design.py で物性ごとの順方向モデルを学習
  ↓
inverse_design.py で既存配合をもとに仮想候補を生成
  ↓
順方向モデルで候補物性を予測
  ↓
用途別の目標物性で候補を採点
  ↓
推薦候補、未達物性、親配合との差分を出力
```


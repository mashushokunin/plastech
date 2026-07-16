# notebooks ディレクトリの説明

`notebooks` は、データ確認、モデル診断、順方向モデル評価、逆解析検討を段階的に進めるための作業用Notebook置き場です。

Notebookを追加したら、このREADMEの「Notebook一覧」に1行追加してください。番号は、作業の流れが分かるように `00_`, `01_`, `02_` のような連番で付けます。

## 更新ルール

- 新しいNotebookを作ったら、目的、主な入力、主な出力、現在の位置づけをこのREADMEに追記する。
- Notebook内で試した処理を再利用する段階になったら、共通処理は `src/plastech` に移す。
- Notebookは「分析と判断」を残す場所にし、繰り返し使う処理はスクリプトやモジュールに寄せる。
- ファイル名は `番号_内容.ipynb` にする。
- 逆解析やモデル評価の重要な結論は、必要に応じて `docs/` にもまとめる。

## Notebook一覧

| Notebook | 目的 | 主な入力 | 主な出力・確認内容 | 位置づけ |
|---|---|---|---|---|
| `00_dataset_overview.ipynb` | データカタログに登録されたデータセット一覧を確認する | `config/data_catalog.local.json` | 利用可能な論理データ名、パス、文字コード、説明 | 最初に見る入口 |
| `01_raw_input_check.ipynb` | 配合・物性入力用の元データを確認する | `raw_input` | 行数、列数、先頭行、基本的な構造 | 元データ確認 |
| `02_raw_inverse_check.ipynb` | 逆解析用の元データを確認する | `raw_inverse` | 行数、列数、先頭行、基本的な構造 | 元データ確認 |
| `03_processed_main_check.ipynb` | 主分析用の加工済みデータを確認する | `processed_main` | 学習・評価で使う主データの構造 | 主データ確認 |
| `04_processed_snapshot_check.ipynb` | 過去時点の加工済みデータを確認する | `processed_snapshot` | 比較用スナップショットの構造 | 比較用データ確認 |
| `05_model_diagnostics.ipynb` | 引張強さを例に、最初の順方向モデル診断を行う | `processed_main` | RandomForestの予測性能、重要特徴量、外れ配合、複数物性の初期比較 | 順方向モデルの初期診断 |
| `06_error_feature_zero_analysis.ipynb` | 外れ配合、重要特徴量、0の扱いを深掘りする | `processed_main` | 外れ配合の傾向、原料別重要度、欠損率・0率 | データ品質と誤差要因の確認 |
| `07_grouped_evaluation.ipynb` | 配合グループ単位でモデルを評価する | `processed_main` | 完全一致配合、主PVC、主可塑剤、主PVC+主可塑剤で分けた評価 | 未知系列への汎化確認 |
| `08_model_comparison_rules.ipynb` | 評価ルールを固定し、RandomForestとXGBoostを比較する | `processed_main` | R2、MAE、最大外れ誤差、ルール判定 | 採用モデル候補の比較 |
| `09_shap_and_rules.ipynb` | XGBoostのSHAPを使い、影響方向と明示ルール候補を確認する | `processed_main` | SHAP重要度、影響方向、外れ配合、材料ルール候補 | モデル解釈 |
| `10_inverse_design.ipynb` | 目標物性から配合候補を探索する基本形を作る | `processed_main` | 既存配合ランキング、仮想配合生成、順方向モデルによる候補採点 | 逆解析の初期実装 |
| `11_lightgbm_target_selection.ipynb` | LightGBMを評価し、逆解析の目標値候補を検討する | `processed_main` | RandomForest/LightGBM比較、OOF誤差、目標値周辺のデータ密度 | 目標値とモデル選定 |
| `12_lightgbm_inverse_simulation.ipynb` | 作りたい材料像から始める逆解析を試す | `processed_main` | 用途別目標、LightGBM順方向モデル、Pareto候補、推薦候補 | 用途別逆解析の試作 |

## 現在の作業の流れ

```text
00-04: データの存在と構造を確認
  ↓
05-06: 順方向モデルの初期診断とデータ品質確認
  ↓
07-09: 評価方法を厳しくし、モデル比較と解釈を行う
  ↓
10: 逆解析の基本形を作る
  ↓
11: LightGBMと目標値候補を検討する
  ↓
12: 用途別の材料像から候補を推薦する
```

現在は、`12_lightgbm_inverse_simulation.ipynb` で試した用途別逆解析の考え方を `src/plastech/inverse_design.py` と `scripts/run_inverse_design.py` に移し、Notebookを開かなくても候補確認できる形へ整理している段階です。

## Notebook追加時のテンプレート

新しいNotebookを追加したら、以下の形式でこのREADMEに追記します。

```markdown
| `13_example.ipynb` | 何を確認するNotebookか | 主に使うデータセットやファイル | 何が分かるか、何を出力するか | 作業上の位置づけ |
```

Notebookの先頭Markdownにも、最低限以下を書いておくと後から追いやすくなります。

```markdown
# 13 タイトル

このNotebookの目的:

- 何を確認するか
- どのデータを使うか
- この結果を次にどう使うか
```

## 運用メモ

- `00-04` はデータ確認系なので、データカタログや元データが変わったときに見直す。
- `05-09` は順方向モデル評価系なので、特徴量やモデル候補を変えたときに見直す。
- `10-12` は逆解析系なので、用途仕様、目標値、候補生成方法を変えたときに見直す。
- 繰り返し実行する処理は `scripts/` に移し、共通関数は `src/plastech/` に置く。


# plastech

プラスチック材料の配合データと物性データを用いて、機械学習による材料開発を試すためのプロジェクト。

## 目的

材料の配合と物性の関係を分析し、将来的には希望する物性から必要な材料配合を推定するモデルを作る。

## ディレクトリ構成

- data/: CSVなどの元データ
- notebooks/: Jupyter Notebookでの分析・実験
- src/: 後で整理したPythonコード
- docs/: 学習メモや設計メモ
- config/: データセットの論理名とローカルファイルの対応表

## データの扱い方

実データのファイル名はコードやNotebookに直接書かず、論理名で扱う。

- `config/data_catalog.local.json`: 自分の環境だけで使う実ファイル名の対応表
- `config/data_catalog.example.json`: Gitで共有する対応表のサンプル
- `src/plastech/data_catalog.py`: 論理名からCSVを読み込むためのヘルパー

例:

```python
from plastech.data_catalog import read_dataset

df = read_dataset("raw_input")
```

現在の論理名:

- `raw_input`: 配合・物性入力用の元データ
- `raw_inverse`: 逆解析用の元データ
- `processed_main`: 主に分析で使う加工済みデータ
- `processed_snapshot`: 過去時点の加工済みデータ

### Notebookの分け方

複数のデータを1つのNotebookで処理し続けると、前処理や確認結果が混ざりやすい。
基本的には、参照する論理データごとにNotebookを分ける。

- `notebooks/00_dataset_overview.ipynb`: データセット一覧の確認
- `notebooks/01_raw_input_check.ipynb`: `raw_input` 用
- `notebooks/02_raw_inverse_check.ipynb`: `raw_inverse` 用
- `notebooks/03_processed_main_check.ipynb`: `processed_main` 用
- `notebooks/04_processed_snapshot_check.ipynb`: `processed_snapshot` 用

各Notebookでは、対象データだけを読み込んで処理する。

### 論理名の命名規則

論理名は `<stage>_<role>` の形にする。

- `stage`: `raw`, `interim`, `processed`, `reference`, `sample`
- `role`: `input`, `inverse`, `main`, `snapshot`, `features`, `targets`, `master`

実ファイル名、日付、社内資料名、`latest` や `final` のように意味が変わりやすい名前は避ける。
日付や版を残したい場合は、論理名ではなく `description` に書く。

## 最初にやったこと

- Jupyter Notebook環境の準備
- pandasのインストール
- CSVファイルの読み込み
- df.head(), df.shape, df.info(), df.describe() による初期確認

## 逆解析の再実行

従来の4物性一括方式と、物性を最適化・管理目標・制約・監視に分けた方式を比較する。

```bash
python3 scripts/analyze_role_based_design.py
```

この再解析では、`config/inverse_design_constraints.example.json` の処方制約と
`config/prediction_error_reference.json` のOOF p90予測誤差を読み込み、次を出力する。

- `experiment_candidates_by_profile.csv`: 性能重視・単純処方重視・監視値重視の各5候補
- `candidate_review_sheet.csv`: 会議後の採否、除外理由、加工・コスト・法規在庫確認欄

制約ファイルを複製・編集した場合は、`--constraints-config` で指定する。

```bash
python3 scripts/analyze_role_based_design.py \
  --constraints-config config/inverse_design_constraints.example.json
```

柔軟・低温用途について、役割分離、暫定10材料以下、主PVC・主可塑剤を特定可能という条件で候補を出す。

```bash
python3 scripts/run_inverse_design.py \
  --design-mode role_based \
  --concept "柔軟・低温用途" \
  --max-material-count 10 \
  --require-identified-primary-materials \
  --output-dir outputs/role_based_cli
```

自動テストは次のコマンドで実行する。

```bash
python3 -m unittest discover -s tests -v
```

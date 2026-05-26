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

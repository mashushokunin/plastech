# plastech

プラスチック材料の配合データと物性データを用いて、機械学習による材料開発を試すためのプロジェクト。

## 目的

材料の配合と物性の関係を分析し、将来的には希望する物性から必要な材料配合を推定するモデルを作る。

## ディレクトリ構成

- data/: CSVなどの元データ
- notebooks/: Jupyter Notebookでの分析・実験
- src/: 後で整理したPythonコード
- docs/: 学習メモや設計メモ

## 最初にやったこと

- Jupyter Notebook環境の準備
- pandasのインストール
- CSVファイルの読み込み
- df.head(), df.shape, df.info(), df.describe() による初期確認
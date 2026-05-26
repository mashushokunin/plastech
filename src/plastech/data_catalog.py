from __future__ import annotations

import json
from pathlib import Path
from typing import Any


CATALOG_PATH = Path("config/data_catalog.local.json")
EXAMPLE_CATALOG_PATH = Path("config/data_catalog.example.json")


def project_root(start: Path | None = None) -> Path:
    """Find the repository root from scripts or notebooks."""
    current = (start or Path.cwd()).resolve()
    for path in (current, *current.parents):
        if (path / ".git").exists() or (path / "README.md").exists():
            return path
    return current


def load_catalog(root: Path | None = None) -> dict[str, Any]:
    root_path = project_root(root)
    catalog_path = root_path / CATALOG_PATH
    if not catalog_path.exists():
        example_path = root_path / EXAMPLE_CATALOG_PATH
        raise FileNotFoundError(
            f"{CATALOG_PATH} がありません。{EXAMPLE_CATALOG_PATH} を参考にローカル用の対応表を作成してください。"
        )

    with catalog_path.open(encoding="utf-8") as file:
        catalog = json.load(file)

    if "datasets" not in catalog:
        raise KeyError(f"{CATALOG_PATH} に datasets がありません。")
    return catalog


def dataset_info(name: str, root: Path | None = None) -> dict[str, Any]:
    datasets = load_catalog(root)["datasets"]
    if name not in datasets:
        available = ", ".join(sorted(datasets))
        raise KeyError(f"Unknown dataset: {name}. Available datasets: {available}")
    return datasets[name]


def dataset_path(name: str, root: Path | None = None) -> Path:
    root_path = project_root(root)
    info = dataset_info(name, root_path)
    return root_path / info["path"]


def list_datasets(root: Path | None = None) -> list[str]:
    return sorted(load_catalog(root)["datasets"])


def read_dataset(name: str, root: Path | None = None, **kwargs: Any):
    import pandas as pd

    info = dataset_info(name, root)
    encoding = kwargs.pop("encoding", info.get("encoding", "utf-8"))
    kwargs.setdefault("low_memory", False)
    return pd.read_csv(dataset_path(name, root), encoding=encoding, **kwargs)

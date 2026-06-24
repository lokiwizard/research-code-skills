"""数据集包：按名字把配置变成数据集实例。

新增数据集：建文件 + 在下面 import + 往 `_DATASETS` 加一行。
"""

from __future__ import annotations

from typing import Any, Dict

from torch.utils.data import Dataset

from .synthetic import SyntheticRegression

_DATASETS = {
    "SyntheticRegression": SyntheticRegression,
}


def build_dataset(cfg: Dict[str, Any]) -> Dataset:
    """按配置实例化数据集：cfg 形如 {"name": "...", **构造参数}。"""
    cfg = dict(cfg)
    name = cfg.pop("name")
    if name not in _DATASETS:
        raise KeyError(f"未知数据集 '{name}'，可选：{list(_DATASETS)}")
    return _DATASETS[name](**cfg)

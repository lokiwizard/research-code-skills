"""数据集包：按名字把配置变成数据集实例。

新增数据集：建文件 + 在下面 import + 往 `_DATASETS` 加一行。
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
from torch.utils.data import Dataset, Subset, random_split

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


def split_train_val(dataset: Dataset, val_ratio: float, seed: int) -> Tuple[Subset, Subset]:
    """按固定种子切出 (train, val)。

    train.py 和 eval.py 都必须走这一个函数，保证两边拿到完全相同的划分——
    否则评估会把训练样本混进"验证集"，指标虚高。
    """
    n_val = int(len(dataset) * val_ratio)
    g = torch.Generator().manual_seed(seed)
    return tuple(random_split(dataset, [len(dataset) - n_val, n_val], generator=g))

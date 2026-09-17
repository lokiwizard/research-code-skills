"""实验目录与日志：让每次跑都留下可追溯的痕迹。

一个实验目录长这样::

    experiments/mlp_2024-01-15_14-30/
    ├── config.yaml      # 当时生效的完整配置
    ├── train.log        # 控制台同款文本日志
    ├── metrics.csv      # 每个 epoch 一行，给画图/分析用
    └── checkpoints/     # 模型权重

目录命名 = 实验名 + 人类可读时间，一眼能看出是什么、什么时候跑的。
同一分钟内重名会自动加 -2、-3 后缀，不会互相覆盖。
"""

from __future__ import annotations

import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


def make_experiment_dir(root: str | Path, name: str) -> Path:
    """新建实验目录：<root>/<name>_<YYYY-MM-DD_HH-MM>，返回目录路径。

    同名同分钟重复创建时追加 -2、-3 后缀，避免覆盖。
    """
    if not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError("实验名必须是单个目录名")
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    for i in range(1, 100):
        dirname = f"{name}_{stamp}" if i == 1 else f"{name}_{stamp}-{i}"
        exp_dir = Path(root) / dirname
        try:
            exp_dir.mkdir(parents=True)
        except FileExistsError:
            continue
        return exp_dir
    raise RuntimeError(f"实验目录创建失败：{root}/{name}_{stamp} 下重名太多")


def setup_logger(log_file: str | Path, name: str = "exp") -> logging.Logger:
    """建一个同时往控制台和文件写的 logger。"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()  # 避免重复运行时句柄叠加导致日志重复
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    return logger


class CSVLogger:
    """逐行追加指标到 csv，列名按首次写入的字段确定。

    用法::

        csv_logger = CSVLogger(exp_dir / "metrics.csv")
        csv_logger.log({"epoch": 1, "train_loss": 0.5, "val_loss": 0.6})

    后续 `scripts/analyze.py` 直接读这个 csv 画曲线、出表格。
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fieldnames: list[str] | None = None

    def log(self, row: Dict[str, Any]) -> None:
        write_header = not self.path.exists()
        if self._fieldnames is None:
            self._fieldnames = list(row.keys())
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames)
            if write_header:
                writer.writeheader()
            # 只写已知列，缺的留空，多的忽略，保证表头稳定
            writer.writerow({k: row.get(k, "") for k in self._fieldnames})

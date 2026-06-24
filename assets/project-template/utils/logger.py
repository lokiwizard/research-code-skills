"""实验命名 + 日志记录：让每次跑都留下可追溯的痕迹。

一个实验目录长这样::

    experiments/baseline_20260624-153000_a1b2c3/
    ├── config.yaml      # 当时生效的完整配置
    ├── train.log        # 控制台同款文本日志
    ├── metrics.csv      # 每个 epoch 一行，给画图/分析用
    └── checkpoints/     # 模型权重

实验名规则：<实验名>_<时间戳>_<配置哈希>，三段分别回答
"这是哪个实验 / 什么时候跑的 / 用的什么配置"，天然避免覆盖和混淆。
"""

from __future__ import annotations

import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


def make_experiment_name(name: str, cfg_hash: str) -> str:
    """拼出唯一实验名：<name>_<时间戳>_<配置哈希>。"""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{name}_{stamp}_{cfg_hash}"


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

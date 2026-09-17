"""读取 YAML、应用命令行覆盖并保存有效配置。"""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any, Dict, List

import yaml


def load_config(path: str | Path) -> Dict[str, Any]:
    """读取 yaml 配置为普通 dict。"""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("配置根节点必须是映射")
    return cfg


def save_config(cfg: Dict[str, Any], path: str | Path) -> None:
    """保存有效配置值；不保留原 YAML 注释。"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)


def _coerce(value: str) -> Any:
    """把命令行传来的字符串尽量转成合适的类型（int/float/bool/None/原样）。"""
    low = value.lower()
    if low in {"true", "false"}:
        return low == "true"
    if low in {"null", "none"}:
        return None
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            continue
    return value


def apply_overrides(cfg: Dict[str, Any], overrides: List[str] | None) -> Dict[str, Any]:
    """应用形如 ['train.optimizer.lr=1e-4', 'model.depth=4'] 的覆盖，返回新字典。

    用点号定位嵌套键；中间不存在的层会自动建出来。原配置不会被改动。
    """
    cfg = copy.deepcopy(cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"覆盖项需写成 key.path=value，收到：{item}")
        key_path, raw = item.split("=", 1)
        node = cfg
        keys = key_path.split(".")
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = _coerce(raw)
    return cfg


def validate_config(cfg: Dict[str, Any]) -> None:
    """校验入口使用的字段；组件参数由各构造函数检查。"""
    sections = {"experiment", "model", "dataset", "loss", "train"}
    if set(cfg) != sections or any(not isinstance(cfg[k], dict) for k in sections):
        raise ValueError(f"配置必须包含且仅包含 {sorted(sections)} 映射")
    allowed = {
        "experiment": {"name", "seed", "output_root"},
        "train": {"epochs", "batch_size", "optimizer", "scheduler", "val_split",
                  "split_seed", "num_workers", "pin_memory", "device", "log_interval",
                  "ckpt_interval", "ckpt_keep", "eval_interval", "eval_keep",
                  "monitor_metric", "monitor_mode"},
    }
    for section, keys in allowed.items():
        extra = set(cfg[section]) - keys
        if extra:
            raise ValueError(f"未知配置项 {section}: {sorted(extra)}")
    t = cfg["train"]
    for key in ["epochs", "batch_size"]:
        if key not in t:
            raise ValueError(f"缺少 train.{key}")
    for key in ["epochs", "batch_size", "log_interval", "num_workers", "ckpt_interval",
                "ckpt_keep", "eval_interval", "eval_keep"]:
        if key in t:
            minimum = 1 if key in {"epochs", "batch_size", "log_interval"} else 0
            if type(t[key]) is not int or t[key] < minimum:
                raise ValueError(f"train.{key} 必须是 >= {minimum} 的整数")
    ratio = t.get("val_split", 0.2)
    if isinstance(ratio, bool) or not isinstance(ratio, (int, float)) or not 0 < ratio < 1:
        raise ValueError("train.val_split 必须位于 (0, 1)")
    for section, key in [("experiment", "seed"), ("train", "split_seed")]:
        if key in cfg[section] and (type(cfg[section][key]) is not int
                                  or not 0 <= cfg[section][key] < 2**32):
            raise ValueError(f"{section}.{key} 必须是 [0, 2**32) 内的整数")
    if t.get("monitor_mode", "min") not in {"min", "max"}:
        raise ValueError("train.monitor_mode 必须为 min 或 max")
    def finite(value):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("配置不能包含 NaN 或 Inf")
        if isinstance(value, dict):
            for item in value.values():
                finite(item)
        elif isinstance(value, list):
            for item in value:
                finite(item)
    finite(cfg)

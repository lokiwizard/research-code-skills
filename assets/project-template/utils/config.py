"""配置读写：yaml 为唯一事实来源（single source of truth）。

约定：
- 一切超参都进 yaml，代码里不写死数字。
- 支持命令行点号覆盖（`--set train.lr=1e-4`），方便快速试参而不改文件。
- 每次运行都把"最终生效的配置"存进实验目录，并算一个短哈希作为实验指纹，
  保证"看到结果 -> 找得到当时的配置"。
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

import yaml


def load_config(path: str | Path) -> Dict[str, Any]:
    """读取 yaml 配置为普通 dict。"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_config(cfg: Dict[str, Any], path: str | Path) -> None:
    """把配置原样写回 yaml（保留中文、不排序、可读优先）。"""
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
    """应用形如 ['train.lr=1e-4', 'model.depth=4'] 的覆盖，返回新字典。

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


def config_hash(cfg: Dict[str, Any], length: int = 6) -> str:
    """对配置做稳定哈希，作为实验指纹（同配置必得同哈希）。"""
    blob = json.dumps(cfg, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.md5(blob).hexdigest()[:length]

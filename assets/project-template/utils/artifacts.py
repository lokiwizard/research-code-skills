"""原子更新验证结果，并按 epoch 数值清理周期快照。"""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path


def prune_snapshots(directory: str | Path, keep: int, pattern: str) -> list[Path]:
    """保留最近 keep 个 epoch 档；0 表示显式保留全部。"""
    if keep < 0:
        raise ValueError("keep 不能为负数")
    if keep == 0:
        return []
    snapshots = []
    for path in Path(directory).glob(pattern):
        match = re.fullmatch(r"epoch_(\d+)\.[^.]+", path.name)
        if match and path.is_file() and not path.is_symlink():
            snapshots.append((int(match[1]), path))
    removed = [path for _, path in sorted(snapshots)[:-keep]]
    for path in removed:
        path.unlink()
    return removed


def write_json_atomic(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def save_evaluation(data: dict, directory: str | Path, improved: bool,
                    periodic: bool, keep: int) -> None:
    """更新 last/best 和周期验证快照；清理仅在新结果写入成功后执行。"""
    if keep < 0:
        raise ValueError("keep 不能为负数")
    directory = Path(directory)
    write_json_atomic(directory / "last.json", data)
    if improved:
        write_json_atomic(directory / "best.json", data)
    if periodic:
        write_json_atomic(directory / f"epoch_{data['epoch']:04d}.json", data)
        prune_snapshots(directory, keep, "epoch_*.json")

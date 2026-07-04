"""从一个基线配置批量生成消融/超参配置文件。

科研里两种常见需求，这里都支持：
- grid（网格）：所有取值的笛卡尔积，用于超参搜索。
- oat（one-at-a-time，逐一变量）：每次只改一个超参、其余保持基线，
  这是"消融实验"的标准做法，能干净地归因每个因素的影响。

例子：
    # 对 lr 和 depth 做网格
    python scripts/make_ablation.py --base configs/default.yaml \\
        --grid train.optimizer.lr=1e-3,5e-4 model.depth=2,4

    # 对 alpha 逐一消融（先把 loss 换成 CombinedLoss）
    python scripts/make_ablation.py --base configs/default.yaml --mode oat \\
        --set loss.name=CombinedLoss --grid loss.alpha=0.0,0.1,0.5,1.0

生成的配置写到 configs/ablations/，文件名编码了被改动的超参，便于后续 analyze.py
按超参取值聚合画图。只依赖标准库 + pyyaml，可独立复制使用。
"""

from __future__ import annotations

import argparse
import copy
import itertools
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml


def _coerce(v: str) -> Any:
    low = v.lower()
    if low in {"true", "false"}:
        return low == "true"
    if low in {"null", "none"}:
        return None
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            continue
    return v


def set_by_path(cfg: Dict[str, Any], dotted: str, value: Any) -> None:
    """按 'a.b.c' 路径写值，中间层不存在则建出来。"""
    node = cfg
    keys = dotted.split(".")
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value


def parse_grid(items: List[str]) -> List[Tuple[str, List[Any]]]:
    """把 ['train.optimizer.lr=1e-3,5e-4', 'model.depth=2,4'] 解析成 [(key, [值...]), ...]。"""
    grid = []
    for item in items:
        key, raw = item.split("=", 1)
        values = [_coerce(v.strip()) for v in raw.split(",")]
        grid.append((key, values))
    return grid


def _tag(key: str, value: Any) -> str:
    """给文件名/实验名用的短标签，如 'lr=0.0005' -> 'lr0.0005'。"""
    short = key.split(".")[-1]
    return f"{short}{value}".replace(" ", "")


def gen_grid(base: Dict[str, Any], grid: List[Tuple[str, List[Any]]],
             base_name: str) -> List[Tuple[str, Dict[str, Any]]]:
    """笛卡尔积：每个组合一个配置。"""
    keys = [k for k, _ in grid]
    out = []
    for combo in itertools.product(*[vals for _, vals in grid]):
        cfg = copy.deepcopy(base)
        tags = []
        for k, v in zip(keys, combo, strict=True):
            set_by_path(cfg, k, v)
            tags.append(_tag(k, v))
        name = base_name + "__" + "_".join(tags)
        cfg["experiment"]["name"] = name
        out.append((name, cfg))
    return out


def gen_oat(base: Dict[str, Any], grid: List[Tuple[str, List[Any]]],
            base_name: str) -> List[Tuple[str, Dict[str, Any]]]:
    """逐一变量：每次只改一个 key 的一个取值，其余保持基线。"""
    out = []
    for key, values in grid:
        for v in values:
            cfg = copy.deepcopy(base)
            set_by_path(cfg, key, v)
            name = f"{base_name}__{_tag(key, v)}"
            cfg["experiment"]["name"] = name
            out.append((name, cfg))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="批量生成消融/超参配置")
    parser.add_argument("--base", required=True, help="基线 yaml")
    parser.add_argument("--grid", nargs="+", required=True,
                        help="扫描项，如 train.optimizer.lr=1e-3,5e-4 model.depth=2,4")
    parser.add_argument("--mode", choices=["grid", "oat"], default="grid",
                        help="grid=笛卡尔积；oat=每次只改一个变量（消融）")
    parser.add_argument("--set", nargs="*", default=[], dest="fixed",
                        help="先施加到基线上的固定覆盖，如 loss.name=CombinedLoss")
    parser.add_argument("--out-dir", default="configs/ablations", help="输出目录")
    args = parser.parse_args()

    with open(args.base, "r", encoding="utf-8") as f:
        base = yaml.safe_load(f)
    for item in args.fixed:  # 先把固定覆盖应用到基线
        key, raw = item.split("=", 1)
        set_by_path(base, key, _coerce(raw))

    base_name = base.get("experiment", {}).get("name", "exp")
    grid = parse_grid(args.grid)
    gen = gen_grid if args.mode == "grid" else gen_oat
    configs = gen(base, grid, base_name)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"生成 {len(configs)} 个配置到 {out_dir}/ （模式：{args.mode}）\n")
    for name, cfg in configs:
        path = out_dir / f"{name}.yaml"
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
        print(f"  python train.py --config {path}")
    print("\n逐条运行上面的命令即可跑完整组消融；跑完用 scripts/analyze.py 汇总。")


if __name__ == "__main__":
    main()

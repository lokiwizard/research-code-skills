"""读取实验记录，输出汇总表、曲线和图注初稿。比较前需核对实验协议。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")  # 无界面环境也能出图
import matplotlib.pyplot as plt
import numpy as np
import yaml


# ----------------------------- 读取层 -----------------------------
def find_experiments(exp_root: Path) -> List[Path]:
    """凡是含 config.yaml 的子目录都视为一个实验。"""
    return sorted(p.parent for p in exp_root.glob("*/config.yaml"))


def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_csv(path: Path) -> List[Dict[str, float]]:
    """读 metrics.csv 为行列表，数值列自动转 float。"""
    if not path.exists():
        return []
    rows: List[Dict[str, float]] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in csv.DictReader(f):
            row: Dict[str, Any] = {}
            for k, v in raw.items():
                try:
                    row[k] = float(v)
                except (ValueError, TypeError):
                    row[k] = v
            rows.append(row)
    return rows


def get_nested(cfg: Dict[str, Any], dotted: str) -> Any:
    """按 'model.depth' 取嵌套值，取不到返回 None。"""
    node: Any = cfg
    for k in dotted.split("."):
        if not isinstance(node, dict) or k not in node:
            return None
        node = node[k]
    return node


class Experiment:
    """把一个实验目录的所有产物包起来，统一查询入口。"""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.name = path.name
        self.config = load_yaml(path / "config.yaml")
        self.rows = read_csv(path / "metrics.csv")
        self.summary = load_json(path / "metrics.json")
        self.eval = load_json(path / "eval.json")

    def hyper(self, dotted: str) -> Any:
        return get_nested(self.config, dotted)

    def metric(self, name: str) -> Optional[float]:
        """按优先级解析一个指标：summary -> eval -> metrics.csv 末行。"""
        if name in self.summary:
            return self.summary[name]
        if name in self.eval:
            return self.eval[name]
        if self.rows and name in self.rows[-1]:
            val = self.rows[-1][name]
            return val if isinstance(val, float) else None
        return None

    def curve(self, name: str):
        """返回 (epochs, values)，用于画 metric vs epoch。"""
        xs, ys = [], []
        for r in self.rows:
            if name in r and isinstance(r[name], float):
                xs.append(r.get("epoch", len(xs) + 1))
                ys.append(r[name])
        return xs, ys


# ----------------------------- 输出层 -----------------------------
def write_caption(out_path: Path, text: str) -> None:
    out_path.write_text(text.strip() + "\n", encoding="utf-8")
    print(f"  caption -> {out_path}")


def savefig(fig, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  figure  -> {out_path}")


# ----------------------------- 三种模式 -----------------------------
def cmd_summary(exps: List[Experiment], out_dir: Path, metrics: List[str]) -> None:
    """跨实验最终指标汇总表（Markdown + CSV）。"""
    # 若用户没指定指标，自动收集所有实验 summary/eval 里出现过的数值键
    if not metrics:
        keys = set()
        for e in exps:
            keys.update(k for k, v in {**e.summary, **e.eval}.items()
                        if isinstance(v, (int, float)))
        metrics = sorted(keys)

    header = ["experiment"] + metrics
    table = [header]
    for e in exps:
        row = [e.name]
        for m in metrics:
            v = e.metric(m)
            if isinstance(v, bool) or v is None:
                row.append("—")
            elif isinstance(v, float):
                row.append(f"{v:.4f}")
            elif isinstance(v, int):
                row.append(str(v))
            else:
                row.append("—")
        table.append(row)

    # 写 CSV
    csv_path = out_dir / "summary.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(table)
    print(f"  table   -> {csv_path}")

    # 写 Markdown 表 + caption
    md = ["| " + " | ".join(header) + " |",
          "| " + " | ".join("---" for _ in header) + " |"]
    md += ["| " + " | ".join(r) + " |" for r in table[1:]]
    caption = (
        "表：每行对应一次运行。指标依次从 metrics.json、eval.json 或 CSV 末行读取；"
        "数据划分、权重选择与统计口径需结合原始记录核对。"
    )
    (out_dir / "summary.md").write_text(
        "\n".join(md) + "\n\n" + caption + "\n", encoding="utf-8")
    print(f"  table   -> {out_dir / 'summary.md'}")


def cmd_curves(exps: List[Experiment], out_dir: Path, metric: str) -> None:
    """多实验叠加的 metric vs epoch 曲线。"""
    fig, ax = plt.subplots(figsize=(6, 4))
    plotted = 0
    for e in exps:
        xs, ys = e.curve(metric)
        if xs:
            ax.plot(xs, ys, marker="o", markersize=3, label=e.name)
            plotted += 1
    if plotted == 0:
        print(f"  [跳过] 没有实验包含指标列 '{metric}'")
        return
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} vs epoch")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    out_path = out_dir / f"curve_{metric}.png"
    savefig(fig, out_path)
    write_caption(
        out_dir / f"curve_{metric}.md",
        f"图：训练过程中 {metric} 随轮次的变化（共 {plotted} 条曲线，每条对应一个实验）。"
        f"横轴为 epoch，纵轴为 {metric}。不同曲线的预算和评估协议需另行核对。",
    )


def cmd_sweep(exps: List[Experiment], out_dir: Path, x: str, y: str,
              x2: Optional[str], mode: str = "min") -> None:
    """指标 vs 超参 的消融图：一维画曲线，二维画 3D 曲面。

    mode 决定 caption 里"最优"怎么标：min=指标越小越好（loss/MSE），
    max=越大越好（PSNR/Accuracy/R²）。
    """
    pick_best = np.argmin if mode == "min" else np.argmax
    points = []
    for e in exps:
        xv, yv = e.hyper(x), e.metric(y)
        x2v = e.hyper(x2) if x2 else None
        if xv is None or yv is None or (x2 and x2v is None):
            continue
        points.append((xv, x2v, yv))
    if not points:
        print(f"  [跳过] 没有实验同时具备超参 '{x}' 和指标 '{y}'")
        return

    if x2 is None:
        # 一维：metric vs 单个超参
        points.sort(key=lambda p: p[0])
        xs = [p[0] for p in points]
        ys = [p[2] for p in points]
        fig, ax = plt.subplots(figsize=(6, 4))
        if len(set(xs)) == len(xs):
            ax.plot(xs, ys, marker="o")
        else:
            ax.scatter(xs, ys)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title(f"{y} vs {x}")
        ax.grid(True, alpha=0.3)
        savefig(fig, out_dir / f"sweep_{y}_vs_{x.replace('.', '-')}.png")
        best_i = int(pick_best(ys))
        write_caption(
            out_dir / f"sweep_{y}_vs_{x.replace('.', '-')}.md",
            f"图：{y} 随超参 {x} 的变化（消融）。在 {x}={xs[best_i]} 时取得"
            f"本组记录中的最优 {y}={ys[best_i]:.4f}；未进行重复实验统计。",
        )
    else:
        # 二维：metric 关于两个超参的 3D 曲面（如 α 与 L 的消融）
        xs = np.array([p[0] for p in points], dtype=float)
        x2s = np.array([p[1] for p in points], dtype=float)
        ys = np.array([p[2] for p in points], dtype=float)
        fig = plt.figure(figsize=(7, 5))
        ax = fig.add_subplot(111, projection="3d")
        coordinates = np.column_stack((xs, x2s))
        if (len(np.unique(coordinates, axis=0)) == len(xs)
                and len(xs) >= 3
                and np.linalg.matrix_rank(coordinates - coordinates[0]) == 2):
            ax.plot_trisurf(xs, x2s, ys, cmap="viridis", edgecolor="none", alpha=0.9)
        ax.scatter(xs, x2s, ys, color="k", s=15)
        ax.set_xlabel(x)
        ax.set_ylabel(x2)
        ax.set_zlabel(y)
        ax.set_title(f"{y} over ({x}, {x2})")
        out_path = out_dir / f"surface_{y}_{x.replace('.', '-')}_{x2.replace('.', '-')}.png"
        savefig(fig, out_path)
        best_i = int(pick_best(ys))
        write_caption(
            out_path.with_suffix(".md"),
            f"图：{y} 关于 {x} 与 {x2} 的二维消融曲面。最优点出现在 "
            f"{x}={xs[best_i]:g}、{x2}={x2s[best_i]:g} 处（{y}={ys[best_i]:.4f}）。"
            f"散点为观测值，曲面（若存在）为插值；该图不独自证明交互作用。",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="实验结果分析与可视化")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # 公共参数挂到每个子命令上，这样 `--exp-root` 写在子命令前后都能用
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--exp-root", default="experiments", help="实验根目录")
    common.add_argument("--out-dir", default="results", help="图表输出目录")

    p_sum = sub.add_parser("summary", parents=[common], help="跨实验最终指标汇总表")
    p_sum.add_argument("--metrics", nargs="*", default=[], help="要汇总的指标键")

    p_cur = sub.add_parser("curves", parents=[common], help="metric vs epoch 曲线（多实验叠加）")
    p_cur.add_argument("--metric", default="val_loss")

    p_sw = sub.add_parser("sweep", parents=[common], help="metric vs 超参（消融）；给 --x2 则画 3D 曲面")
    p_sw.add_argument("--x", required=True, help="超参点号路径，如 loss.alpha")
    p_sw.add_argument("--x2", default=None, help="第二个超参，给了就画 3D 曲面")
    p_sw.add_argument("--y", required=True, help="指标键，如 best_val_mse")
    p_sw.add_argument("--mode", choices=["min", "max"], default="min",
                      help="标注最优点的方向：min=loss/MSE 类，max=PSNR/Accuracy 类")

    args = parser.parse_args()
    exp_root = Path(args.exp_root)
    out_dir = Path(args.out_dir)
    exps = [Experiment(p) for p in find_experiments(exp_root)]
    if not exps:
        print(f"在 {exp_root}/ 下没找到实验（需含 config.yaml）。先跑 train.py。")
        return
    print(f"找到 {len(exps)} 个实验，输出到 {out_dir}/")

    if args.cmd == "summary":
        cmd_summary(exps, out_dir, args.metrics)
    elif args.cmd == "curves":
        cmd_curves(exps, out_dir, args.metric)
    elif args.cmd == "sweep":
        cmd_sweep(exps, out_dir, args.x, args.y, args.x2, args.mode)


if __name__ == "__main__":
    main()

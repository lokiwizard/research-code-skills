"""评估入口：加载某个实验的 best.pt，在验证集上算指标。

    python eval.py --exp-dir experiments/baseline_20260624-153000_a1b2c3

之所以从实验目录读取，是因为目录里已经存着 config.yaml（怎么搭模型/数据）
和 checkpoints/best.pt（权重），评估完全可复现，不依赖你记得当时的命令行。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from datasets import build_dataset
from models import build_model
from trainers.trainer import resolve_device
from utils import METRICS, load_checkpoint, load_config, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="评估入口")
    parser.add_argument("--exp-dir", required=True, help="训练产出的实验目录")
    parser.add_argument("--ckpt", default="best.pt", help="checkpoints/ 下的权重文件名")
    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)
    cfg = load_config(exp_dir / "config.yaml")
    set_seed(int(cfg["experiment"]["seed"]))
    device = resolve_device(cfg["train"].get("device", "auto"))

    # 用与训练相同的配置重建模型与数据，再灌入权重
    model = build_model(cfg["model"]).to(device)
    state = load_checkpoint(exp_dir / "checkpoints" / args.ckpt, map_location=str(device))
    model.load_state_dict(state["model"])
    model.eval()

    dataset = build_dataset(cfg["dataset"])
    loader = DataLoader(dataset, batch_size=int(cfg["train"]["batch_size"]))

    preds, targets = [], []
    with torch.no_grad():
        for x, y in loader:
            preds.append(model(x.to(device)).cpu())
            targets.append(y)
    pred_all, target_all = torch.cat(preds), torch.cat(targets)

    results = {name: fn(pred_all, target_all) for name, fn in METRICS.items()}
    print(json.dumps(results, ensure_ascii=False, indent=2))
    with open(exp_dir / "eval.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

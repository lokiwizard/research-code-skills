"""评估入口：加载某个实验的 best.pt，在验证集上算指标。

    python eval.py --exp-dir experiments/baseline_20260624-153000_a1b2c3
    python eval.py --exp-dir ... --split all      # 需要时也可评训练集/全集

划分用与 train.py 同一个 split_train_val（同种子），保证评的验证集就是训练时
看到的那一份——否则训练样本会混进来，指标虚高。之所以从实验目录读取，是因为
目录里已经存着 config.yaml（怎么搭模型/数据）和 checkpoints/best.pt（权重），
评估完全可复现，不依赖你记得当时的命令行。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from datasets import build_dataset, split_train_val
from models import build_model
from trainers.trainer import resolve_device
from utils import METRICS, load_checkpoint, load_config, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="评估入口")
    parser.add_argument("--exp-dir", required=True, help="训练产出的实验目录")
    parser.add_argument("--ckpt", default="best.pt", help="checkpoints/ 下的权重文件名")
    parser.add_argument("--split", choices=["val", "train", "all"], default="val",
                        help="在哪部分数据上评估；默认 val=训练时的验证集划分")
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

    # 复现训练时的 train/val 划分（同一函数 + 同一种子），按 --split 取对应子集
    dataset = build_dataset(cfg["dataset"])
    if args.split != "all":
        train_set, val_set = split_train_val(
            dataset, float(cfg["train"].get("val_split", 0.2)),
            int(cfg["experiment"]["seed"]))
        dataset = val_set if args.split == "val" else train_set
    loader = DataLoader(dataset, batch_size=int(cfg["train"]["batch_size"]))

    preds, targets = [], []
    with torch.no_grad():
        for x, y in loader:
            preds.append(model(x.to(device)).cpu())
            targets.append(y)
    pred_all, target_all = torch.cat(preds), torch.cat(targets)

    # split 一并写进 eval.json，回看时知道这组数字是在哪部分数据上算的
    results = {"split": args.split,
               **{name: fn(pred_all, target_all) for name, fn in METRICS.items()}}
    print(json.dumps(results, ensure_ascii=False, indent=2))
    with open(exp_dir / "eval.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

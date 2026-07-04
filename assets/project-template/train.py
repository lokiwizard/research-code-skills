"""训练入口：`python train.py --config configs/default.yaml`。

这个文件只做一件事——**装配**：读配置、建实验目录、固定种子、按名字
搭出 模型/数据/损失，然后交给 Trainer 跑。所有"怎么训"的细节在 trainers/，
所有"训什么"的细节在 models/ datasets/ losses/，这里保持薄而清晰。

常用：
    python train.py --config configs/default.yaml
    python train.py --config configs/default.yaml --set train.optimizer.lr=5e-4 model.depth=4
    python train.py --resume experiments/baseline_20260624-153000_a1b2c3   # 断点续训
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from torch.utils.data import DataLoader

from datasets import build_dataset, split_train_val
from losses import build_loss
from models import build_model
from trainers import Trainer
from utils import (
    apply_overrides,
    config_hash,
    load_config,
    make_experiment_name,
    save_config,
    set_seed,
    setup_logger,
)


def build_dataloaders(cfg, seed):
    """建数据集并按 val_split 切出训练/验证两个 DataLoader。

    切分走 datasets.split_train_val（固定种子），eval.py 用同一函数，
    保证训练和评估看到完全相同的划分。
    """
    tcfg = cfg["train"]
    dataset = build_dataset(cfg["dataset"])
    train_set, val_set = split_train_val(dataset, float(tcfg.get("val_split", 0.2)), seed)

    loader_kwargs = dict(
        batch_size=int(tcfg["batch_size"]),
        num_workers=int(tcfg.get("num_workers", 0)),   # 重预处理任务（图像等）调大
        pin_memory=bool(tcfg.get("pin_memory", False)),  # cuda 下开了能加速 H2D 拷贝
    )
    train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)
    return train_loader, val_loader


def prepare_experiment(args):
    """根据是否续训，准备好 (配置, 实验目录, 是否续训)。

    - 新实验：读配置 + 命令行覆盖，新建带时间戳的实验目录并存下配置。
    - 续训：复用已有实验目录与其中的 config.yaml（允许 --set 微调，如增大 epochs）。
    """
    if args.resume:
        exp_dir = Path(args.resume)
        if not (exp_dir / "config.yaml").exists():
            raise SystemExit(f"续训目录里没有 config.yaml：{exp_dir}")
        cfg = apply_overrides(load_config(exp_dir / "config.yaml"), args.overrides)
        if args.overrides:
            # 覆盖项也要落盘，config.yaml 必须始终等于"实际生效的配置"（可追溯原则）
            save_config(cfg, exp_dir / "config.yaml")
        return cfg, exp_dir, True

    if not args.config:
        raise SystemExit("请提供 --config（新实验）或 --resume（续训）")
    cfg = apply_overrides(load_config(args.config), args.overrides)
    exp_name = make_experiment_name(cfg["experiment"]["name"], config_hash(cfg))
    exp_dir = Path(cfg["experiment"].get("output_root", "experiments")) / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, exp_dir / "config.yaml")  # 存下当时生效的完整配置
    return cfg, exp_dir, False


def main() -> None:
    parser = argparse.ArgumentParser(description="训练入口")
    parser.add_argument("--config", help="yaml 配置路径（新实验必填）")
    parser.add_argument("--resume", default=None,
                        help="断点续训：传入已有实验目录，从 last.pt 接着训")
    parser.add_argument("--set", nargs="*", default=[], dest="overrides",
                        help="命令行覆盖，如 train.optimizer.lr=1e-4 model.depth=4")
    args = parser.parse_args()

    # 1) 配置与实验目录（区分新实验 / 续训）
    cfg, exp_dir, resuming = prepare_experiment(args)
    logger = setup_logger(exp_dir / "train.log")
    logger.info(f"{'续训' if resuming else '新实验'}：{exp_dir}")

    # 2) 复现：固定种子
    set_seed(int(cfg["experiment"]["seed"]))

    # 3) 按名字装配组件（这里完全不出现具体类名，新增组件无需改本文件）
    train_loader, val_loader = build_dataloaders(cfg, int(cfg["experiment"]["seed"]))
    model = build_model(cfg["model"])
    loss_fn = build_loss(cfg["loss"])
    logger.info(f"模型：{cfg['model']['name']} | 损失：{cfg['loss']['name']}")

    # 4) 训练（续训时先恢复进度）
    trainer = Trainer(cfg, model, train_loader, val_loader, loss_fn, exp_dir, logger)
    if resuming:
        trainer.resume_from(exp_dir / "checkpoints" / "last.pt")
    summary = trainer.fit()

    # 5) 落盘最终摘要，供 scripts/analyze.py 汇总
    with open(exp_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

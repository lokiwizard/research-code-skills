"""科研项目脚手架生成器：把 assets/project-template/ 复制成一个新项目。

用法：
    python new_project.py my-experiment                 # 在当前目录生成 my-experiment/
    python new_project.py my-exp --dest ~/research       # 指定父目录
    python new_project.py my-exp --git --uv              # 顺便 git init 并 uv sync

做的事很简单（刻意保持透明）：
1. 递归复制模板目录到 <dest>/<name>/；
2. 把文件里的占位符 {{PROJECT_NAME}} 替换成项目名；
3. 可选：初始化 git、用 uv 建环境。

只用标准库，复制后的项目是完全独立、可直接运行的真实代码。
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

# 模板目录相对本脚本定位：scripts/ 的上一级是技能根，再进 assets/project-template
TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "assets" / "project-template"

# 复制时跳过的脏目录（万一模板被跑过留下缓存）
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".venv", "uv.lock",
                                ".ruff_cache", ".pytest_cache", ".DS_Store",
                                "experiments", "results")

# 需要做占位符替换的文本文件
SUBSTITUTE_FILES = ["pyproject.toml", "README.md"]


def render_placeholders(project_dir: Path, project_name: str) -> None:
    """把 {{PROJECT_NAME}} 替换成真实项目名。"""
    for rel in SUBSTITUTE_FILES:
        path = project_dir / rel
        if path.exists():
            text = path.read_text(encoding="utf-8")
            path.write_text(text.replace("{{PROJECT_NAME}}", project_name), encoding="utf-8")


def restore_runtime_dirs(project_dir: Path) -> None:
    """IGNORE 跳过了 experiments/ results/，这里补回空目录 + .gitkeep。"""
    for d in ["experiments", "results", "configs/ablations"]:
        target = project_dir / d
        target.mkdir(parents=True, exist_ok=True)
        (target / ".gitkeep").touch()


def main() -> None:
    parser = argparse.ArgumentParser(description="生成科研项目脚手架")
    parser.add_argument("name", help="项目名（会作为目录名与包名）")
    parser.add_argument("--dest", default=".", help="父目录，默认当前目录")
    parser.add_argument("--git", action="store_true", help="生成后执行 git init")
    parser.add_argument("--uv", action="store_true", help="生成后执行 uv sync 建环境")
    parser.add_argument("--force", action="store_true", help="目标已存在时覆盖")
    args = parser.parse_args()

    if not TEMPLATE_DIR.exists():
        sys.exit(f"找不到模板目录：{TEMPLATE_DIR}")

    project_dir = Path(args.dest).expanduser().resolve() / args.name
    if project_dir.exists():
        if not args.force:
            sys.exit(f"目标已存在：{project_dir}（加 --force 覆盖）")
        shutil.rmtree(project_dir)

    shutil.copytree(TEMPLATE_DIR, project_dir, ignore=IGNORE)
    render_placeholders(project_dir, args.name)
    restore_runtime_dirs(project_dir)
    print(f"已生成项目：{project_dir}")

    if args.git:
        subprocess.run(["git", "init", "-q"], cwd=project_dir, check=False)
        print("  已 git init")
    if args.uv:
        print("  正在 uv sync ……")
        subprocess.run(["uv", "sync"], cwd=project_dir, check=False)

    print("\n下一步：")
    print(f"  cd {project_dir}")
    print("  uv sync  # 若未加 --uv" if not args.uv else "  uv run python train.py --config configs/default.yaml")
    if not args.uv:
        print("  uv run python train.py --config configs/default.yaml")


if __name__ == "__main__":
    main()

# research-code-skills

面向科研代码的 Claude Code / Codex 技能：论文方法实现、实验设计、PyTorch 脚手架、配置与消融、结果分析和复现交付。

主说明见 [SKILL.md](SKILL.md)，实验要求见 [实验设计与验收](references/experiment-protocol.md)。已有项目保留原结构；新项目可使用附带的合成回归模板。

## 安装

```bash
./install.sh                 # 软链接到 Claude Code 与 Codex
./install.sh claude          # 仅 Claude Code
./install.sh codex           # 仅 Codex
./install.sh both --copy     # 复制
./install.sh both --dry-run  # 查看目标路径
```

## 直接使用模板

```bash
python scripts/new_project.py my-experiment --dest ~/research --git --uv
cd ~/research/my-experiment
uv run python train.py --config configs/default.yaml
```

模板提供训练、验证、checkpoint、续训和基础作图；没有独立测试集、多种子统计或分布式训练。正式研究需按任务补充。命令示例见 [walkthrough](examples/walkthrough.md)。

## 验证技能

在已具备模板依赖的 Python 环境运行 `python -m unittest discover -s tests -v`。测试在临时目录生成项目，覆盖配置拒绝、文件保留、恢复训练及结果分析；退出后清理测试产物。

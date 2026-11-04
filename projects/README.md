# 项目管理

本仓库采用独立 Git 工作区隔离任务。每个工作区有自己的源码文件和分支，
共享 Git 对象库，不共享正在编辑的文件。改楼梯工作区的 `supermimic/world_model`
也不会改变键盘工作区的同名文件。分支隔离不是硬件安全或策略成功证明。

## 工作区清单

| 工作区标识 | 职责 | 本机位置 | 分支 |
| --- | --- | --- | --- |
| `keyboard` | 现有参考动作键盘控制、状态估计、部署准备 | 当前目录 | `main` |
| `keyboard-training` | 现有速度接口、五动作训练与停止修复 | `~/.codex/worktrees/x2-velocity-control/supermimic` | `x2-velocity-control` |
| `stairs` | 不同尺寸楼梯、仅双脚、ACP 式规划与世界模型训练 | `../supermimic-stairs` | `x2-stairs` |

已有键盘训练工作区保持原位。其 `locomotion/` 实现与主目录不同；主目录中的部分
历史训练脚本通过 `--source` 指向该版本。不能把这些脚本当作主目录上已接通的训练入口。
其他 IMU、no-anchor 等历史工作区保留，未更名、清理或合并。

机器可读清单：[workspaces.json](workspaces.json)。`baseline` 只标识本次分区时的
源码起点，**不是经验证的最佳策略版本**。后续工作区 HEAD 可以前进，已发布源码
包和已验证模型的固定提交不能随之自动变化。本机路径变化时显式更新清单。

## 新代码归属

- 键盘：在现有对应工作区维护。现有入口和模块路径保持兼容，避免移动文件破坏
  服务器复制包、历史检查点和脚本间导入。
- 楼梯：新任务代码放 `src/supermimic_stairs/`，命令放 `scripts/stairs/`，
  配置放 `configs/stairs/`，测试放 `tests/stairs/`，记录放 `projects/stairs/`。
  不从键盘启动脚本或运行时替换训练类的补丁安装器借用任务逻辑。
- 公共算法：各工作区保留自己固定版本的 `src/supermimic/`。楼梯通过组合或明确
  接口使用它；必要修改只在楼梯工作区完成。确认通用、测试通过后，逐项移植到
  需要它的项目。禁止默认合并楼梯分支回键盘分支，也不复制整套旧实验脚本作为新入口。
- 不在公共配置上增加不断扩大的 `if keyboard / if stairs` 开关树。
  任务观测、目标、奖励、终止、replay 字段和评测由各任务负责，公共模块保留稳定契约。

## 日常检查与发布

从当前管理目录运行，工具只用 Python 标准库，不启动训练或电机程序：

```bash
python3 scripts/project.py status
python3 scripts/project.py check stairs
python3 scripts/project.py export stairs --out outputs/releases/stairs-具体版本
```

`check` 验证工作区根、分支、源码基线、未提交改动和跨目录源码链接。
`export` 只导出干净工作区的固定提交，生成 `source.tar` 和 `release.json`，
记录项目、提交、源码 SHA256 与依赖锁 SHA256；已有目标目录一律拒绝覆盖。
它不运行测试，不证明依赖环境正确，也不代表模型通过验收。

服务器后续采用以下布局；**此次只建立本机隔离，未迁移服务器或修改机器人**：

```text
数据根/projects/keyboard/{releases,runs,datasets,checkpoints,cache}/
数据根/projects/stairs/{releases,runs,datasets,checkpoints,cache}/
```

发布源码解压到各自 `releases/<commit>/source/`。程序必须从该源码导入，并记录
实际模块路径；不能继续指向另一项目的 editable 安装。保留每次运行的
`release.json`、实际环境版本、数据/场景/初始权重 SHA256、完整命令、GPU、PID、日志
和验收结果。原始资产可复用固定清单和哈希对应的版本，项目转换结果和缓存分开写。

键盘现有环境不变。楼梯使用规划中的独立环境
`/home/2025-Long.xh/HHD/venvs/supermimic-stairs`；当前没有安装或修改此环境。
不能给两个项目链接同一个可变环境，也不能从另一个项目的训练运行中直接恢复优化器。

## 变更与回归

1. 先确认工作区，修改前保存 Git 回档点。
2. 在项目内完成实现与针对性验证，提交准确的已验证/待验证状态。
3. 通用修复按具体提交移植；接收项目重新检查数据、权重接口与关键行为。
4. 键盘至少覆盖命令超时、观测布局、动作映射和对应仿真行为；楼梯覆盖场景与
   接触一致性、非足部支撑判定、完整登顶停止和保留尺寸测试。
5. 重新发布固定源码版本，再决定部署。旧源码包、旧权重和旧日志不覆盖。

管理工具的隔离回归可在本机运行：`python3 tests/test_project_management.py`。
训练/策略的回归仍按 AGENTS.md 在服务器执行。

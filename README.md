# SuperMimic：X2 Ultra 仿真复现

当前目标是在 Ubuntu、4×RTX 3090 上复现 X2 Ultra 的动作跟踪仿真。LAFAN1 已跑通数据、短训、首轮训练与完整片段评测，后续续训继续进行；现已接入 Webster 视频动作的数据准备与有界短训，首轮正式训练与评测已完成，完整动作评测失败。当前范围不包含实机部署。

参考实现：[tianshuwu/supermimic-retargeted](https://github.com/tianshuwu/supermimic-retargeted)。机器人为 X2 Ultra 旗舰版；[智元官方模型](https://github.com/AgibotTech/agibot_x2_urdf)对应 `X2_URDF-v1.3.0/`。复现时先使用参考实现固定的资产版本，官方模型用于核对，不直接替换训练模型。

## 当前状态

- 已引入参考实现 `ff69a0e5305126eb5b80b323bcd815805bc3df4e`，保留本项目规范；原说明见 [上游 README](docs/upstream/README.md)。
- 已连接服务器并核验 Ubuntu 22.04.5、四张 RTX 3090、驱动 580.178.04。主训练与 SOMA 环境已核验；40 段 LAFAN1 已完成准备和首轮完整评测，后续续训继续进行。GEM-X 使用独立环境，Webster 参考数据已通过碰撞/接触检查与100步短训；这不代表策略已学会该动作。详见[本项目实验记录](docs/LOCAL_EXPERIMENT.md)。
- 以下主机与目录布局已连接核验；GPU 占用和磁盘余量以每次启动前的检查为准。
- 参考项目的安装与运行入口见其 [README](https://github.com/tianshuwu/supermimic-retargeted/blob/main/README.md)、[数据文档](https://github.com/tianshuwu/supermimic-retargeted/blob/main/docs/DATA.md)和[训练文档](https://github.com/tianshuwu/supermimic-retargeted/blob/main/docs/USAGE.md)。引入源码后，以固定版本的实际接口为准。

## 运行位置

本地仅做源码与文档编辑、静态检查、需要境外网络的下载，以及文件传输。仿真、动作重定向、轨迹采集、训练、策略评测和仿真录像在服务器执行。

| 用途 | 主机 / 路径 | 状态 |
| --- | --- | --- |
| SSH 主机 | `2025-Long-robot` | 已连接核验 |
| 本地项目 | `/Users/orangelong/Project/supermimic` | 已存在 |
| 服务器源码 | `/home/2025-Long.xh/SSD/supermimic` | 已上传固定源码 |
| 数据、资产、缓存与运行产物 | `/home/2025-Long.xh/HHD/data/supermimic` | 已创建 |
| 主训练环境 | `/home/2025-Long.xh/HHD/venvs/supermimic` | 已安装并核验 |
| SOMA 重定向环境 | `/home/2025-Long.xh/HHD/venvs/supermimic-soma` | 已安装并核验 |
| GEM-X 视频恢复环境 | `/home/2025-Long.xh/HHD/venvs/supermimic-gem` | 已安装，CPU恢复已运行 |

`HHD` 本身可能是符号链接，比较路径和检查磁盘时使用 `readlink -f`、`df -h` 核对真实位置，不根据目录名称推断介质或剩余容量。

## 网络与下载

- 服务器按“可访问国内站点、不可访问境外站点”的规则使用，不依赖训练时临时访问境外服务。
- 国内源或镜像可获得的依赖与资源，直接在服务器下载；Python 通用依赖优先使用清华源：`https://pypi.tuna.tsinghua.edu.cn/simple`。
- 境外源码、Git LFS 网格、权重、数据与依赖先在本地下载，再上传服务器。安装脚本及首次运行触发的自动下载也遵守此规则。
- 本地准备的安装包必须匹配服务器 Linux、CPU 架构、Python 和 CUDA 组合，并包含所需传递依赖；不能把 macOS 安装包直接传到 Ubuntu 使用。
- PyTorch CUDA 包、Git 固定版本和专用包源以项目锁文件为准，不为使用镜像而改成不匹配的版本。记录资源来源、版本或提交号；重要压缩包传输后校验 SHA256。

## 环境规范

1. 主训练与 SOMA 使用独立环境，不复用 G-WAM 或其他项目环境，不向系统 Python 或全局环境安装项目依赖。
2. 服务器项目根目录的 `.venv` 链接到主训练环境；引入 SOMA 后，其 `vendor/soma-retargeter/.venv` 链接到 SOMA 专用环境。创建链接前检查已有目标，不覆盖现存环境。
3. 安装前核对锁文件、Python 版本、GPU 驱动和 CUDA 包兼容性。参考实现要求 Python 3.12，具体依赖以引入的固定版本为准，不沿用 G-WAM 的旧环境版本。
4. 使用明确的环境解释器或已核对的项目启动脚本；运行前检查解释器真实路径、Python、PyTorch、CUDA 可用性与设备信息。
5. 若系统 `PYTHONPATH` 引入 ROS 或其他项目包，先诊断其影响；必要时仅对当前命令清除，避免修改全局配置。
6. 不擅自升级 NVIDIA 驱动、系统 CUDA 或修改其他用户的配置。需要系统级变更时先说明依据和影响。

## 数据、缓存与产物

大文件集中保存在专用数据目录，源码目录仅保留必要入口。引入参考代码后，按实际接口建立符号链接或配置路径：

| 源码目录入口 | 对应数据目录下的规划位置 |
| --- | --- |
| `data/` | `data/` |
| `assets/` | `assets/` |
| `logs/` | `logs/` |
| `videos/` | `videos/` |

两列分别相对于服务器源码根目录和数据根目录。缓存放在数据根目录的 `cache/`；可按需设置 `XDG_CACHE_HOME`、`HF_HOME`、`UV_CACHE_DIR` 和 `TORCH_HOME`，保留与已有离线缓存的兼容性。

原始数据、转换中间文件、训练数据和检查点分别保存。每次运行使用唯一名称，保留配置、源码版本、环境信息、数据清单、日志和检查点。重跑不覆盖旧结果；训练中断保留故障日志。大数据、环境、网格、权重和缓存不提交到源码仓库。

## GPU 与任务规范

- 四张 3090 不代表当前均可用。启动前检查 GPU 显存、利用率、进程所有者和已有任务；只使用已确认空闲或已分配的卡。
- 显式指定 GPU；多卡训练按每卡容量规划。数据并行不会自动把四张卡合并为一块大显存设备。
- 先单卡短训，再测小规模完整更新的显存峰值，最后扩大环境数、批量和卡数。此前讨论的参数是试跑建议，尚无本项目显存实测。
- 长任务使用服务器已有的任务管理方式，或使用 `tmux` 等保留会话与日志；记录会话、PID、运行目录、GPU 和启动命令。
- 只停止本项目中可确认归属的任务；禁止按宽泛进程名批量结束任务或干扰其他用户。
- 短训通过只说明流程可执行，正式结果需要完整动作评测。参考项目结果不作为本项目已获得的结果。

## 下一步

1. 继续核验既有 LAFAN1 续训与完整片段评测。
2. Webster GPU0单卡8192环境的6小时续训已完成，最终62450步；完整片段评测84/84步通过，较此前短训35/84步改善。
3. 连续策略录像可见翻转与落地，但片段结束后失稳，2.7秒触发root_height；尚未达到落地后持续站稳，详见实验记录。

代理执行约束见 [AGENTS.md](AGENTS.md)。具体已执行阶段与结果以[本项目实验记录](docs/LOCAL_EXPERIMENT.md)和服务器日志为准；上游历史记录不代表本项目结果。

## 键盘行走实验分支

`codex/x2-velocity-control` 增加了共享解码器的速度编码器、动作跟踪/速度联合训练、浏览器键盘输入与连续启停评测。默认关闭；原训练入口通过 `--velocity.enabled` 启用。已完成一小时试训和 60 秒评测，8/8 未跌倒，但最终策略前进不足且存在偏航，尚未达到键盘行走目标。实测与运行位置见[速度控制记录](docs/VELOCITY_CONTROL_20261028.md)。

已修复参考模式失效后无法恢复的采样缺陷；随后定位并补齐停止行为监督，17 项接口测试通过。最新策略在三个评测种子的 60 秒启停中 24/24 未跌倒，停止平均平面速度降至 0.041–0.047 m/s，保留前进与转向。高速停止仍有减速瞬态和轻微晃动；长视野模型误差及长期续训保持尚未解决。当前权重、根因对照和录像见[停止修复记录](docs/VELOCITY_STOP_DIAGNOSIS_20261028.md)，此前阶段见[采样修复与 A/B](docs/VELOCITY_REPAIR_20261028.md)。

在服务器的独立源码快照中，使用明确解释器和当前源码路径。例如先验证配置：

```bash
PYTHONPATH="$PWD/src" CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/training/train.py \
  --robot x2_ultra --preset smoke --velocity.enabled --env.anchor-obs \
  --data.clips walk1_subject1 walk2_subject1 --print-config
```

训练权重兼容时可用 `--initialize-from <跟踪检查点>` 初始化；`--resume <速度检查点>` 恢复速度训练。选择新的运行名称和输出目录，启动前检查 GPU 与任务所有者。

速度检查点的连续评测与交互入口（均在服务器运行）：

```bash
PYTHONPATH="$PWD/src" CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/robot.py x2_ultra \
  scripts/evaluation/eval_velocity.py --ckpt <速度检查点> --out <新评测文件.json>
PYTHONPATH="$PWD/src" CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/robot.py x2_ultra \
  scripts/evaluation/play_velocity.py --ckpt <速度检查点>
```

交互时通过 SSH 转发查看器端口（通常为 8080，以输出为准）和键盘端口 8081。在键盘页面按住 W 前进、A/D 转向，松键、空格或失焦归零。连续评测会保存每次首次失败、重置次数和实际速度，未达到要求的试验不会因重置而被计为成功。

# SuperMimic：X2 Ultra 仿真与部署准备

本项目基于 SuperMimic，面向 X2 Ultra 旗舰版开展动作重定向、世界模型训练、仿真评测和实机部署准备。参考实现及固定版本见 [UPSTREAM.md](UPSTREAM.md)；机器人资产与上游说明见 [上游 README](docs/upstream/README.md)。

## 项目状态

- 用户反馈 2026-11-03 的 11500 实机部署成功，策略接管与五类键盘动作请求已有运行日志；该次同时出现状态租约故障和异常大的策略目标，尚未通过连续控制或安全验收。见 [实机部署记录](docs/X2_HARDWARE_DEPLOYMENT_20261103.md)。
- LAFAN1 数据准备、训练和完整片段评测已经运行；具体配置与结果见 [实验记录](docs/LOCAL_EXPERIMENT.md)。
- Webster 视频动作和其他动作已进行参考数据处理、训练及评测。各实验的结果和限制见 `docs/` 下对应记录；参考动作通过检查不等于策略完成动作。
- X2 当前采用 **500 Hz IMU 积分与状态估计路线**；点云和 FAST-LIO 定位路线已放弃，其文档保留为历史记录。实现、测量及四元数顺序勘误见 [IMU 状态估计记录](docs/X2_SHADOW_CONTACT_FUSION_20261102.md)。现有结果主要为静止只读验证，运动精度和完整状态链尚未验收。
- X2 11500 策略的实机准备、只读观测和仿真接管测试见 [部署准备记录](docs/X2_READINESS_OFFLINE_20261102.md)。记录中的接管动力学测试未通过，不能据此认定实机部署已通过验收。

实验记录反映当时的测量和进度；运行状态、GPU 占用及设备状态应在使用前重新核验。

## 文档入口

- [原始 IMU 估计器 PC2 容器接入与检查指令（2026-11-04）](docs/X2_IMU_HARDWARE_STAGING_20261104.md)：静止校准、零偏扣除、传感器积分；已做只读检查，实机运动精度未验收。

- [使用与训练](docs/USAGE.md) · [数据](docs/DATA.md) · [架构](docs/ARCHITECTURE.md)
- [本项目实验记录](docs/LOCAL_EXPERIMENT.md) · [历史仿真实验](docs/EXPERIMENTS.md)
- [X2 部署基础与当前路线](docs/X2_DEPLOYMENT.md) · [500 Hz IMU 状态估计](docs/X2_SHADOW_CONTACT_FUSION_20261102.md) · [PC2 容器准备](docs/X2_LXH_PC2_PREPARATION_20261031.md)
- [Git 回档](docs/GIT_RECOVERY.md) · [上游来源](UPSTREAM.md)
- [项目操作约束](AGENTS.md)

## 项目目录与入口

| 目录或文件 | 用途 |
| --- | --- |
| `src/supermimic/` | 训练、世界模型、策略、环境和机器人适配源码 |
| `scripts/training/`、`scripts/evaluation/` | 训练和评测入口；使用方法见[使用文档](docs/USAGE.md) |
| `scripts/data/`、`scripts/setup/` | 数据处理、重定向与环境资产准备 |
| `scripts/deploy/`、`scripts/experiments/` | 部署工具与实验专用入口 |
| `scripts/dev/`、`tests/` | 开发检查与测试 |
| `configs/`、`manifests/` | 项目配置与数据清单 |
| `docs/` | 使用说明、部署准备和按主题/日期保存的实验记录 |
| `media/` | 纳入版本管理的项目演示素材 |
| `素材/` | 本地原始视频素材，文件不纳入 Git |
| `outputs/` | 本地实验产物及预览，文件不纳入 Git |
| `vendor/` | 本地外部依赖，文件不纳入 Git |
| `pyproject.toml`、`uv.lock`、`requirements/` | Python 项目、依赖锁和环境配置 |
| `AGENTS.md`、`UPSTREAM.md`、`LICENSE` | 操作约束、上游来源和许可证 |

`sm-train` 和 `scripts/training/train.py` 对应同一训练入口。脚本按用途归类；实验脚本的具体参数和运行位置以对应记录为准。

## 运行位置

| 用途 | 主机 / 路径 |
| --- | --- |
| 本地源码 | `/Users/orange/project/supermimic` |
| 训练服务器 | `2025-Long-robot` |
| 服务器源码 | `/home/2025-Long.xh/SSD/supermimic` |
| 数据、资产和产物 | `/home/2025-Long.xh/HHD/data/supermimic` |
| 主训练环境 | `/home/2025-Long.xh/HHD/venvs/supermimic` |
| SOMA 环境 | `/home/2025-Long.xh/HHD/venvs/supermimic-soma` |

仿真、重定向、训练和评测在训练服务器运行；机器人 PC2 上的项目程序在 rootless Podman 容器 `lxh` 内运行。环境、网络、GPU 和实机操作要求以 [AGENTS.md](AGENTS.md) 为准。

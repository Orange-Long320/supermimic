# X2 Ultra PC2 `lxh` 容器准备（2026-10-31）

## 实际状态

通过用户授权的 SSH 别名 `x2-robot` 连接到 PC2（`10.0.1.41`，Jetson Orin NX，Ubuntu 22.04，SKU `X2 Ultra`）。官方容器指南使用 rootless Podman；Docker 系统服务处于 masked 状态。未解除屏蔽或修改系统服务，也未停止其他人的容器。

按官方方案用本机现有镜像 `localhost/agi/aimdk-dev:dev`（镜像 ID `0db3dbf9c9bdcf93d637698f2b281e7852c32d970f4f87ed3ca59e70b2d9038d`）创建并启动了独立容器 `lxh`。采用 host 网络和 IPC、只读挂载宿主 ROS、SDK 和 DDS 配置；当前不挂 GPU 或完整 `/dev`，CPU 试运行。此容器不含已通过验收的电机控制器。宿主数据备份在 `/home/agi/lxh-supermimic/`，容器内独立数据路径相同；两者不是绑定挂载，修改不会同步。不要覆盖或清理其他用户的容器或 SDK。

## 模型与参考

- 用户选择 11500 步 checkpoint，SHA256 `ff1f91d33631530d9d7f90f9f0675359fa1bff18445df43449312b9cb85db34b`；导出 `policy.pt` SHA256 `4424f231fc1896da098b2fca366e8522df6791596638b2b9257ba682520bbe4c`。
- 使用当时用户选择的 `jog_parallel_stance_20261031_r1/reference/dataset`，manifest SHA256 `40a2a31c1b84160f2ea6aa26548eb966f2e3c9ec23e10fa46ec1e9b2532b5b7d`；10 个片段在容器内均与 manifest 的哈希一致。
- 用已开展相同参考官方仿真对照的源码 `ac6c755`，压缩包 SHA256 `de860f31e6f739bc7f018380d59afde2ec99099a2899607278fb5677bb606b90`。`source-selected-ac6c755` 为原文件；`source-runtime-ac6c755` 只将 `supermimic/envs/__init__.py` 替换为不初始化训练环境的文档字符串。另复制 mjlab 1.6.0 的原版数学模块（只含 math，并保留许可证），并非完整 mjlab。
- 推理使用 Python 3.10、torch `2.14.1+cpu`、numpy `2.2.6`、scipy `1.15.3`、mujoco `3.16.0`。与服务器 Python 3.12、numpy 2.5.4、scipy 1.18.1 不同，进行了下面的固定输入验证；不能据此证明所有输入或训练环境兼容。运行依赖安装在容器内部目录 `python-cpu`，不是宿主系统。
- `lxh-fk-only` 删除网格和碰撞，仅用于参考锚点正运动学；服务器对 50 个随机姿态逐身体比较，与原模型的最大位置和四元数分量差均为 0。不得用于动力学、碰撞、受力或仿真验收。

## 已验证和限制

在容器中以服务器固定输入对比 `policy.pt`，200 次纯 CPU 推理最大绝对输出误差 `3.94e-6`；中位 13.93 ms、P99 17.12 ms、最大 17.62 ms。没有包含观测、调度、通信、计时或执行器，不能作为 50 Hz 的实时性验收。未来参考使用合成输入依次覆盖五技能与站立，共 360 帧，与服务器最大绝对差 `1.79e-7`；它不是在线实机参考验证。

只读 ROS 订阅收到 31 个期望关节、首个样本错误码均为 0，胯部 IMU topic 收到 `frame_id=base_link`；单次读取不构成连续稳定性、物理零位、方向或 IMU 标定验收。三个候选 odometry 话题存在但在检查时均为 0 个发布者、6 秒内无样本，骨盆位置估计尚不可用。机载 SDK 的 `/home/agi/aimdk/version` 显示 `aimdk v0.0.0-g14fe2e8`，PC2 固件标记 `release-lx2501_3_t2d5-soc1-v0.9.4`；与用户描述的 1.1.5 是否一致尚未确认，不推测。

**还没有实机键盘控制、控制权切换、硬件看门狗或电机发布器。** 仿真收步距离和站立质量仍有限，状态估计、标定、实时性、硬件约束、失联处理和受控安全验证均未完成。不要运行仿真专用的关节发布脚本控制实机。结果在本项目 `outputs/x2_lxh_pc2_20261031_r1/`；容器中的 `inference-report.json`、`goal-report.json`、`reference-check.json`、`odom-inventory.json` 与其对应。

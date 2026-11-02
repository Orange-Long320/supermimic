# X2 在线只读骨盆定位接入（2026-10-31，链路已运行）

> 历史记录：用户已于 2026-11-02 决定放弃点云与 FAST-LIO 定位路线，当前改用 [500 Hz IMU 计算路线](X2_SHADOW_CONTACT_FUSION_20261102.md)。本文保留原始实验结果，不作为当前实施方案。

用户要求打通实时雷达/IMU→骨盆状态链路，并确认当前机器人静止、双脚接地。范围为 PC2 `lxh` 容器内在线定位和日志验证；不发送电机命令、不切换控制权。

修改前快照 `c72dc55`。PC2 主机和 rootless 容器已核验，CPU ARM64、Ubuntu 22.04、Python 3.10；机载目录 `/home/agi/lxh-supermimic/online_lio_20261031_r1` 在容器内部。原部署版本保留。

复用既有 FAST-LIO 源码归档，SHA256 `2586007a7d816df288034f65dfa44c5d1734b605c75777809cb4ad98cf321cb0`，来源和提交见 `configs/localization/x2_fastlio_sources.json`。服务器现有二进制是 x86-64，不能直接在 PC2 使用，需容器内 ARM64 构建。

计划接入实时点云/IMU，使用已核对的字段转换、原生静态 TF、设备腰部 URDF FK；LIO 校正之外用带年龄检查的 IMU 传播输出状态。原生 header 时间与硬件采样同步仍有已知未验证项，不声明控制精度通过。程序不得将过旧状态换成新时间戳伪装成新测量。

实时点云/IMU → FAST-LIO → IMU 因果传播 → 设备腰部 FK → 骨盆 odometry 已打通并经独立订阅确认。只读链路保持运行，没有电机发布器或控制客户端。动态精度、模型闭环、准备站姿、吊挂卸载和接管均未验收。依赖仅安装在现有 `lxh` 容器，没有修改机器人宿主、其他容器或只读 SDK。

容器缺少 PCL/Eigen 开发依赖；已用任务专用清华 Ubuntu ports 源安装容器内依赖，保存安装计划与前后包清单。首次配置发现只读挂载的 ROS 导出元数据引用宿主 NumPy 绝对头文件路径；仅在本项目 CMake 导入目标中改用容器当前解释器实际的 NumPy include，没有修改 SDK 文件或伪造缺失目录。失败 `build.log` 和续建 `build-r2.log` 分别保留。

`online_inertial.py` 的 5 个因果传播/地面测试已在服务器和 ARM64 容器各通过一次。在线初始化使用实时静止 IMU/关节检查、点云地面拟合和设备 URDF 腰部 FK。输出沿用最新已接收 IMU 的采样帧头，不将发布时间当成新采样时间。程序只发布独立传感器适配、骨盆 odometry 和状态，状态固定 `control_ready=false`、`motor_output_enabled=false`。

## 当前会话与版本

- 下文 `RUN=/home/agi/lxh-supermimic/online_lio_20261031_r1`，均为容器内部路径，不与宿主同名目录自动同步。当前会话 `RUN/live-r2`，session ID `3b2c7037a3814182b92a5cebe1699ad1`；记录时 bridge PID 6158、FAST-LIO PID 6195，均为容器 PID，信号操作前须重新核对完整命令行。
- 实现快照 `00ca7b5`、`7234893`；运行中的桥接/点云转换代码为 `5cfa5c1`。最终记录与停止入口另作 Git 快照，确切提交号保存于 `RUN/provenance.json` 和服务器源代码 bundle。
- FAST-LIO 上游提交 `2fffc570a25d0df172720bac034fbdb6a13d2162`，ikdTree 来源见上述 sources 配置；私有补丁只导出内部状态、加速度比例和回调计数，不改变滤波计算。
- ARM64 可执行文件 `RUN/install/lib/fast_lio/fastlio_mapping`，2,015,760 字节，SHA256 `f2bf14a23b9fe0f69ec4bc1df7a388841e27b9f10f37385386b61b469838e3b9`。
- Python 3.10.12、ROS Humble、PCL 1.12.1、Eigen 3.4.0、g++ 11.4。仅 CPU，无 GPU。原 11500 软件部署保留，见 [部署记录](X2_PC2_RELEASE_20261031.md)；当前持续进程没有策略推理。

## 输入、坐标与时效

输入为实时胸前雷达点云（约 10 Hz）、其 IMU（约 200 Hz）及四组 HAL 关节。点云沿用已核对字段，纳秒转秒并排序，没有生成方位角假时间。原生 host header 保留；已知约 15 ms 点云帧头差异没有猜测修正，硬件实际采样同步仍未核验。

LIO 校正外使用去偏置、滤波器重力和比例校正的 IMU 传播，仅到最新已收到的 IMU 时间，不向未来外推。最大 LIO 年龄 200 ms、IMU 年龄 50 ms、IMU 间隔 30 ms、关节年龄 50 ms、组间偏差 30 ms；不满足条件时停止位姿发布并记录原因。

初始化要求连续 200 个静止 IMU 样本和低关节速度，再按重力调平、实时地面拟合设定地面高度。此次地面 946/1199 内点、拟合 RMS 7.22 mm、倾角 0.922°，是几何拟合结果，不是真值精度。每次会话重设 XY 原点和地面坐标，不能跨会话拼接模型历史。

设备 URDF 腰部链及原生 torso→LiDAR→IMU 安装变换用于移动安装点 FK，并扣除腰部相对运动及安装点速度。冻结变换在 `RUN/extrinsics.json`；当前实时 TF 仅重新见到并核对 `lidar_chest_front → lidar_imu_chest_front`，torso→LiDAR 仍用此前原生记录，不能声称全部安装 TF 已在线复核。

输出 `/supermimic/pelvis/odometry`，父坐标 `supermimic_world`、子坐标 `pelvis`；四元数 ROS xyzw，Odometry twist 在子坐标，日志另存世界速度。内部传感器、状态、TF 隔离到 `/supermimic/x2_online_lio/*`，不替换原生 TF。状态话题 `/supermimic/x2_online_lio/status`。协方差未验收，不可当作精度保证。

## 实测结果与边界

独立订阅 `RUN/audit-r2/`：30 秒观察窗口收到 1395 帧；49.09 Hz 是首末接收帧之间的频率，不是 50 Hz 硬实时保证。

| 项目 | 实测 |
| --- | --- |
| 位姿发布者 | 1 |
| 接收间隔 P50 / P99 / 最大 | 20.13 / 33.09 / 129.93 ms |
| 大于 60 ms 的间隔 | 3 次 |
| 帧头年龄 P50 / P99 / 最大 | 11.60 / 45.64 / 73.02 ms |
| 数值和时间戳 | 全部有限，四元数范数约 1，帧头非递减；1394 个不同采样时间 |
| 该窗口高度 | 0.61946–0.62686 m |
| 该窗口相对首帧最大 / 末帧位置变化 | 2.51 / 0.71 cm |

帧头年龄不是已验证的传感器端到端物理延迟。一个相同采样时间被保留，没有为制造严格递增而改写时间。

全程摘要 `RUN/whole-session-summary.json`：最初 363.22 秒、17781 帧中，相对首帧最大估计位置变化 **10.46 cm**、末帧变化 7.01 cm，高度 0.61934–0.63705 m。包含预热和故意暂停定位的测试，期间没有独立运动真值，也没有连续确认实际姿态不变。这些是估计值变化，不能称为测得的定位误差，亦不能仅凭较好的 30 秒窗口放行控制。

只暂停经 PID/完整会话参数确认归属本任务的 FAST-LIO，未操作机器人控制。第二次暂停 1.403 秒，独立观察器确认暂停 300 ms 后至恢复之间收到 **0 帧**，状态 `LIO correction stale`；恢复后约 66.3 ms 收到首帧。证据为 `dropout-event-r2.json`、`audit-dropout-r2/`、`dropout-verification.json`。第一次观察窗口未覆盖暂停，只保留为未对齐尝试，不作为验证证据。该测试验证诊断输出时效门控，不是硬件看门狗验收。

桥接 Python 仍约占一个 CPU 核，C++ 约 16% 单核占用；正常运行偶有 IMU 过旧、采样间隙或关节偏差拒绝。`dynamic_accuracy_validated=false`、`hardware_sampling_time_verified=false`、`covariance_validated=false` 保留。腰俯仰、双肩侧摆、左腕俯仰与模型/设备文件限位不一致的问题未裁剪或零偏绕过。

## 失败记录与检查

`build.log` 保留首次配置失败；`build-r2.log` 已编译安装成功，末尾退出 127 是缺少 `file` 工具的后置检查，之后以 Python ELF 头确认 AArch64，构建脚本也改用此方式。容器依赖安装含传递包更新，保存 `apt-*.log`、`packages-before.txt`、`packages-after.txt`；没有改宿主或共享 SDK。

`live-r1` 的 PointCloud2 字节赋值触发生成 setter 的逐字节验证，导致 IMU 排队过旧而无位姿。根因修复位于原有 `prepare_x2_fastlio_bag.py`：改用 `array('B', ...)` 批量赋值，字段和线格式不变。1 MiB setter 基准约 150 ms 降至 0.08 ms；`live-r2` 点云完整回调中位约 2.9 ms、上述观察窗最大约 11.0 ms。不能据此声称 CPU 占用已降低；首轮失败日志保留。

ARM 容器通过 12 项影子内核、7 项骨盆几何、5 项在线惯性、4 项点云/变换测试，共 28 项不同测试。5 项惯性测试在服务器也通过，不重复计数。依赖安装后复检原部署 200 次固定输入模型误差约 `3.94e-6`、360 帧合成目标误差约 `1.79e-7`，仅验证兼容性，不是在线模型/执行器时序验收。

## 维护入口

当前会话保持运行，无开机自启。连续日志在 `RUN/live-r2/`（`status-latest.json`、`status.jsonl`、`pelvis-frames.jsonl`、`initialization.json`、`command.json`、`fastlio.log`）。有界证据快照和源码备份保存至项目服务器 `/home/2025-Long.xh/HHD/data/supermimic/logs/x2_online_lio_20261031_r1`；快照截止时间以 manifest 为准，持续日志仍留机载。

停止本任务定位（核验 PID 身份后仅停止该会话，不发送机器人命令）：

```bash
ssh x2-robot 'podman exec lxh /usr/bin/python3 /home/agi/lxh-supermimic/online_lio_20261031_r1/code/scripts/stop_online_x2_lio.py --session /home/agi/lxh-supermimic/online_lio_20261031_r1/live-r2'
```

如需重启，先停止旧会话，再选择未存在的新目录并满足实际静止条件。以下 `NEW_SESSION` 是待替换名称，未执行：

```bash
ssh x2-robot 'podman exec -d lxh bash -lc "bash /home/agi/lxh-supermimic/online_lio_20261031_r1/run-online.sh /home/agi/lxh-supermimic/online_lio_20261031_r1 /home/agi/lxh-supermimic/online_lio_20261031_r1/NEW_SESSION > /home/agi/lxh-supermimic/online_lio_20261031_r1/NEW_SESSION-launch.log 2>&1"'
```

任务锁阻止重复启动；重启改变会话与坐标原点。没有启动或部署自动电机服务。

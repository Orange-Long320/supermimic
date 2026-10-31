# X2 Ultra 雷达、IMU 与估计器只读检查（2026-10-31）

结论：本次已收到雷达点云、5 路 IMU 和原厂腿部里程计；之前短时未收到里程计的记录不能用于描述本次状态。原厂估计器能否脱离 MC 独立运行仍未确认。

用户仅授权只读检查。没有停止/启动原厂应用、容器或运控，没有切换模式、调用控制服务、发布电机命令或启动建图/重定位。探针在 PC2 执行，仅订阅和读取 ROS 图；每个探针结束后销毁自身节点。没有修改机器人配置或安装依赖。

## 设备与证据

- 通过 `ssh x2-robot` 连接 PC2，主机名 `agi`、`aarch64`；环境标记 `AGIBOT_SKU=X2 Ultra`、硬件 `t2d5`、软件 `v1.1.5`。
- SDK 文件标记仍为 `aimdk v0.0.0-g14fe2e8`，不能直接据此认为 SDK 和软件版本一一对应。
- 现有开发容器均为 `Created`，本次没有启动。解释器 `/usr/bin/python3`，NumPy 1.26.1，使用现有 ROS Humble 与 `/home/agi/aimdk/install`。
- 本地独立证据目录：`outputs/x2_sensor_readonly_20261031_162111/`。含 `probe.py`、`sensor-audit.json`、`timing-probe.py`、`imu-timing.json`、`tf-odom-probe.py`、`tf-odom.json` 和 `config-evidence.json`；最后一份保留所读配置的 SHA256 与行号摘录。原始证据保持忽略，不提交。
- 第一轮 32.23 秒，实际收到样本的跨度约 29.8 秒（包含发现时间）；第二轮仅订阅三路 IMU 与静态 TF，22.02 秒；第三轮仅订阅腿部里程计与静态 TF，12.00 秒。

探针执行方式（文件路径按对应探针替换）：

```bash
ssh -o BatchMode=yes -o ConnectTimeout=8 x2-robot \
  "bash -c 'source /home/agi/.aima/env/bashrc; source /home/agi/aimdk/install/setup.bash; /usr/bin/python3 -'" \
  < outputs/x2_sensor_readonly_20261031_162111/probe.py
```

## 数据实测

| 数据 | 收到样本 / 接收频率 | 结果与范围 |
| --- | --- | --- |
| `/aima/hal/sensor/lidar_chest_front/lidar_pointcloud` | 298 帧 / 10.001 Hz | 每帧 25,639–25,692 点；全部收到的约 765 万个 XYZ 均有限，消息布局一致 |
| `/aima/hal/sensor/lidar_chest_front/imu` | 复测 4,402 条 / 200.09 Hz | 加速度、角速度字段存在；首轮无非有限数 |
| `/aima/hal/imu/chest/state` | 复测 11,008 条 / 500.33 Hz | 首轮姿态四元数范数约 1；加速度、角速度均有限 |
| `/aima/hal/imu/torso/state` | 复测 11,005 条 / 500.23 Hz | 首轮姿态四元数范数约 1；加速度、角速度均有限 |
| `/aima/hal/sensor/stereo_head_front/imu` | 首轮 5,578 条 / 186.95 Hz | 收到数据，未单独复测其频率 |
| `/aima/hal/sensor/rgbd_head_front/imu` | 首轮 2,906 条 / 97.37 Hz | 收到数据，未单独复测其频率；orientation 协方差首项 -1 |
| `/aima/mc/leg_odometry` | 第三轮 2,025 条 / 169.60 Hz | 位姿和 twist 均有限，四元数范数约 1，时间戳持续前进 |

第一轮同时解码两路点云并订阅全部 IMU/里程计，depth=1；胸部/胯部 IMU 接收频率仅约 447/440 Hz、雷达 IMU 约 189 Hz。改用 depth=200、移除点云处理后恢复约 500/500/200 Hz，说明首轮接收统计受探针处理范围与队列设置影响，不能据首轮频率判定驱动不足频。即使复测接收频率接近标称，单次接收最大间隔仍约 36 ms；不是实时性验收。

全部有样本的话题在首轮均未发现 header 时间戳重复或回退。胸部/胯部 IMU 的 header 是机器人接收时间，不能将 `wall_minus_stamp` 当成传感器采样到控制的端到端延迟。雷达 IMU 首轮 orientation 始终为单位四元数，不能将此字段当作已验证姿态输出。

点云字段实测为 `x/y/z/intensity/ring/timestamp`，`point_step=32`；ring 范围 0–95。末帧 timestamp 范围 332,000–95,174,000，时间单位与扫描内参考点语义尚未从驱动接口核验，不能直接传给去畸变算法。当前驱动配置为 `use_lidar_clock: false`。

## 外参与坐标系

前两轮使用 BEST_EFFORT + TRANSIENT_LOCAL 订阅 `/tf_static`，发现发布者但未收到历史样本；第三轮改为 RELIABLE + TRANSIENT_LOCAL 后收到 22 对变换。因此前两轮无样本不能表述为外参不存在。

已收到关键静态变换：

- `base_link -> pelvis`；
- `torso_link -> lidar_chest_front`；
- `lidar_chest_front -> lidar_imu_chest_front`；
- `pelvis -> imu_in_pelvis_link` 与 `torso_link -> imu_in_torso_link`。

腿部里程计发布者为 `/mc_ros2_node3888`，`frame_id=leg_odom`、`child_frame_id=lidar_imu_chest_front`。它不是骨盆位姿的直接输出；接入需使用腰部关节对应的动态运动学变换。本次仅检查静态变换存在与数据内容，未验收完整动态 TF 链、外参精度、物理轴向或运动中状态估计精度。

## 原厂估计器独立运行调查

已在 PC2 只读检查：

- SLAM 进程 `4616` 以 `./slam --cfg_file_path=../config/slam/slam_config.yaml` 运行。
- 该配置的 MappingModule 与 LocalizationModule 都把 `/aima/mc/leg_odometry` 列为 `sub_dr_topic`（259、278 行）。这证明配置存在依赖输入，不证明腿部里程计是所有算法路径的硬性前提，也不证明 SLAM 可以在 MC 停止后独立输出。
- 原厂腿部里程计与部分 TF 的 ROS 发布者都是 `mc_ros2_node`。这是当前发布链证据，不能仅凭节点名称判定估计器内部是否可拆分。
- 在已安装 SDK 的相关源文件中未找到 `leg_odometry` 或 estimator-only 开关。检查范围不包含未获访问的 PC1 运控包。
- 经 PC2 跳转可以到达 PC1 `10.0.1.40` 的 SSH，但当前密钥认证被拒绝；没有读取到 PC1 的 MC 进程/启动配置，没有尝试停止应用或绕过认证。

[官方系统模式文档](https://x2-aimdk.agibot.com/zh-cn/latest/Interface/system_state/index.html)说明 `Develop_MC` 会停用原生运动控制；未列出只运行估计器的模式。[官方 FAQ](https://x2-aimdk.agibot.com/zh-cn/latest/faq/index.html)要求直接控制 HAL 前停止 MC。[第三方项目](https://github.com/chifongip/RoboJuDo-Plus#todo)报告停止 MC 后腿部里程计停止发布，但本机没有执行该测试。

当前不能确认原厂支持独立估计器。需进一步只读检查 PC1 运控配置，或由原厂确认 v1.1.5 是否支持关闭电机输出、保留估计器的接口。现有雷达/IMU数据和关键静态外参具备后续离线评估条件；时间同步、动态坐标变换与运动精度尚未验证。

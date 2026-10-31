# X2 独立 FAST-LIO 离线探索（2026-10-31）

后续 [逐点时间与离线静止复核](X2_LIO_TIMING_AUDIT_20261031.md) 已从原始包核实逐点字段，并发现约 15 ms 的点云帧头偏差；本页保留首次探索的配置与结果，不代表后续验证已通过。

状态：传感器录制、输入适配、服务器 CPU 编译和一次离线回放已完成；尚未验证动态定位精度，不是可接入机器人控制的定位器。无预建地图、无原厂腿部里程计输入，地图由当前点云增量构建。

用户希望在原厂 MC 停止后使用不依赖预建地图的状态估计。本次延续只读传感器范围：没有启动/停止原厂应用或容器，没有切换机器人模式，没有发布电机命令，也没有请求或驱动机器人移动。机器人仅新增本任务目录中的录制脚本与原始 bag；FAST-LIO 在项目服务器运行，未部署到机器人。

## 路线与来源

- 算法来源：[hku-mars/FAST_LIO](https://github.com/hku-mars/FAST_LIO)。本次实际构建的是 [Ericsii/FAST_LIO_ROS2](https://github.com/Ericsii/FAST_LIO_ROS2) `2fffc570a25d0df172720bac034fbdb6a13d2162`，ikd-Tree 子模块 `e2e3f4e9d3b95a9e66b1ba83dc98d4a05ed8a3c4`。
- [RuanJY/robosense_fast_lio](https://github.com/RuanJY/robosense_fast_lio) 明确包含 E1R 适配；其 [E1R 配置](https://github.com/RuanJY/robosense_fast_lio/blob/master/config/robosenseE1R.yaml)仅作为传感器接入参考，未将其结果当作本机验收。
- 本次把 X2 的 `PointCloud2` 转成 ROS2 端口已有的逐点时间输入格式（`lidar_type=2`、feature extraction 关闭），保留真实逐点时间，禁止退回用旋转雷达方位角推算时间的路径。ring 以本机实测 0–95 校验。没有修改 FAST-LIO 的估计数学或用虚假里程计补输入。
- 上游 CMake 声明的 `pcl_ros` 未被源码使用，实际使用了 `tf2_ros`；本次只替换该构建依赖。补丁在 `configs/localization/x2_fastlio_dependencies.patch`。不安装系统包、不改共享环境。
- 上游仍编译包含 Livox 接口的分支，因此从官方 `Livox-SDK/livox_ros_driver2` `21445540f0d100dc86a7e6df312dd70bbdb4afdf` 提取两个原版消息定义和许可证，构建独立的接口包；未安装或启动 Livox 驱动。来源与哈希在 `configs/localization/x2_fastlio_sources.json`。

## 运行位置与快照

- 机器人原始录制：`/home/agi/lxh-supermimic/lio_readonly_20261031_r1/capture/`。
- 服务器独立目录：`/home/2025-Long.xh/HHD/data/supermimic/logs/x2_fastlio_offline_20261031_r1/`。源码、构建、接口包安装、bag、转换结果和日志均位于其中，未改服务器主源码 checkout 或项目训练环境。
- 本地仅保存小型报告：`outputs/x2_fastlio_offline_20261031_r1/`；原始 bag 未落到本地源码仓库，保留在机器人与服务器。
- 录制源码快照 `da2d5be`；构建准备 `2a98942`；实际转换/回放源码快照 `74468fc`。构建 PID `3680936`，离线执行会话 `78844`，均已结束。
- 服务器 Python 3.10.12、ROS Humble、NumPy 1.21.5、Eigen 3.4.0、PCL 1.12.1；只使用现有 CPU 依赖，编译 nice=10、`-j2`，没有使用 GPU。
- 源码归档 SHA256：`2586007a7d816df288034f65dfa44c5d1734b605c75777809cb4ad98cf321cb0`。服务器 `manifest.json` 包含各脚本、二进制和数据文件的 SHA256。

## 录制与适配

`scripts/deploy/record_x2_lio_readonly.py` 在 PC2 运行 20.00 秒，收到了 194 帧原始点云、3,884 条雷达 IMU 和 4 条静态 TF 消息。雷达数据 header 覆盖约 19.3 秒，IMU 约 19.4 秒。没有请求运动，也没有独立测量机器人是否完全静止。

原始数据库在传输前后 SHA256 一致：

```text
84959dd1ee33f16068cdd42304415d42a908cce8f9d2908777c9926ba7942199
```

`scripts/evaluation/prepare_x2_fastlio_bag.py` 的适配内容：

- 从原始 TF `lidar_chest_front -> lidar_imu_chest_front` 求逆，生成 FAST-LIO 所需的“LiDAR 在 IMU 坐标系中的位姿”。本机平移约 `[0.0042744, -0.0157518, -0.0112120]` 米，旋转矩阵随 audit 保存，不使用单位阵替代。
- 输入 `timestamp` 明确采用 **候选**缩放 `1e-9`，得到帧内约 0.332–95.181 ms；与 10 Hz 扫描时长相符，但单位、帧内零点与 header 的对应关系尚未获得驱动源码/原厂接口的独立确认。这不是时间同步验收。
- 按真实逐点时间稳定排序，再写为 `time` 秒值，避免 FAST-LIO 将非最大末点时间误当作扫描结束；保留位置、强度与 ring 的对应关系。
- 雷达 IMU 加速度模长实测约 9.809–9.828 m/s²。姿态字段没有用作姿态真值。
- 外参在线估计关闭，软件时间自同步关闭，时间偏移候选设为零；`cube_side_length=100`，点云/PCD 导出关闭。配置中的噪声与滤波参数只是本次探索设置，未经调参与验收。

## 一次离线回放结果

服务器仅本机 ROS 域 230（`ROS_LOCALHOST_ONLY=1`），输入和输出均使用 `/supermimic/x2_lio_r1/` 下的独立话题；未连接机器人 ROS 网络。回放速率 0.5，未测机载实时性能。

- 发布了全部 194 帧点云、3,884 条 IMU。
- FAST-LIO 输出 188 个位姿，时间跨度 18.900 秒；均为有限值，四元数范数约 1，进程正常退出。
- 初始化阶段一条 `No point, skip this scan` 警告，随后初始化局部地图；没有 ERROR 日志。
- 从第一个输出算，全程净位置变化 6.92 cm、最大变化 7.81 cm。
- 去掉前 5 秒后，以余下首帧重设比较起点，13.80 秒内净位置变化 1.10 cm、最大变化 1.84 cm。

这些是算法输出的相对位置变化，**没有运动真值，不能称为定位误差或静态漂移验收**。它们只证明当前输入可驱动这套离线算法产生有限位姿；不能证明走路、小跑、停步或快速转身时精度足够，也不能证明长时间稳定。

4 个格式/几何测试在服务器通过：非单位旋转下的外参求逆、无效四元数拒绝、非连续大端字段与乱序点时间转换、错误时间单位拒绝。测试不代替动态估计验证。

## 可复查的命令与产物

原始录制命令（机器人已结束，不应为看结果而重复录制）：

```bash
source /home/agi/.aima/env/bashrc
/usr/bin/python3 /home/agi/lxh-supermimic/lio_readonly_20261031_r1/record.py \
  --out /home/agi/lxh-supermimic/lio_readonly_20261031_r1/capture --seconds 20
```

服务器转换与回放命令（位于上述独立运行目录；输出目录必须不存在）：

```bash
source /opt/ros/humble/setup.bash
source install/share/livox_ros_driver2/local_setup.bash
source install/share/fast_lio/local_setup.bash
unset FASTRTPS_DEFAULT_PROFILES_FILE
export ROS_DOMAIN_ID=230 ROS_LOCALHOST_ONLY=1
/usr/bin/python3 prepare_x2_fastlio_bag.py \
  --bag raw/capture/bag --out prepared --point-time-scale 1e-9
/usr/bin/python3 replay_x2_fastlio_offline.py --prepared prepared \
  --binary install/lib/fast_lio/fastlio_mapping --out replay_r1 --rate 0.5
/usr/bin/python3 code/tests/test_x2_fastlio_bag.py
```

构建入口 `scripts/setup/build_x2_fastlio_offline.sh` 只接受已校验的源码归档，使用独立前缀。首次手工构建在读取不存在的聚合 `install/setup.bash` 时退出；独立 CMake 安装只生成包级环境脚本，确认后改用 `install/share/livox_ros_driver2/local_setup.bash` 续建成功。失败日志 `build.log` 与成功续建日志 `build-fastlio.log` 均保留，没有调整系统依赖版本。

主要结果：`prepared/audit.json`、`prepared/fastlio.yaml`、`replay_r1/report.json`、`replay_r1/trajectory.csv`、`replay_r1/settling-analysis.json`、`replay_r1/command.json`、`adaptation-tests.log`、`manifest.json`。

## 后续必要工作

先核实点云的时间单位、帧内参考点与 IMU 同步关系，再用有运动和独立位移/转角参考的数据验证。当前 FAST-LIO 输出随点云更新，约 10 Hz；若未来接入 50 Hz 策略，还需经验证的 IMU 前向传播和延迟处理。输出表示雷达 IMU 的位姿，需要结合腰部关节动态运动学转换到骨盆。本次没有实现这些控制侧环节，也没有部署或接管实机。

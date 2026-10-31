# X2 逐点时间与离线静止定位复核（2026-10-31）

逐点时间字段的单位和参考点已用原始包验证；发现点云帧头相对 IMU 存在约 15 ms 的打戳偏差。已增加按原始包扫描起点恢复点云帧头的离线模式。完整 200 Hz IMU 的硬件时间仍未恢复，不宣称采样同步或动态定位精度通过。

用户确认录制期间机器人一直静止。全部机器人操作仅限已有传感器话题订阅、原始数据录制和文件读取；没有发送电机命令、切换模式、启动/停止原厂应用、修改驱动/时钟/网络配置或安装系统依赖。FAST-LIO 仅在项目服务器的 localhost ROS 域 230 回放，无腿部里程计或预建地图输入。

## 数据和源头证据

机器人目录：`/home/agi/lxh-supermimic/lio_time_audit_20261031_r1/`。服务器独立目录：`/home/2025-Long.xh/HHD/data/supermimic/logs/x2_lio_time_audit_20261031_r1/`。

| 录制 | 墙钟时长 | 点云 | 雷达 IMU | 原始包快照 |
| --- | --- | --- | --- | --- |
| `capture_native` | 15.00 s | 151 | 3,007 | 150 |
| `capture_static60` | 60.01 s | 598 | 11,958 | 598 |

原始 bag 在机器人与服务器的 SHA256 一致：

- 15 秒：`cfdd8aa2a4a5a9f380708ea02eb1021bd257237820b37564ece262db4be20329`。
- 60 秒：`4ec40fb51f082aea3a4d1f1b214a0f8f9945b3a5faff5bb453228945e1c7192b`。

安装的 HAL 驱动为未剥离符号的 ARM64 ELF；只读检查使用本机已有 `nm`/`objdump`，没有运行其控制入口。文件 `/agibot/software/hal_sensor_orin/bin/aima-hal_sensor-app-main` 的 SHA256 为 `834074532c6df0879fb5ca01ed863156f285fd54245a5284a80a210d0b53ff3e`，Build ID `8c0329aad52bd33e7549979c9f705cc98dc23f5d`。选定函数的反汇编、原生消息定义及哈希保存在 `evidence/`。

原始消息类型取自现有 `/agibot/software/common/share/aimdk_msgs/local_setup.bash`，仅影响该次命令环境。开发 SDK 中没有 `LidarRawData`；第一次尝试 HAL 包级 overlay 时，缺少共享库路径而在订阅建立阶段退出，失败目录 `capture/` 保留。核对库归属后使用原生 common overlay 成功录制，没有修改或替换 SDK/系统库。

## 逐点时间的实际定义

本机 E1R `handleDataPacket`（`0x1ce1b0`，关键指令 `0x1ce530–0x1ce5a8`）先取包内的秒和微秒，再对每个点执行与下式等价的计算：

```text
packet_us = big_endian_seconds_6bytes * 1,000,000 + big_endian_microseconds_4bytes
point_delta_us = packet_us + point_offset_us - first_accepted_packet_us
timestamp = float64(float32(float32(point_delta_us) * 1000))
```

所以 `PointCloud2.timestamp` 是相对本帧首个已接受 MSOP 包时间的**纳秒单位偏移**。原始时间分辨率为微秒，并含 float32 中间运算；不是纳秒测量精度。不是绝对 Unix 时间，也不是相邻点之间的时间差。

`audit_x2_lio_timing.py` 从每个 1200 字节 MSOP 包重建时间和 ring，并复现本机驱动的距离过滤、浮点运算与字段布局：

- 15 秒有 150/150 对原始包/点云精确匹配，首帧点云缺少同步开始之前的对应 raw 快照，配对对照中显式排除。
- 60 秒有 598/598 对精确匹配，每帧首末包序号为 0/287、包数为 288。
- 实测字段范围 332,000–95,181,000 ns，即约 0.332–95.181 ms。
- 60 秒内硬件扫描起点间隔为 99.999–100.001 ms，无回退。

绑定配置时核对原始数据库 SHA256，防止将其他 bag 的时间审计或原始包锚点用于当前数据。

## 帧头与 IMU 同步关系

当前配置 `use_lidar_clock=false`。`handleFirstBlock` 的 false 分支（`0x1cdf00–0x1cdf2c`）以主机 `system_clock::now()` 生成 PCL 帧头，并量化至微秒；帧内点偏移仍来自硬件包。`publishImuData` 的 false 分支（`0x1d0180–0x1d01bc`）以 `rclcpp::Clock(RCL_SYSTEM_TIME).now()` 为 IMU 打戳。因此同用主机时钟并不代表点云和 IMU 的采样时刻已经对齐。

新协议 DIFOP 的硬件时间位于字节 100–109，六个 IMU 浮点量位于 180–203；这一参数顺序来自 `processNewVersionPacket`（`0x1d1014–0x1d1020`）的实际调用。以六个原始浮点量逐位匹配 ROS IMU：15 秒匹配 149 个快照、60 秒匹配 597 个快照；边界各一个快照无匹配。所有已记录 DIFOP 快照报告 PTP mode=2（E2E_L2）、status=1（成功）。

60 秒数据的时间差：

| 时间差 | 中位数 | P99 | 最大值 |
| --- | --- | --- | --- |
| 点云原生帧头 − 硬件扫描起点 | 15.185 ms | 16.669 ms | 17.136 ms |
| 匹配 IMU 的主机帧头 − 硬件时间 | 0.261 ms | 5.988 ms | 10.457 ms |
| 上述两种偏移之差 | 14.872 ms | 16.250 ms | 16.988 ms |

15 秒对应的相对偏移中位数为 15.089 ms。这里的差值是所录窗口中的证据，不是永久标定常数。没有搜索或调节固定偏移来优化漂移。

## 离线修正与对照

`prepare_x2_fastlio_bag.py --timing-audit ... --cloud-stamp-mode raw_sensor` 使用每帧 raw MSOP 的硬件起点替换点云帧头；保留原始 bag、逐点偏移和 IMU 主机帧头。`host` 对照使用相同配对点云集合，只保留原生点云帧头。两版使用相同外参、滤波参数和 IMU 数据。

这是点云帧头语义修正，尚非完整硬件采样同步：raw 话题每帧只保留一个 256 字节 DIFOP 快照，无法恢复所有 200 Hz IMU 的硬件时间，IMU 接收抖动仍在。配置中的 `time_semantics_verified=false` 保留，逐点字段另以 `point_time_semantics_verified=true` 标记。

初次 1 倍速回放的输出覆盖不一致（host 144 帧、raw_sensor 131 帧），不作为优劣对照。新增末帧覆盖检查，并以 0.5 倍速重放后，两版 15 秒数据均输出 146 帧、覆盖末帧，最大输出间隔分别约 102.93/100.02 ms。净静止位置变化约 1.85/1.67 cm，最大变化约 2.25/3.07 cm；没有一致改善，不能按净变化择优宣布修复。

60 秒原版输入队列的半速回放中，host 输出 594 帧，raw_sensor 输出 592 帧，后者有一次 300 ms 输出缺口；四分之一速度复测仍有缺口。因此仅覆盖末帧不足以认定回放完整，新增单调性及跨扫描周期缺口检查。

在单独的 `instrumented/` 副本加入输入回调、同步组和定位发布计数（`x2_fastlio_offline_counts.patch`），不修改估计数学。1 倍速诊断发布 598 帧点云/11,958 条 IMU，FAST-LIO 实际回调仅 593/11,691 次，同步组 593、发布定位 590 次；接收端准确记录了这 590 次发布。证明该次缺口发生在算法输入接收之前，不能当作完整输入的定位精度对照。

原版 PointCloud2 输入使用 `SensorDataQoS`（BEST_EFFORT、浅队列），IMU 使用 RELIABLE、depth=10。已在新的 `lossless/` 离线副本添加默认关闭的 `sm_offline_reliable_inputs` 参数：仅显式启用时，两种输入改为 RELIABLE/KEEP_ALL，有限 bag 的回放发布者也使用相同模式。计数器继续保留以核验实际接收数；该设置不部署到机器人，不改变算法参数，也不用于声明实时性。补丁为 `x2_fastlio_offline_input_qos.patch`，对应源码快照 `31d0978`。

15 秒可靠输入复测收到全部 150 帧点云/3,007 条 IMU，同步组 150，发布并记录 147 帧定位，无输出缺口。

最终 60 秒对照使用相同的可靠输入设置、1 倍速回放、相同 598 帧点云和 11,958 条 IMU。两版原生输入回调与发布计数完全一致，同步组均为 598，初始化后均发布并记录 595 帧定位，覆盖约 59.4 秒，输出时间戳单调且最大间隔分别约 101.61/100.02 ms。末帧已覆盖，末尾尚有少量 IMU 位于最后扫描之后，这是正常尾部未积分数据，不是输入丢失。

| 静止输出指标（3D 位移） | 原生主机帧头 `host` | 按原始包恢复 `raw_sensor` |
| --- | --- | --- |
| 全程净变化 / 最大变化 | 6.23 / 7.23 cm | 5.81 / 6.83 cm |
| 去掉前 5 秒后的净变化 / 最大变化 | 2.97 / 3.99 cm | 3.57 / 4.59 cm |
| 最后约 30 秒净变化 / 最大变化 | 0.49 / 1.25 cm | 0.82 / 1.82 cm |

每个窗口都从该窗口首帧重新计算相对位移。去掉前 5 秒后的实际跨度为 54.30/54.40 秒；最后窗口约 30 秒。这是用户确认静止条件下的单段输出漂移，没有独立的运动真值仪器。修正没有带来各窗口一致改善，不能按全程净变化稍小而宣称定位修复完成，也不能把旧浅队列结果用于比较精度。

最终可靠配对运行脚本 `run60-reliable-pair.sh`，父进程 PID `3719107`，源码快照 `31d0978`；运行已结束。原半速父进程 PID `3700101`、四分之一速父进程 PID `3705502` 也已结束。中间结果和失败尝试保留，不用于择优验收。汇总见 `static_comparison_reliable.json`。

复现入口（服务器独立运行目录；输出目录均须不存在）：

```bash
source /opt/ros/humble/setup.bash
source ../x2_fastlio_offline_20261031_r1/install/share/livox_ros_driver2/local_setup.bash
source lossless/install/share/fast_lio/local_setup.bash
unset FASTRTPS_DEFAULT_PROFILES_FILE
export ROS_DOMAIN_ID=230 ROS_LOCALHOST_ONLY=1
/usr/bin/python3 audit_x2_lio_timing.py --bag raw/capture_static60/bag --out audit_verified60
/usr/bin/python3 prepare_x2_fastlio_bag.py --bag raw/capture_static60/bag \
  --out prepared_sensor60 --point-time-scale 1e-9 \
  --timing-audit audit_verified60 --cloud-stamp-mode raw_sensor
/usr/bin/python3 replay_x2_fastlio_offline.py --prepared prepared_sensor60 \
  --binary lossless/install/lib/fast_lio/fastlio_mapping \
  --out replay_sensor60_reliable --rate 1 --reliable-inputs
```

host 对照只将 `--cloud-stamp-mode` 设为 `host`，使用不同输出目录。原版输入、仅计数和可靠输入使用不同的源码/二进制目录，未覆盖最初的探索版本。独立构建复用既有 Livox 消息接口前缀时显式设置 `livox_ros_driver2_DIR`；没有改动系统依赖版本。可靠模式下回放器会核验原生回调计数，不一致时标记错误并退出失败。

## 验证范围和后续

服务器通过 7 个格式、几何和协议测试：包括大端非连续字段、逐点关联和时间排序、外参方向、错误单位/四元数拒绝、微秒到纳秒维度、原始包形状和 CDR 边界。实际数据另做全部配对帧的原始包精确重建与输入绑定检查。

只有静止数据，不能通过运动激励验证时间偏移、去畸变效果、动态精度或骨盆状态估计。本次没有把静止漂移小当作动态定位通过。下一步若要完整验证 IMU 采样同步，需要完整速率 DIFOP 原始包或原厂提供硬件采样时间接口，再结合有独立运动参考的数据；不应通过插值生成未测硬件时间来宣称完成。

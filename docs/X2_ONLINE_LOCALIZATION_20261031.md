# X2 在线只读骨盆定位接入（2026-10-31，进行中）

用户要求打通实时雷达/IMU→骨盆状态链路，并确认当前机器人静止、双脚接地。范围为 PC2 `lxh` 容器内在线定位和日志验证；不发送电机命令、不切换控制权。

修改前快照 `c72dc55`。PC2 主机和 rootless 容器已核验，CPU ARM64、Ubuntu 22.04、Python 3.10；机载目录 `/home/agi/lxh-supermimic/online_lio_20261031_r1` 在容器内部。原部署版本保留。

复用既有 FAST-LIO 源码归档，SHA256 `2586007a7d816df288034f65dfa44c5d1734b605c75777809cb4ad98cf321cb0`，来源和提交见 `configs/localization/x2_fastlio_sources.json`。服务器现有二进制是 x86-64，不能直接在 PC2 使用，需容器内 ARM64 构建。

计划接入实时点云/IMU，使用已核对的字段转换、原生静态 TF、设备腰部 URDF FK；LIO 校正之外用带年龄检查的 IMU 传播输出状态。原生 header 时间与硬件采样同步仍有已知未验证项，不声明控制精度通过。程序不得将过旧状态换成新时间戳伪装成新测量。

当前为准备记录，构建、运行频率、延迟和动态精度均待实测。依赖仅允许放入本项目现有 `lxh` 容器或专用前缀，不修改机器人宿主、其他容器或共享 SDK。

容器缺少 PCL/Eigen 开发依赖；已用任务专用清华 Ubuntu ports 源安装容器内依赖，保存安装计划与前后包清单。首次配置发现只读挂载的 ROS 导出元数据引用宿主 NumPy 绝对头文件路径；仅在本项目 CMake 导入目标中改用容器当前解释器实际的 NumPy include，没有修改 SDK 文件或伪造缺失目录。失败 `build.log` 和续建 `build-r2.log` 分别保留。

`online_inertial.py` 的 5 个因果传播/地面测试已在服务器和 ARM64 容器各通过一次。在线初始化使用实时静止 IMU/关节检查、点云地面拟合和设备 URDF 腰部 FK。输出沿用最新已接收 IMU 的采样帧头，不将发布时间当成新采样时间。程序只发布独立传感器适配、骨盆 odometry 和状态，状态固定 `control_ready=false`、`motor_output_enabled=false`。

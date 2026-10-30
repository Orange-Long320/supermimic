# X2 Ultra 部署基础（实验阶段）

2026-10-30：本次用户要求增加实机部署框架，范围扩展到为实机做准备；未连接机器人或发送关节命令。这里的实现是**只读与影子推理基础，并非已完成的实机控制器**。训练出的策略尚未完成实机物理验收，不能靠本文判断它能安全行走。

## 官方依据和下载

- [AimDK 1.1 快速开始](https://x2-aimdk.agibot.com/zh-cn/latest/quick_start/prerequisites.html)：SDK 必须与机器人固件版本匹配；Ubuntu 22.04、ROS 2 Humble；开发计算单元从 1.2.0 起须在容器中运行二开程序。
- [获取 SDK](https://x2-aimdk.agibot.com/zh-cn/latest/get_sdk/index.html)：RL Deploy 是单独扩展包。[仿真与强化学习](https://x2-aimdk.agibot.com/zh-cn/latest/sim_rl/index.html)明确两种示例只在仿真验证，电机 KP/KD 不可直接用于真机。
- [底层关节接口](https://x2-aimdk.agibot.com/zh-cn/latest/Interface/control_mod/joint_control.html)：关节命令没有超时或失效保护，停止下发不会自动进入安全状态；不能把本进程的异常退出或软件限幅当作安全系统。
- 官方扩展包 `https://x2-aimdk.agibot.com/downloads/aimdk-extra-x2_rl_deploy-v1.1.zip`；服务器 `/home/2025-Long.xh/HHD/data/supermimic/downloads/aimdk-1.1/x2_rl_deploy-v1.1.zip`，SHA-256 `cbf772de7875a70f85b4bd4f3097e763065d64fbebc0c811e65833a3c2339f17`。包内 `JointStateArray`、`JointCommandArray` 和 `motion_control_node.cc` 已核对；示例关节发布周期 2 ms。用户提供机器人版本 1.1.5；公开文档显示 SDK 1.1.0，尚不能确认与 1.1.5 的对应关系。
- 旗舰版 X2 Ultra 使用 v1.3.0 模型；Ultra-N 和 EDU 使用 v1.4.0，按[官方铭牌说明](https://github.com/AgibotTech/agibot_x2_urdf)核对。仓库训练模型不等于已核实的实机标定。

## 分层和当前代码

`deploy/core.py` 定义具名关节状态、明确配置的关节顺序和限位、状态时间检查、策略及目标接口、故障锁存和仅返回目标值的影子控制器。时间必须是同一个本地 monotonic 时钟的接收时间；订阅线程必须为每组数据记录独立接收时间，不用 ROS header 代替。状态不全、重复、故障码、过期、时间错位、非有限值、越限或目标步长超限均中止计算。限位和步长均需来自经验证的实际机器人配置；没有提供假定的真机默认值。故障锁存只影响影子计算，不会向机器人发送任何安全命令。

`scripts/deploy/observe_aimdk.py` 只订阅官网所列关节和 IMU 话题，输出发现的名称、错误码和 IMU frame，绝不发布命令。在版本匹配的 SDK 环境中 `python3 scripts/deploy/observe_aimdk.py --imu torso`；设备尚未连接，订阅 QoS、时间同步、关节和 IMU 标定均未实测。

`scripts/deploy/export_policy.py` 将已有检查点导出为 CPU TorchScript，输出模型契约和检查点 SHA-256；`deploy/torch_policy.py` 可加载该导出。输出应放数据目录，不能提交权重。示例：

```sh
SM_ROBOT=x2_ultra PYTHONPATH=src .venv/bin/python scripts/deploy/export_policy.py \
  --checkpoint /path/to/ckpt_0011500.pt --out /new/output/directory
```

服务器本次导出位于 `/home/2025-Long.xh/HHD/data/supermimic/logs/x2_deploy_foundation_20261030_r1/export/`，模型 SHA-256 `4424f231fc1896da098b2fca366e8522df6791596638b2b9257ba682520bbe4c`，源检查点 SHA-256 `ff1f91d33631530d9d7f90f9f0675359fa1bff18445df43449312b9cb85db34b`。CPU 上一组随机输入的导出前后最大误差为 0；不代表时序、精度跨平台或物理行为已验证。TorchScript 由 PyTorch 2.14.1 导出，API 已弃用，目标设备需要验证运行时兼容性，后续可评估替代格式。

影子内核的观测按训练配置构造：关节相对默认位置、关节速度、基座角速度、投影重力、上次动作，历史按**项优先**拼接；历史填满前不推理。输入假设 IMU 姿态为 body-to-world 的 wxyz、角速度已在策略基座坐标系；这两项必须在设备上核实并转换。未来目标由 `GoalProvider` 注入，不得以零向量代替。当前训练策略是 10 帧历史、10 帧未来参考加 9 维 anchor；锚点在当前仿真中依赖 MuJoCo 和机器人世界坐标。实机必须实现并验证相同的 FK、姿态、基座位姿/漂移及参考时序，才能构造正确目标。影子历史初始化行为也须与仿真对齐后才能比较首帧。

## 完成部署仍需

1. 获取目标机器人具体型号、固件匹配 SDK 和标定，核对关节名称、方向、零位、单位、限位、IMU 坐标及基座估计；设置真实经验证的关节限速、力矩及温度等保护。
2. 实现和测试 AimDK 数据聚合、时钟和目标提供器，与现有仿真逐帧对比观测、历史、锚点、未来窗口、动作及关节目标；验证追帧、延迟、丢包和切换。
3. 基于实测实现有明确控制权交接的独立实时发布层、独立硬件/系统级看门狗及经过厂商确认的故障处置；禁止与官方 MC 或其他控制器同时驱动。Python 影子内核不保证实时性，不能承担硬件安全。
4. 在服务器官方 RL 仿真和本项目仿真中完成连续键盘动作、多时相启停、长时站立及异常注入；当前策略已有脚部穿透、漂移等局限，尚未完成物理验收。之后才考虑受控实机测试。

未实现：硬件命令发布、实际控制权切换、真机安全看门狗、在线目标构造和完整实机部署。以上前置条件完成前不可接入电机输出。

2026-10-31 更新：已增加仅本机隔离的官方仿真策略及键盘接口；仿真提供骨盆真值，不等于实机状态估计。测试发现小跑停步距离较长、站立漂移及控制周期抖动，尚未通过实机部署验收。详情见 [官方仿真记录](X2_OFFICIAL_SIM_20261031.md)。

## 2026-10-31：机载推理和电脑键盘的准备

用户确认恢复五动作训练时共同使用的原始站姿及收步参考，数据为服务器 `logs/five_gait_stops_5h_20261030_r2/dataset`，继续使用该次训练保存的 `ckpt_0011500.pt` 导出策略；不使用后来的脚尖平齐候选。新参考的实验结果仍保留，不覆盖原数据。

目标连接为电脑浏览器键盘 → SSH 本地端口转发 → 机载开发计算单元 PC2 上的技能调度和策略推理 → 经验证的状态及控制接口。当前只在服务器上的隔离官方仿真中测试前半段，不存在可投入真机的电机发布器。键盘网页顺序处理请求，输入心跳中断 0.5 秒后选择站立参考：这只是技能指令超时，不是电机级安全看门狗。服务只绑定 127.0.0.1，切勿直接公开在网络上。

仿真实验入口 `scripts/deploy/run_official_trial.sh` 支持第六个参数 `keyboard:端口` 选择浏览器控制；第八个参数显式指定原数据。电脑侧可用 SSH 转发 `ssh -N -L 127.0.0.1:19181:127.0.0.1:19181 2025-Long-robot`，随后访问 `http://127.0.0.1:19181`。W 前进、S 后退、A/D 转向、Shift+W 小跑，松键、页面失焦或心跳丢失切换站立。页面必须保持焦点；浏览器模拟控制仅供仿真。测试时必须等待仿真状态就绪才能开始按键，监听端口已打开不代表策略已接收到状态。

`scripts/deploy/probe_target.py` 在目标机器只读打印系统架构、Python 和关键依赖可用性、AimDK 消息包路径，不发布机器人命令或启动控制节点。需在 PC2 的实际 SDK 环境中执行，并配合 `observe_aimdk.py` 核对关节与 IMU 消息；不要在 PC1 上运行二次开发程序。[官方资料](https://x2-aimdk.agibot.com/zh-cn/latest/about_agibot_X2/onboard_computer.html)列出 Ultra 开发单元通常为 Jetson Orin NX，但具体设备尚未核验。用户称设备 1.1.5 且已安装 SDK，SDK/固件匹配和消息格式仍需在该设备核实。

实机部署依然阻塞于可用且经过验证的机载骨盆状态估计、关节零位和 IMU 坐标、ARM64 推理依赖/性能、官方控制权切换、独立看门狗和故障交接，以及实际模型与仿真的参数差异。当前仿真用真值 odometry，不能用伪造或固定坐标替代；[官方文档](https://x2-aimdk.agibot.com/zh-cn/latest/Interface/control_mod/joint_control.html)指出底层关节命令无超时保护，现有仿真阻尼策略不能直接用于真机。已有收步偏差和较长制动距离也尚未通过真实安全或多条件验收。

原参考远程输入实验 `logs/keyboard_remote_preflight_20261031_r1/keyboard-original-r2/`：使用 `3a750a1` 源码与官方模拟器，电脑本地经 SSH 转发模拟浏览器 HTTP 心跳，依次发送五种技能并松开；控制记录 73.62 秒、3680 帧，各技能均有记录，零重置、无高度或倾斜终止、未来窗口重叠误差为零，末段为站立。测试并未操作真实键盘页面，也未覆盖各松键时相、异常输入或实机。首轮 `keyboard-original/` 在仿真状态就绪前过早发送前进和后退，记录缺失，不能记作五动作通过。随后 `7abc669` 加入状态就绪前的 HTTP 409 门槛，独立键盘单元测试通过；这项门槛仅属仿真行为。

独立 `logs/keyboard_ready_gate_20261031_r1/ready-check/` 使用 `7abc669` 验证启动门槛和超时：就绪前 104 次请求返回 409；就绪后发送小跑心跳约 3 秒，然后在小跑状态直接停止心跳，记录显示约 0.5 秒后选择站立并进入收步。30.51 秒无重置控制、无高度或倾斜终止、未来窗口重叠误差为零，最后为站立模式。此测试只验证仿真进程仍运行时的技能指令超时，未验证控制进程崩溃、电机命令超时、真实网络或安全接管。

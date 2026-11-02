# X2 定位到骨盆状态的只读影子推理（2026-10-31）

> 历史记录：用户已于 2026-11-02 决定放弃点云与 FAST-LIO 定位路线，当前改用 [500 Hz IMU 计算路线](X2_SHADOW_CONTACT_FUSION_20261102.md)。本文保留原始实验结果，不作为当前实施方案。

后续说明：本页保留首轮输入/输出统计。“目标角超机械范围”是 PD setpoint 的文件比较，不能单独作为策略故障结论；实测关节范围和硬件接口仍未通过验证。设备腰部 FK 与启动姿态的后续修正见 [启动输入差异处理](X2_STARTUP_ALIGNMENT_20261031.md)。

已完成一次真实传感器录包、服务器 CPU 离线定位、骨盆状态转换和模型输出检查。**严格输入检查未通过；诊断推理的 723 帧输出全部有限，但每帧都有目标超出训练模型关节范围，不能接电机。** 未创建电机发布器或控制客户端，未切换控制权、启动原厂运控、改时钟或修改机器人配置。

用户确认当前为吊挂保护、脚底接地，并要求保持静止。接地用于初始高度注册，没有独立接触或运动真值。仿真项目的模型关节范围也不是已校验的硬件安全限位。

## 模型与数据绑定

- 用户已选择的 11500 步 checkpoint：`ff1f91d33631530d9d7f90f9f0675359fa1bff18445df43449312b9cb85db34b`。
- 实际执行的 `policy.pt`：`4424f231fc1896da098b2fca366e8522df6791596638b2b9257ba682520bbe4c`。
- 参考：`jog_parallel_stance_20261031_r1/reference/dataset`；manifest SHA256：`40a2a31c1b84160f2ea6aa26548eb966f2e3c9ec23e10fa46ec1e9b2532b5b7d`。10 个 NPZ 均重新验证与 manifest 一致。
- 参考构造使用所选源码 `ac6c755` 的 `SkillGoal` 和 `MotionLibrary`；新增转换模块独立备份。通过 `SM_ROBOT=x2_ultra` 和明确的 profile 选择 31 关节维度。
- 使用服务器原有完整网格模型只做正运动学，不推进物理或渲染。profile SHA256：`9160f6411b531928dfa099fe7ec1c315e9382eb78abc9669ace2e41b37909fcb`；XML：`b3858223a36e7fb377628053f0016af8cd5c066a08bf022ea2fdc54c3147d60a`。与 PC2 已选 profile 的关节顺序、默认角、action scale、锚点和 tracked bodies 均相同；这不验证硬件零位、方向或外参。
- CPU 环境：Python 3.12.13、torch 2.14.1+cu130、NumPy 2.5.4、MuJoCo 3.16.0、SciPy 1.18.1；`CUDA_VISIBLE_DEVICES=''`。未占用 GPU。

## 采样和定位

PC2 专用目录：`/home/agi/lxh-supermimic/shadow_readonly_20261031_r1/`。服务器完整记录：

```
/home/2025-Long.xh/HHD/data/supermimic/logs/x2_shadow_readonly_20261031_r1
```

首轮 `capture` 使用深度 200 的状态队列，JSON/录包处理跟不上高频状态，产生约 400 ms 积压，头部约 600 ms。此包保留，未用于最终推理。状态订阅改为 BEST_EFFORT、KEEP_LAST(1)，避免记录器积累旧状态；雷达和积分用 IMU 仍按原传感器队列录制。没有更改原生发布器 QoS。该记录器不是完整 500 Hz 状态无损采集器，丢弃过时状态是明确设计。

最终 `capture_latest` 为 20 秒，18:12:02–18:12:22（机器人记录时间）：

- 201 帧点云、200 帧 raw 包、4007 个雷达 IMU、4 条静态 TF 消息。
- 四组关节各 4895–6280 个样本；torso/chest IMU 各 4247/3898 个。
- 各关节组与躯干 IMU 的 header→记录器接收延迟中位约 1.3–2.2 ms，P99 约 2.4–3.4 ms，最大 15.9 ms。这不是硬件采样延迟证明。
- 最终原始数据库 SHA256：`98e42813892519f0cb30dcfb232236447e0f73ddc015d8361ba00ca71659c773`，机器人与服务器一致。

新包再次经过原生 MSOP/点字段时间审计，再使用 `raw_sensor` 点云扫描起始时间；雷达 IMU 保留原生 host header。既有完整 IMU 硬件时间缺口仍存在，见 `X2_LIO_TIMING_AUDIT_20261031.md`。

服务器以 localhost-only、ROS domain 231、1 倍速、有限包 RELIABLE KEEP_ALL 回放。原生回调准确收到 200 个点云、4007 个 IMU，形成 200 组同步数据并发布/记录 197 个位姿；最后一帧覆盖完整，最大间隔约 100.01 ms。仅发布隔离的传感器回放话题，未连接机器人 DDS 域。

## 骨盆转换与观测构造

1. FAST-LIO 源码 `IMU_Processing.hpp` 用均值加速度初始化重力、保留初始旋转为单位矩阵，故 `camera_init` **不能直接当成 Z 轴朝上的世界坐标系**。使用首个位姿后 0.5 秒内的 100 个静止 IMU 样本和已输出姿态估计向上方向，再做固定重力对齐；初始化数据全部早于推理开始。原方向约 `[8.1998, 0.1622, 5.3962]`。没有优化时间偏移或假设硬件时钟已验证。
2. `T_PI(q)` 使用实测关节的骨盆→torso 正运动学，以及原厂 `torso_link→lidar_chest_front→lidar_imu_chest_front` 外参。前一段安装 TF 来自本会话此前的只读记录 `outputs/x2_sensor_readonly_20261031_162111/tf-odom.json`；后一段来自本次录包。腰部可动，不能用固定胸部→骨盆偏移。
3. 按 `T_WP = T_WI × inverse(T_PI(q))` 转换；用于此转换的关节按 LIO 位姿时间向过去选择，不拿当前腰角代替扫描时腰角。雷达 IMU 角速度旋转到骨盆后，减去腰部链条的角速度贡献。未使用雷达 IMU 恒单位 orientation 充当姿态测量。
4. 初始高度用原训练模型两脚碰撞网格的最低点和用户接地条件注册，保留原始脚部网格。点云地面平面仅作独立诊断，不回写注册、不改变输入。估计骨盆高度为 0.6643–0.6717 m；点云地面候选与脚部注册相差约 1.50 cm，拟合 RMS 7.09 mm、倾角 0.65°。这是几何一致性检查，不是绝对高度标定。转换后的重力方向与 native torso IMU 对比约差 2.55°，其原生 frame_id 含义尚未独立确认，不能把该 IMU 当作骨盆真值。
5. 推理时钟为原录包接收时间轴上的 50 Hz。仅选择已接收、非未来的关节和 IMU；状态 header 年龄上限 50 ms，组间偏差上限 30 ms。LIO 可用时间来自服务器 1 倍速回放回调映射到录包时间轴，含服务器计算耗时，**不是 PC2 在线延迟测量**。
6. LIO 保持最近可用的 10 Hz 骨盆位姿，不外推、不声称已完成 IMU 传播；年龄上限 150 ms。本包年龄约 40–140 ms，关节组间偏差最大 7.01 ms。静止时骨盆位置相对首帧最大变化约 2.88 cm。
7. 连续 10 个真实时间采样后才开始推理；观测为 term-major 的关节偏差、关节速度、骨盆角速度、投影重力、上一动作，990 维；站立参考和锚点共 729 维；输出 31 维。归一化由导出模型内的既有尺度处理，未额外归一化或裁剪。

策略从未执行，真实策略上一动作不存在。因此同时保存两种**反事实**动作历史：`feedback` 使用本分支此前未执行的输出，`zero` 始终使用零。两者都不能代替闭环实机历史。

## 输入与输出结果

最终脚本的严格检查 `shadow_strict_final/report.json` 拒绝了全部 732 个候选输入时刻，没有输出。以下实测角度在最终包中保持不变：

| 关节 | 实测 rad | 训练模型范围 rad |
| --- | ---: | ---: |
| left_shoulder_roll_joint | -0.120895 | [-0.061087, 2.98451] |
| right_shoulder_roll_joint | 0.084847 | [-2.98451, 0.061087] |
| left_wrist_pitch_joint | -0.573167 | [-0.523599, 0.523599] |

之后通过显式 `--diagnose-out-of-range` 做日志诊断，保留原角度、逐帧标记输入越界；未修改模型范围、硬件配置或裁剪数据。最终结果在 `shadow_diagnostic_final/`：

| 项目 | feedback | zero |
| --- | ---: | ---: |
| 输出帧数（14.44 秒） | 723 | 723 |
| 有限数值 | 全部 | 全部 |
| 存在超模型范围目标的帧数 | 723 | 723 |
| raw action 最大绝对值 | 6.9613 | 5.2626 |
| 换算目标最大绝对值 rad | 2.5637 | 2.5529 |
| 相邻日志目标最大变化 rad | 0.5164 | 0.00578 |

换算严格使用 `target = default + action_scale × action`，只写日志。两分支目标最大差约 1.302 rad；反馈分支的变化不能解释为机器人实际运动。主要越界涉及踝关节、腰部，部分帧也涉及肘部。

单帧对照（全部是反事实输入，不是机器人结果）：

- 保留实测本体观测，仅将定位锚点误差置零，输出仍有踝、腰和肘目标越界。
- 换为匹配参考站姿的本体观测、零上一动作和零锚点误差，该单帧输出没有超模型关节范围。
- 对实测输入的锚点 X 增加 2 cm，日志目标最大变化约 0.0049–0.0058 rad。

这些对照说明，不能仅靠清除定位漂移或锚点误差来解释并修复当前输出。应先只读核对实机零位/方向、模型限位及当前关节姿态与训练参考的差异；不能由单帧对照断定训练策略本身、状态转换或任何硬件标定已经合格。未改变参考、模型权重或训练方式。

## 验证、复现与保留记录

- 4 个几何/因果选择测试通过，含移动安装链变换反解、扣除腰部角速度、重力对齐、拒绝未来和过旧数据。
- 既有点云格式 4 项、raw 时间解析 3 项测试通过。
- `final-validation.json` 逐帧核对全部 723 帧维度、有限性、换算目标、接收与 header 因果关系、20 ms 时钟、反馈分支前一动作及零分支历史，均通过。**这些是计算合同检查，不是输入或执行安全验收。**
- `code/` 保留本轮脚本/测试，`source_selected/` 为 `ac6c755` 源码加新转换模块；`capture_latest/`、`audit_latest/`、`prepared_latest/`、`extracted/`、`replay_latest/`、`shadow_diagnostic_final/` 保留各阶段数据。
- `capture` 为积压诊断；`shadow_r1` 无输出；`shadow_diagnostic_r2` 因未显式选择 X2 导致维度断言退出，`r3` 因调用了旧版 `SkillGoal` 不支持的参数退出；修正选择和调用后 `r4` 及 `final` 完成。失败目录和日志保留，不作为验证结果。

最终推理复现命令（必须指定**新的输出目录**）：

```bash
cd /home/2025-Long.xh/HHD/data/supermimic/logs/x2_shadow_readonly_20261031_r1
CUDA_VISIBLE_DEVICES='' PYTHONPATH="$PWD/source_selected/src" \
  /home/2025-Long.xh/HHD/venvs/supermimic/bin/python shadow_x2_localization.py \
  --root "$PWD" \
  --profile /home/2025-Long.xh/HHD/data/supermimic/assets/robots/x2_ultra/profile.json \
  --dataset /home/2025-Long.xh/HHD/data/supermimic/logs/jog_parallel_stance_20261031_r1/reference/dataset \
  --policy /home/2025-Long.xh/HHD/data/supermimic/logs/x2_deploy_foundation_20261030_r1/export/policy.pt \
  --out NEW_OUTPUT_DIRECTORY --diagnose-out-of-range
```

本轮没有 PC2 端在线推理实时性、全频硬件时间同步、动态定位精度、硬件零位/方向/限位、闭环启动或稳定性验收。当前输出仅供诊断，不可下发。

# 站立及原地转向去除水平位置反馈试验（2026-11-04）

用户同意在 stand 和原地左右转时去除水平相对位移反馈，保留其他目标。当前用户已明确机器人由吊架或外部支撑托住、脚不承重。本次只完成软件与实际资产的无输出对照，未启动电机、切换模式或执行策略接管。

## 行为

`SkillGoal(..., zero_horizontal_anchor_modes=('stand','turn_left_inplace','turn_right_inplace'))` 在计算参考锚点与实际锚点的世界系位移后，将世界 X/Y 误差置零，再转换到实际锚点坐标系。高度误差、相对旋转、未来参考窗口、动作尺度、kp/kd 及关节裁剪配置不变。

不能简单对身体坐标的目标末尾两个分量清零：身体倾斜时，世界水平漂移会投影到身体 Z 中。本次世界系处理使三维相对位移整体不受世界水平位置漂移影响，同时保留垂直误差；因此倾斜时局部 X/Y 仍可含有合法的高度误差投影，不要求日志中两个数恒为零。

walk/backward/jog 保持原水平反馈。估计器仍照常积分并记录漂移，没有位置或速度重设，没有接入“整段静止”速度假设或足端接触假设。这是目标反馈对照，不是位置估计修复。

## 实际调用与开关

容器独立版本 `/home/agi/lxh-supermimic/releases/11500_zero_xy_20261104_r7`，父版本 r6 保留。r7 的 `start-keyboard.sh` 默认设置 `SM_ZERO_HORIZONTAL_ANCHOR_FEEDBACK=1`；runner 将其转成 worker 的 `--zero-horizontal-anchor-feedback`，worker 在构造实际 SkillGoal 时启用三个模式，并在本次 trial 保存 `anchor-feedback.json`。

`SM_ZERO_HORIZONTAL_ANCHOR_FEEDBACK=0` 可在同一 r7 中关闭对照。其他旧入口默认关闭此新选项，不全局覆盖 SDK 或共享脚本。实际 trial 虽仍复制 LIB 的 early_handover，干预在 SkillGoal 中执行，避免新 early_handover 被旧复制步骤覆盖。原有 `zero_stand_anchor_xy` 是旧的身体坐标干预，本次保持 false，不混用。

现场人工入口：

```bash
ssh -t x2-robot 'podman exec -it lxh bash /home/agi/lxh-supermimic/releases/11500_zero_xy_20261104_r7/start-keyboard.sh'
```

入口仍要求脚承重、防跌落、有人掌握急停及初始校准物理静止。**当前脚不承重，不满足入口的 feet-loaded 确认；不能将当前吊挂状态确认成已承重。** 本次未移除或绕过现场条件确认。软件对照可以在无电机输出下完成，不能证明落地站立、原地转向不迈步或吊挂控制效果。

## 验证

16 项目标/worker/原有干预回归通过，包括新选项开启/关闭时预热和对端门槛、倾斜姿态下世界 XY 漂移被完整隔离、垂直误差和转向特征保留、模式切换恢复行走反馈。使用专用推理源码副本和既有 mjlab math；未安装依赖、未运行本地物理仿真。直接从训练源码导入时缺少完整 mjlab，保留此检查限制，未为测试修改生产 env 初始化。

容器实际模型、参考、11500 NumPy 权重检查：`reports/r7-anchor-intervention.json`，可复核脚本 `reports/check-anchor.py`。使用真实静止初始化后的记录姿态/编码器与固定观测，人工向估计位置加入 `[11,-7,0] m`：

- stand、turn_left_inplace、turn_right_inplace：完整目标变化与策略动作变化均为 0。
- walk、backward、jog：开启新选项与原逻辑完全一致，仍受位移反馈影响。
- 所有模式的未来参考窗口与相对旋转特征逐元素不变；世界垂直误差保持原值。
- 策略原固定输入最大误差 4.992e-7，小于既有 1e-4 门槛。

首次资产检查忘记设置 SM_ROBOT/SM_ROBOT_SPEC，触发参考维度断言；显式指定 X2 Ultra profile 后通过，没有修改模型或参考数据。

该检查没有 ROS 状态/关节发布器，也没有物理仿真步进。不证明动作稳定性、完整运行时序或位置精度；r6 的 IMU 漂移及延迟局限仍存在。切换到行走时原位置反馈重新生效，也尚未完成实机验证。

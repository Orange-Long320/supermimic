# 所有键盘动作去除水平位移反馈（2026-11-04）

用户要求将原先仅 stand/原地转向的水平相对位移干预扩展到所有动作。当前容器版本 `/home/agi/lxh-supermimic/releases/11500_all_zero_xy_20261104_r9`，父版本 r8 保留。

## 行为与实际入口

stand、walk、backward、jog、turn_left_inplace、turn_right_inplace 全部在世界坐标系将水平锚点位移误差置零，再转回身体坐标。世界垂直误差、相对朝向、未来动作参考、原始观测历史、kp/kd、动作缩放及裁剪配置保留。估计器照常积分和记录；本次不修复或重置估计位置，不启用静止/足端速度约束。

新 worker 参数 `--zero-horizontal-anchor-all-modes`，runner 环境选项 `SM_ZERO_HORIZONTAL_ANCHOR_FEEDBACK=2`，r9 包装入口默认 2。旧值 0 为保留位置反馈，1 为仅 stand/原地左右转去除反馈；r7/r8 原文件未修改，当前已运行的 trial 源码也未修改。

现场人工入口：

```bash
ssh -t x2-robot 'podman exec -it lxh bash /home/agi/lxh-supermimic/releases/11500_all_zero_xy_20261104_r9/start-keyboard.sh'
```

仍保留原入口对脚承重、初始物理静止、防跌落和急停的条件确认。脚悬空时不能确认 feet-loaded 条件。旧程序需要由现场操作员结束后才可启动新入口；没有自动终止或重启机器人程序。

## 验证与限制

19 项目标、worker 预热及既有干预回归测试通过。覆盖所有六模式水平漂移隔离、倾斜姿态下保留垂直误差、保留未来参考和朝向、旧值 0/1 与新值 2 的开关及预热门槛。shell 语法检查通过。

在 PC2 lxh 中，以实际 X2 Ultra 模型、参考数据、NumPy 11500 策略和记录姿态/编码器进行无输出对照。人为向估计位置加入 `[11,-7,0] m`，六种模式完整目标与策略动作变化均为 0；所有未来参考与旋转特征逐元素不变，世界垂直误差保持原值。原固定输入策略误差仍为 4.992e-7。报告 `reports/r9-all-anchor-intervention.json`，复核脚本 `reports/check-all-anchor.py`。

没有 ROS 状态/关节发布、物理仿真步进、电机启动、模式切换或策略接管。更改只证明水平漂移不再通过锚点位移目标影响六种模式，不证明行走、站立或转向稳定性。行走仍由动作参考和其中的运动目标驱动，去除位移反馈后的实际步态效果需另行验证。

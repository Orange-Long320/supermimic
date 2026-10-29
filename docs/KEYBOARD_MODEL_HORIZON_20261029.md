# 键盘速度控制：模型预测范围诊断与修正

## 已验证根因

2026-10-29，在2025-Long-robot GPU1固定评测1000步权重（SHA256 899ca76c2cc44e3ca793ffbe613b5eb2da3c015f4e59d8f436dc48de88ed3843）。运行 logs/keyboard_model_diagnosis_20261029_r1，PID1924514、会话sm-keyboard-model-diagnosis；diagnosis.json保留全部指标，launch.json保留命令。3300个物理步、8环境、66秒，无失败；无优化器更新。同一初始物理状态、同一环境参数及同一实际动作序列，逐一比较模型1/8/32步预测与MuJoCo结果；不是比较不同策略。

实际配置wm_trainer.window=8、policy_trainer.horizon=32。前进段1/8/32步vx预测RMSE分别0.0046/0.0151/0.1551 m/s，32步vy误差0.2235 m/s、yaw误差0.3974 rad/s。小跑32步vx/yaw误差0.1903 m/s、0.3374 rad/s；左转yaw预测均值0.2887，物理0.2192 rad/s。模型的低短期NSE和想象存活不证明32步速度梯度可靠。这是已测到的训练配置/模型误差问题；尚不能说是唯一原因。

本轮持续教师监督正常（1200步有效率约85.3%，alignment非零）；此前监督中断已修复。500步五动作底座已完成短片段物理执行；因此现有证据不支持简单归咎于底座完全不会这些动作。此证据也不证明更长片段或任意切换合格。

## 修正与验证安排

- keyboard_supervision.install_velocity_horizon将整次速度更新限制到8步，包括actor归一化与critic/GAE；跟踪更新保留32步。入口提供显式--velocity-horizon（默认8，32可重现旧配置）。旧运行快照/进程保持原样。
- refine_keyboard_physical.py提供有界真实物理修正，只训练速度编码器，冻结motion encoder、decoder和世界模型。正/负/零速度、左右转、小跑均衡采样，actor与critic都输入实际指令，加入停止关节运动约束与同历史反事实教师约束。独立评测通过前所有输出均为candidate。
- 不能用这次模型误差诊断声称修正后策略已成功。必须同66秒连续协议、多种子测速度/方向/停止/失败，再验证手臂停止。物理修正是独立阶段，不能直接恢复旧速度优化器。

正在进行的旧一小时任务仍按原预算结束；不把其结束结果与新修正混淆。其他聊天正在编辑的UniLab文件不属于本次提交。

## 物理修正容量检查

源码快照fbca550。logs/keyboard_physical_correction_20261029_r1/capacity，GPU1、PID1927405，会话sm-keyboard-physical-correction；256环境40轮×32步已完成，训练日志分开记录terminated和timeout，均为0。仅说明更新流程与资源可用，不能作为闭环成功；下一阶段从原1000步源独立启动，避免把容量随机采样指标混成验收。

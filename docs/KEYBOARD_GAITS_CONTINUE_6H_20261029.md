# 键盘五动作完成当前训练后再续训六小时

用户授权：“训练完以后再训练六个小时”。依赖运行是 `logs/keyboard_gaits_fixed1h_20261029_r2`；先等待其一小时训练、66秒全指令与42秒停止评测全部正常执行结束，再续训六小时。物理指标保留，不把评测执行成功等同于整套策略验收通过。

续训目录 `logs/keyboard_gaits_continue6h_20261029_r1`，独立键盘源码快照 `logs/keyboard_gaits_continue6h_20261029_source`。从父运行最终实际保存的ckpt_latest.pt复制成不可变resume-source.pt，核对父rank0记录的SHA256及双卡最终参数同步，保留父评测和来源记录。恢复模型、所有优化器、速度critic、步数与更新数；回放缓冲按原入口重新采集，不声称恢复仿真状态或回放。不是重新加载旧quiet-stop初始权重。

沿用GPU0/2，每卡8192环境，其中跟踪/速度各4096；wm batch2048、policy总batch2048（各1024），域随机化及原噪声开启，持续配对监督与零速停止代价保持。模型结构、数据及更新配方相同，复用已验证容量与父完整双卡训练证据，不重复初始化或重置优化器。

队列入口scripts/experiments/continue_keyboard_gaits.py以独立tmux进程等待父运行完成，检查GPU空闲后启动；不停止其他进程。父任务失败/取消或等待超过3小时即保留错误退出，不使用旧检查点悄悄启动。续训本身计时六小时，约每20步检查一次时间，可能略超；每100步保存检查点。六小时结束后自动执行66秒五动作连续控制和42秒持续停止/手臂运动回归。

当前尚未开始六小时段；以controller.json的waiting_for_parent/train/failed等状态和rank记录为准。预计一小时段19:40左右结束，评测后六小时段约19:45开始，次日01:45左右结束；具体以实际启动时间为准。

## 已实际入队

续训入口提交e7cd5a0。服务器会话sm-keyboard-continue6h、控制器PID1926621，已核验controller.json状态waiting_for_parent、requested_train_hours=6.0、GPU[0,2]，尚未启动六小时训练。launch.json保留完整argv、解释器、源版本及入口SHA。当前一小时段未中断、未修改运行快照。六小时入口、监督修复与两项评测均在独立源码快照内，父最终权重尚未产生，SHA与实际起始步数将在接续时自动记录。

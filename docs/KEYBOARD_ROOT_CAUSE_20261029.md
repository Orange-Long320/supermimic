# 键盘五动作响应弱：已确认缺陷与修复边界

## 证据

原运行 `logs/keyboard_gaits30m_20261029_r2` 不修改，保留其30分钟预算。400步固定权重 SHA256 为 `67675ebaf49f1f93a717d2ccfec981b82f1c0b60664994e5ad70480a9153b996`。

原训练速度编码器的唯一参考对齐项来自速度环境回放的 `command_reference`。速度环境在片段结束后继续物理运行，frame停在末帧，reference_exhausted阻断参考模式。4096个速度环境并不等于4096个有教师的样本；速度回放可完全没有教师。TensorBoard正式训练560/580/600步的alignment为0，而sim_linear_error约0.27m/s、sim_yaw_error约0.40rad/s。更早的alignment非零，因此不是从未启用对齐。

服务器逐帧检查五段平滑指令的合法率：后退75.98%、小跑52.67%、左转100%、右转100%、步行73.53%。部分前进/小跑标签超出速度范围或侧向容差，但不足以解释全部监督消失。没有放宽容差把不匹配标签强行作为真值。

## 已修复的训练缺陷

`keyboard_supervision.py` 在速度更新时从独立的动作跟踪回放采样教师历史，用同一历史的motion latent监督velocity latent；保留原速度奖励、停止反事实重标记及部署输入。动作跟踪环境在片尾复位，监督不再依赖速度环境过期的标记。

`train_keyboard_gaits.py` 默认安装该修复；`--no-paired-supervision`可复现旧行为。需将keyboard_supervision.py与入口放在同一目录。原服务器运行快照未改，未热改运行中进程。

回归入口 `check_keyboard_supervision.py`：32个真实仿真环境采集32步，将速度回放参考标记全部清零。相同条件原对齐损失0，新对齐损失0.053699，教师有效率81.25%，梯度有限且编码器参数实际更新。运行目录 `logs/keyboard_rootfix_check_20261029_r2`，GPU3。该检查证明监督缺口修复，不证明策略已收敛。

## 干预试验及未解决问题

`bootstrap_keyboard_gaits.py` 在GPU1用五段tracking策略生成26340组历史/指令/latent/action配对样本，独立2000次小批次更新，仅改变速度编码器。其他policy参数逐项torch.equal确认不变。固定源400步，seed17；held-out随机行latent MSE 0.074317→0.000299，action MSE 0.175899→0.000476。随机行分割只检查拟合，不是跨动作或闭环泛化。

候选 `logs/keyboard_rootfix_bootstrap_20261029_r1/candidate.pt` 只用于评测；删除了不匹配的velocity optimizer状态，明确不可直接resume。原权重和结果未覆盖。

66秒连续指令、8环境、seed0实际评测结果：8/8无失败或重置。去掉各阶段首秒后，前进指令0.3，平均vx0.19437；小跑0.6，vx0.26250；后退-0.2，vx仍+0.22228；左转+0.4，yaw+0.05642；右转-0.4，yaw-0.28088。最后零指令段vx+0.17010m/s；全体停止阶段各环境平均平面速度0.209–0.281m/s。不得把此候选称为修复完成或可用键盘策略。

这说明有效教师监督能恢复运动，但不足以建立正确的全指令控制。五段动作教师未覆盖静止保持及连续切换；接近零速度的动作帧也不能自动作为稳定站姿教师。完整成功条件仍未达到。下一阶段需在该底座校准停止教师、加入静止/运动切换监督，再在相同66秒协议下验收；不能把拟合损失下降当作可部署结果，也不能只延长原训练。

## 运行与备份

bootstrap会话 `sm-keyboard-rootfix-bootstrap`、评测 `sm-keyboard-rootfix-eval`、回归 `sm-keyboard-rootfix-check2`，均为有界任务。入口与日志保留在上述服务器目录。源码路径位于本仓库scripts/experiments；原键盘工作树未修改。

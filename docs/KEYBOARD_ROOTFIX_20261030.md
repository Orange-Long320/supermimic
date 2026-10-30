# 运动后停止漂移与前进退化：因果排查

用户要求从根因修复、保留前进/后退/小跑/双向转向，并最终交付键盘控制策略。未通过完整物理验收的权重仍为 candidate，不延长旧配方代替原因定位。

起点源码5276490，工作区干净。已读README与前轮记录，复核上一轮precision1200权重SHA256 `e6f92efeb6d7f87d37ee3311a7d1a28a8989eb41ca2f772bfbd291ad7d64ddb8`：冷启动/20秒保持通过，但66秒前进约0.168m/s及运动后停止漂移失败。旧移动核心固定不代表实际动作固定：新增velocity_feedback_weight仍参与移动编码器。

静态核查确认移动先验只在速度已经接近目标时保护原动作，起步、换挡、超限状态不受该先验保护；末段日志prior_valid_fraction约0.06–0.067（全部样本中占比，移动约一半）。这是保护范围缺口，尚不证明它是唯一原因。停止anchor在命令范数降到0.1以下前最后一帧冻结，而停止奖励到近零才开启；该时序与后停止漂移的因果关系待测。

已按用户许可使用两名GPT-6.1 Terra medium只读并行审计，没有让其修改源码或启动实验。主任务在GPU2执行同权重、seed0、8环境、66秒对照：完整反馈、关闭移动反馈、关闭站立反馈。仅诊断关闭参数不作为部署或通过验收的方案。

服务器已核验amax-Super-Server、4×RTX3090、专用环境canonical路径及HHD真实挂载点。GPU0/1/3被其他用户占用，未干扰。诊断入口`diagnose_control_feedback.py`启动前再次确认GPU2空闲，tmux `sm-rootcause-ablation-1030`、controller PID2179124、首个eval PID2179139。独立输出`logs/keyboard_rootcause_ablation_20261030_r1`，入口命令、权重/评测SHA、子进程及结果写diagnosis.json，原隔离源码不改。当前运行中，未宣称任何修复完成。

# UniLab 完整训练与 X2 接入验证（2026-10-29）

用户随后明确目标是 **UniLab + SuperMimic 与 mjlab + SuperMimic**，保持学习算法不变。本轮FlashSAC接入偏离这个目标，以下吞吐及策略结果不能用于判断用户所问两种组合谁更快；正确的同算法适配与对照尚未完成。

已通过官方 G1 CLI 和独立 X2 包装，实际执行完整 FlashSAC 的采集、回放、学习和检查点保存。此前独立 mjbatch 内核短测不能代表 UniLab；本记录取代其对框架速度的推断。

## 已验证结果

服务器同一张 GPU3（RTX3090），256环境、batch512、replay128、预热16轮、正式320轮、每轮4次更新。保留官方默认 learner compile、whole-cycle CUDA graph、BF16 AMP；只限制编译线程为2。CPU采集池指定16个CPU编号8–23，没有测满整台双路Xeon Silver4216的极限性能。

| 任务 / 仿真后端 | 正式循环耗时 | 尾100轮环境转移/秒 | 含初始化和编译的框架耗时 | 结果 |
| --- | --- | --- | --- | --- |
| 官方G1 / CPU MuJoCo | 19.162s | 6483.81 | 612.809s | 320轮完成，检查点已保存 |
| 官方G1 / GPU Warp | 16.323s | 6917.18 | 114.505s | 320轮完成，检查点已保存 |
| 项目X2 / CPU MuJoCo | 33.790s | 3608.74 | 363.936s | 320轮完成，检查点已保存 |
| 项目X2 / GPU Warp | 17.733s | 6511.60 | 93.005s | 320轮完成，检查点已保存 |

每组含预热共86,016条环境转移。四组均退出0。首次编译和后续运行的缓存条件不同，因此不能用含启动耗时直接比较后端快慢。

X2配置逐字段比较，仅有training.sim_backend、training.log_dir、training.play_render_mode三处不同；播放关闭，因此最后一项不参与训练。X2两端collector_metrics_interval均显式32。X2 GPU稳定吞吐约为指定CPU配置的1.804倍；这不是UniLab相对mjlab的倍数。

G1两端沿用官方owner的不同collector_metrics_interval（CPU32、GPU100），只用于证明两条官方流程可运行，不作为严格后端性能对照。

CPU X2检查点确认actor/温度优化器各640步、critic优化器1280步，确实执行学习更新。run_summary的runtime_manifest记录了采集/回放链路、BF16、actor与critic CUDA graph，以及update_cycle范围。

## X2 动作评测

使用同一CPU MuJoCo后端、同一原始动作，分别加载两份新检查点。从第0帧确定性执行，禁用观测噪声、重置扰动和自动重置，首个非timeout失败立即停止。先通过官方训练契约与actor语义校验，再明确记录评测协议变更。原动作6574帧、50Hz、131.46秒。

| 训练后端 | 首次失败 | 终止项 | 完整片段 |
| --- | --- | --- | --- |
| CPU | 0.26s，第13个控制步 | ee_body_pos | 未通过 |
| GPU | 0.24s，第12个控制步 | ee_body_pos | 未通过 |

这里是末端跟踪误差终止，不能直接称为已经跌倒。两份短训策略都没有完成动作；320轮只证明训练流程，不证明收敛。评测也不替代本项目严格的关节、穿透和稳定性验收。没有证据说明UniLab比现有SuperMimic更快达到相同动作质量。

## 方法、版本与数据

- UniLab源码 https://github.com/unilabsim/UniLab ，固定 `fa577b265fcc185ee5615e0abe7c06da067aad35`（1.5.3）；原代码未修改。源码存于服务器 `vendor/unilab`，归档source.tar.gz及锁定版本导出的requirements保留实验目录。
- 新独立环境 `/home/2025-Long.xh/HHD/venvs/supermimic-unilab`，vendor/unilab/.venv链接至此。Python3.12.13、Torch2.14.0+cu130、MuJoCo/MuJoCo Warp3.11.0、mjbatch-uni0.2.5、Warp1.16.0、UniSim2.2.3。主环境未改动。
- 依赖按上游uv.lock导出，通用包使用清华源；CUDA大依赖优先复用版本完全相同的本项目下载缓存。中途只中断本轮安装PID1921341，没有停止训练。Torch Linux cp312 x86_64轮子本地下载后上传，SHA256 `9c5693d3ca91743202034094ad3b3df20b5eb1373d5e0bc8a5d976be12c35302` 与上游锁一致。
- 上游明确exclude torchvision；uv pip check因此提示rsl-rl-lib的torchvision依赖缺失。没有擅自添加锁外版本；本轮FlashSAC实际运行通过。额外build工具uv-build0.12.24符合上游build-system范围。
- 官方G1网格、纹理与dance1_subject2_part动作从Hugging Face本地下载后上传；64文件各有来源与SHA256，清单位于专用资产目录manifest.json。训练全程HF_HUB_OFFLINE=1，TensorBoard本地记录，未使用在线日志。
- X2使用既有LAFAN1 dance1_subject2.npz，源SHA256 `046f4fcbf754366e3c64211821ea6138ed8ffa6b4c986ec9489b9aa09447fc1a`。注册独立X2UltraMotionTracking，组合官方通用tracking配置后替换31轴、root/anchor/末端、动作scale和传感器；通过官方train_offpolicy.main运行完整训练。这是实验接入，不是官方原生X2支持，也不是SuperMimic算法复现。
- 新数据显式按身体名字重排，补world及经无关节祖先检查的静态terrain行，动态机器人身体缺失则失败。转换NPZ SHA256 `69f5cc4da836e2085ad0694fb8802a451d2ec8c91583f36389241c8057232aae`。
- 场景沿用项目原网格、碰撞、质量与执行器参数，只增非动力学传感器和站姿keyframe。XML补写17位物理参数，并逆变换mesh编译时的质心/主轴变换；指定物理数组重载误差最大约3.4e-15。场景SHA256 `0c1fe840b648844a03ec2c3182da0c9da2c2b456c9d851c0d9370a454f433978`。不同MuJoCo版本的动力学数值等价仍未认证。

## 运行与回档

服务器根目录 `/home/2025-Long.xh/HHD/data/supermimic/logs/unilab_full_20261029_r1/`。

- `g1-cpu`、`g1-gpu`、`x2-cpu`、`x2-gpu`：各有controller-as-run、PID、起止UTC、控制台日志、退出码，以及train/run_config.json、run_summary.json、TensorBoard与model_320.pt。
- tmux会话依次为sm-unilab-g1-cpu-20261029、sm-unilab-g1-gpu-20261029、sm-unilab-x2-cpu-20261029、sm-unilab-x2-gpu-20261029。GPU3串行执行；未停止其他聊天在GPU0/1/2上的进程。
- `prepared-x2-r3`保存最终场景、动作与来源映射。前两次prepare失败日志保留：先修复mesh变换重复应用，再明确处理静态terrain；没有放宽物理校验。
- `eval-x2-cpu`、`eval-x2-gpu`保存评测配置、逐步数据与summary；CPU/GPU训练检查点SHA256分别为 `167bff40f39c0a9d875f06174075e2de3fdc20e9b6b0498321b29b280c7c5680`、`1069167eb2a8c0b425b062fb011f8148e42c19ab8d7990e810cb6db84e7132cc`。
- 入口位于scripts/experiments/unilab_x2及run_unilab_official.sh、run_unilab_x2.sh。启动前源码回档点10e25c7、991b121、72908dc、d732cb6、89d7295、315496f的相关bundle保留服务器实验目录。服务器部署新增独立vendor代码、环境与实验目录，未把本轮X2任务接入现有正式训练。

## 与现有项目吞吐的补测

同卡GPU3、同一X2动作、256环境、两个batch512，先试默认编译配置；在首次策略反向传播等待约601秒后主动中断，退出130，未取得稳定吞吐。记录为中断，不视为程序故障或已完成训练。

再使用项目近期已验证的不编译配置运行SuperMimic，达到100步后为结束吞吐短测而主动中断，退出130。60至100步耗时108.258秒，约94.59环境转移/秒。80至100步日志记录20步中仿真和采集0.6秒、世界模型9.5秒、策略更新44.0秒；日志计时有0.1秒舍入。以实测总耗时54.18秒计算，仿真和采集约占1.1%。在本配置且其余工作耗时不变的假设下，即使采集耗时降为零，理论总加速上限也约1.01倍。该估算不能外推到8192环境或其他训练设置。

用户明确要比较保持SuperMimic算法的两种框架组合；尚未完成UniLab环境接入SuperMimic及同算法直接对照，不能断言哪个框架实际更快。当前固定UniSim 2.2.3的mjwarp后端要求MuJoCo Warp 3.11，现有项目使用3.16，接入需要处理版本和环境接口差异，不能直接切换。实验日志和profile均保留在本节服务器目录。

这两套方案学习算法、网络、观测、奖励、更新比和日志检查开销不同，依赖版本也不同；即使吞吐有差距，也不能直接归因于mjlab框架或解释为同等质量收敛加速。

# X2 Ultra 本项目实验记录

## 2026-10-26：环境准备与首次短训

- 目标：LAFAN1 → X2 Ultra 重定向 → 单卡 smoke → 小规模完整更新及显存测量。
- 参考源码：`https://github.com/tianshuwu/supermimic-retargeted.git`，提交 `ff69a0e5305126eb5b80b323bcd815805bc3df4e`。本项目 README 和 AGENTS 保留；原 README 存于 `docs/upstream/README.md`。
- 服务器：`2025-Long-robot`，Ubuntu 22.04.5，x86_64，220 GiB RAM，4×RTX 3090 24 GiB，驱动 `580.178.04`，驱动报告支持 CUDA 13.0。
- 初次检查 GPU 0、1 被其他用户的 openpi 任务占用，GPU 2、3 无计算进程。首轮选择 GPU 2，启动前再次检查。
- SSD 剩余约 229 GiB；HHD 实际路径 `/mnt/local_DATA_HHD/2025-Long.xh`，所在文件系统剩余约 3.2 TiB。这些是检查时快照。
- 已创建 README 规划的专用源码、数据目录及两个虚拟环境。Python `3.12.13` 来自官方 python-build-standalone `20260602` Linux x86_64 包，运行时放在数据根目录 `runtime/python312`。
- 两套环境已安装并通过 `uv pip check`。主环境 PyTorch `2.14.1+cu130`、Warp `1.18.0`、MuJoCo `3.16.0`、mjlab `1.6.0`；SOMA 按其独立锁文件使用 Warp `1.16.0`。CUDA 设备查询正常。通用依赖由锁文件导出后从清华源安装；PyTorch CUDA 和主环境 Warp 从专用源本地下载后上传。安装时 `uv --no-config` 避免项目内海外 index 配置覆盖镜像。
- 原锁文件保持不变。mjlab 从固定源码构建本地 wheel 后安装；实际依赖快照存于服务器 `provenance/{main,soma}-freeze-20261026.txt`。
- X2 模型已构建，31 个受控关节，`nq=38`、`nv=37`。`tests/test_x2_wrist_tracking.py` 的 3 项测试通过。
- 40 段 LAFAN1 动作完成骨架对齐，最大非根节点世界坐标位置误差 `4.547473508864641e-13 cm`。GPU 2 正在执行完整重定向。
- 已启动 `tmux` 会话 `sm-first-lafan-20261026`，运行 `scripts/experiments/first_lafan.sh x2_lafan_first_20261026 2 /home/2025-Long.xh/HHD/data/supermimic/logs/retarget-x2-20261026.exit`。依次等待重定向成功、转换全部40段数据、执行128和512环境各100步短训；失败即停。
- 两次短训使用 smoke 预设、`anchor_obs=true`、replay容量256、世界模型batch256×2、策略batch128、展开32步、关闭PyTorch编译。512环境试跑仅扩大环境数，不能代表默认4096环境或更大策略batch的显存需求。
- 流程日志：`logs/x2_lafan_first_20261026/`；每轮完整日志 `train-{128,512}.log`，显存记录 `profile-{128,512}.json`，检查点分别位于 `logs/x2_lafan_first_20261026_{128,512}/`。这些是输出位置，尚未完成的阶段不能视为已产出。
- 重定向会话 `sm-retarget-20261026`；日志 `logs/retarget-x2-20261026.log`。首次运行包含Warp内核编译时间。

服务器完整安装日志：`/home/2025-Long.xh/HHD/data/supermimic/logs/install-*.log`；同名 `.exit` 文件记录已结束任务的退出码。

外部源码版本见 `configs/upstreams.json`，mjlab 固定为 `8bb2bc9e187797689ec82c78c35019afd2fd0376`。LAFAN1 固定为 Ubisoft `94084601bacdf9cc3764b5c73daaeccae6035fac`。X2 资产与 LAFAN1 ZIP 从对应仓库的 Git LFS 下载，逐文件验证大小与 SHA256，不使用 LFS 指针作为模型或数据。

GPU 0、1 上的其他用户任务未触碰。长训尚未启动，等待短训与显存测量结果。上游 `docs/results/`、`docs/EXPERIMENTS.md` 等仅为参考历史记录，不是本项目产物。

## 2026-10-26 21:21：重定向完成，转换进行中

- 服务器核验：重定向日志报告全部 40 段成功，耗时 `00:25:42`，`logs/retarget-x2-20261026.exit` 为 `0`。
- 原控制流程已自动进入 `prepare_lafan.py`，8 个工作进程正在转换，主进程 PID `3801527`；审计输出位置为 `logs/x2_lafan_first_20261026/preparation.json`，尚未确认转换完成。
- `sm-first-lafan-20261026` 会话保持运行；两轮短训及正式训练仍未启动。没有重复启动已有任务。
- 用户已授权短训成功后按实测显存继续正式训练及完整片段评测；每 10 分钟核验一次，阶段推进、失败或需要用户处理时通知。

## 2026-10-26 21:31–21:35：短训通过，扩大单卡测量

- 数据转换完成：40 段、0 拒绝、总计约 2.4506 小时；审计见首轮 `preparation.json`。控制流程退出码为 0。
- 128/512 环境均完成 100 步、24 次更新，检查点可读取，世界模型、策略及 critic 参数未发现非有限值。初始未更新时的 `nan` 指标在实际更新后变为有限值，不据此声称收敛。
- 128 环境：PyTorch 峰值 allocated/reserved 为 1332.65/1450 MiB，设备采样峰值 1941 MiB；512 环境分别为 1410.17/1528 MiB、2115 MiB。设备每秒采样可能遗漏瞬时峰值。
- 下一步为正式训练前容量测量：1024 环境、replay 512、世界模型 batch 1024×10、策略 batch 1024、展开 32，开启默认编译，100 步/最多 1 小时。此前 smoke 关闭编译且 batch 较小，不能将其峰值直接当作正式配置峰值。
- 新入口 `scripts/experiments/capacity_lafan.sh`；GPU 2；会话 `sm-capacity-20261026-r2`，运行目录 `logs/x2_lafan_capacity_20261026_r2/`，训练输出 `logs/x2_lafan_capacity_20261026_r2_train/`。环境、配置、完整命令和启动脚本随运行保存。
- 两次启动前检查失败已保留：初始目录的环境记录命令引号错误；r1 的测量入口用于配置预览时预先创建 profile 文件，后续按防覆盖规则退出。已修正为普通训练入口预览并使用 r2 独立目录；前两次均未进入训练。
- 正式长训尚未启动，待扩大配置的实际编译、更新与峰值测量结果后选择参数。

## 2026-10-26 21:53–21:55：容量测量通过，正式训练等待 GPU

- r2 完成 100 步、24 次更新，退出码 0，耗时 878.81 秒（含首次编译）。检查点可读取，世界模型、策略、critic 参数均有限；最终 nse1/nse8 为 0.2121/1.3948，尚不代表动作跟踪成功。
- PyTorch 峰值 allocated/reserved 为 4919.12/5272 MiB。设备采样峰值 19124 MiB 包含他人进程：后续核查 GPU 2 已被 `2025-Chen.g` 的计算任务占用，故不将设备峰值和吞吐当作独占条件下的本项目指标。
- 已创建正式训练等待会话 `sm-formal-20261026`，入口 `scripts/experiments/formal_lafan.sh x2_lafan_formal_20261026 2`。运行记录目录 `logs/x2_lafan_formal_20261026/`；训练产物位置 `logs/x2_lafan_formal_20261026_train/`。等待 GPU 2 无计算进程后自动启动；当前不能声称训练已开始。
- 首轮正式实验从头训练，沿用已测 1024 环境、replay 512、WM batch 1024×10、策略 batch 1024及默认编译；总步数上限 320000，首轮时间预算 3 小时，每100步记录、每1000步检查点。时间预算结束不等于完成320000步或收敛，需依据实际检查点与完整片段评测决定后续续训。
- 所有 GPU 当前都有其他用户计算任务，未停止、修改或抢占其进程。等待流程每60秒检查GPU 2，无需用户操作。

## 2026-10-26 22:32：正式训练改为定时启动

- 用户要求在预计 GPU 释放时间附近启动，已设为北京时间 **2026-10-27 02:30**（epoch `1793039400`），到时检查 GPU 2，无计算进程后才启动；仍占用则继续等待。
- 已确认旧会话尚未生成 `train.pid`，仅取消本项目旧等待会话 `sm-formal-20261026`，保留原输出目录。
- 新会话 `sm-formal-scheduled-20261027`；命令 `bash scripts/experiments/formal_lafan.sh x2_lafan_formal_20261027_0230 2 1793039400`；记录目录 `logs/x2_lafan_formal_20261027_0230/`，训练产物 `logs/x2_lafan_formal_20261027_0230_train/`。
- 服务器持久会话负责时间门槛和 GPU 检查，每10分钟的跟进负责核验，不提前或重复启动。

## 2026-10-27 09:09：首轮正式训练已完成，启动完整片段评测

- 核验定时任务于02:30:07在GPU 2启动，约05:30达到3小时时间预算后正常退出（controller.exit=0）；最终19900步、7944次更新，并非完成320000步。
- 最终 train_ret=0.474，WM nse1/nse8=0.0456/0.0579。PyTorch峰值allocated/reserved=4919.52/5272 MiB，设备采样峰值6057 MiB；耗时10846.55秒。检查点 `logs/x2_lafan_formal_20261027_0230_train/ckpt_latest.pt` 可读取，模型参数均有限。
- 用户要求启动实验时，上述训练已结束。GPU 0、1无计算进程，GPU 2、3由其他任务占用；已选择GPU 0启动40段完整片段评测，不重复从头训练。
- 评测会话 `sm-eval-20261027`，记录目录 `logs/x2_lafan_eval_20261027_0910/`；保留检查点SHA256与训练配置。命令：`CUDA_VISIBLE_DEVICES=0 bash scripts/run_local.sh scripts/robot.py x2_ultra scripts/evaluation/eval_benchmark.py --ckpt logs/x2_lafan_formal_20261027_0230_train/ckpt_latest.pt --num-envs 40 --out logs/x2_lafan_eval_20261027_0910/results.json`。日志 `eval.log`，退出码 `eval.exit`。
- 评测逐段从首帧开始、无噪声和随机化，运行到片段结束或提前触发终止；以完成率及失败片段判断续训，不能仅凭训练回报声称成功。监控已按用户要求改为每30分钟一次。

## 2026-10-27 09:24：评测完成，续训排队

- 评测09:14:04正常结束，约4分10秒；40段中31段未触发提前终止、9段失败。帧完成率0.920354、mean_norm_return=0.623573、step_weighted_norm_return=0.635262、full_clip_weighted_return=0.584666。未提前终止不代表跟踪误差已达标。
- 失败片段：dance1_subject1、dance2_subject1、dance2_subject4、fallAndGetUp1_subject1、fallAndGetUp2_subject3、fight1_subject3、fightAndSports1_subject4、jumps1_subject2、walk3_subject3。
- 尚未完成原320000步训练预算，继续使用相同配置，从19900步检查点恢复权重、critic、优化器和动作采样统计；不重新从头训练。入口新增可选第4参数为resume路径，经过bash语法检查。
- 09:24核验GPU 0、1已被2025-Huang.w的新任务占用，2、3亦有其他任务。因此续训仅排队：会话sm-resume-20261027，记录目录logs/x2_lafan_resume_20261027_0925/，产物目录logs/x2_lafan_resume_20261027_0925_train/，等待GPU 1空闲；实际启动后再运行3小时并评测。
- 命令：`bash scripts/experiments/formal_lafan.sh x2_lafan_resume_20261027_0925 1 0 logs/x2_lafan_formal_20261027_0230_train/ckpt_latest.pt`。当前不能声称续训已经开始。

## 2026-10-27 09:26：用户授权扩大规模，改为双卡分阶段验证

- 用户提出增加batch和多卡，并授权代理决定scale。选择两卡，每卡1024环境、replay512、WM batch2048×10、策略batch2048，合计2048环境、全局WM/策略batch4096。不是已测结论；暂不扩大至四卡。
- 已核对项目现有多卡入口：每rank独立仿真与replay，梯度all-reduce，代码按sqrt(world_size)缩放学习率。恢复将加载模型、critic、优化器、迭代数及采样统计；仿真/replay与随机数状态并非原位精确恢复。
- 新增入口 `scripts/experiments/scaled_lafan.sh` 和逐rank测量包装 `scripts/training/profile_distributed.py`，已做shell语法和Python AST静态检查；GPU实测仍待执行。
- 流程：等待GPU0空闲→单卡更大batch跑100步并验证检查点有限及reserved峰值<20GiB→等待GPU0、1同时空闲→双卡跑100步并逐卡检查→再次确认两卡空闲→从原19900步检查点正式续训3小时。两次短测均为独立产物，正式续训不使用短测改变后的学习率日程。
- 旧单卡续训等待会话确认无train.pid后已取消，保留其日志；新会话 `sm-scale-20261027`，命令 `bash scripts/experiments/scaled_lafan.sh x2_lafan_scale_20261027 logs/x2_lafan_formal_20261027_0230_train/ckpt_latest.pt`。记录目录 `logs/x2_lafan_scale_20261027/`，阶段产物分别 `logs/x2_lafan_scale_20261027_{capacity,dual,train}/`。
- 当前所有卡有其他任务，流程等待空闲，不修改或停止他人进程。单/双卡显存和吞吐通过实测后再决定是否继续扩展。

## 2026-10-27 10:47：单卡扩容测试已启动

- 等待流程在10:44:43检测GPU 0无计算进程后，自动启动capacity阶段；进程3979391从19900步恢复，最新日志已到19940步，首次更大batch编译/更新尚在进行，未确认通过。
- 当前GPU 0又出现其他用户新进程（3981445），GPU 1也有新进程3981446；因此设备整体显存与速度不能直接归为本项目独占测量。继续单独记录PyTorch峰值，不触碰其他任务。
- 后续仍需容量验证及GPU 0、1同时空闲才进入双卡验证与正式续训，不重复启动。

## 2026-10-27 10:57：单卡扩容测试通过，等待双卡空闲

- capacity阶段完成19900→20000步、24次新增更新，耗时708.29秒（含更大batch首次编译），控制流程已验证检查点步数及世界模型、策略、critic参数有限；日志显示 `Validated capacity at 20000`。
- 每卡1024环境、WM batch2048×10、策略batch2048的PyTorch峰值allocated/reserved为8596.07/9344 MiB（约8.39/9.13 GiB），通过reserved<20GiB门槛；单卡容量足够。设备采样峰值21199 MiB包含后来进入GPU0的他人进程，不作为本项目独占峰值。
- 10:56:45起流程等待GPU0、1同时无计算进程；10:57两卡仍有2025-Huang.w的3981445/3981446，故双卡短测和正式续训尚未启动。既有会话将空闲后自动推进，无需重复启动。

## 2026-10-27 10:59：用户要求跳过短测，直接扩大环境数续训

- 用户明确要求“直接开始不用测试了”，覆盖先短测的流程要求。正式配置改为每卡2048环境、replay512、WM batch2048×10、策略batch2048；双卡总4096环境/全局batch4096，默认编译、既有sqrt2学习率缩放，3小时时间预算/320000步上限。2048环境和双卡峰值尚未测量，不将此前1024环境结果当作新配置已验证。
- 新增 `scripts/experiments/direct_dual_lafan.sh`，shell语法检查通过，复用逐rank显存测量入口并保留运行期间测量，无额外短训。
- 确认旧流程无dual.pid/train.pid，仅取消本项目等待会话sm-scale-20261027，保留capacity产物和cancelled.txt。
- 已创建唯一新会话sm-direct-dual-20261027；命令 `bash scripts/experiments/direct_dual_lafan.sh x2_lafan_direct_dual_20261027 logs/x2_lafan_formal_20261027_0230_train/ckpt_latest.pt`。记录logs/x2_lafan_direct_dual_20261027/，训练产物logs/x2_lafan_direct_dual_20261027_train/，从原19900步检查点恢复。
- 10:59:27日志确认等待GPU0、1同时空闲，两卡仍由其他用户占用；尚未进入训练。30分钟跟进已更新为此入口，空闲后直接正式续训，不重复旧测试。

## 2026-10-27 14:13：切换等待GPU2、3，释放后直接续训

- 用户要求估算2、3卡释放时间并释放后立即使用。按此前任务所有者授权核查：2025-Chen.g训练已完成，最终1280段评测已完成960段，已用约44.5分钟；按平均速度余约15分钟，仅作估计，不保证GPU释放时间。
- direct_dual_lafan.sh新增可选GPU参数，空闲检查间隔改为10秒；仍以两卡无计算进程为启动条件，参数沿用每卡2048环境/batch2048、3小时续训，不添加无效显存占位。
- 确认旧logs/x2_lafan_direct_dual_20261027/无train.pid后，仅取消sm-direct-dual-20261027等待会话，保留cancelled.txt。新唯一会话sm-dual23-20261027，14:12:44日志确认等待GPU2、3。
- 命令：`bash scripts/experiments/direct_dual_lafan.sh x2_lafan_dual23_20261027 logs/x2_lafan_formal_20261027_0230_train/ckpt_latest.pt 2 3`；记录logs/x2_lafan_dual23_20261027/，训练产物logs/x2_lafan_dual23_20261027_train/。shell语法检查通过，监控已同步新队列，禁止重复启动。

## 2026-10-27 14:42：GPU2、3双卡正式续训已启动

- 等待流程于14:23:43检测两卡空闲后启动train；两rank从原19900步检查点恢复。进程4089322/4089323属2025-Long.xh，显式GPU2、3，每卡2048环境/batch2048，总4096环境/全局batch4096。
- 14:41日志已到21500步、8568次更新，最近约1.5–1.6 it/s，WM nse1/nse8为0.0442/0.0597，指标有限，尚不能证明收敛。
- 两卡各约10987MiB设备显存占用，仅本项目计算进程；这是当前占用而非完整训练峰值。逐rank峰值将在结束后由profile输出。
- 首轮3小时续训预计17:25前后结束，实际以时间预算及进程退出为准，随后进行40段完整片段评测。会话sm-dual23-20261027和独立目录保持不变。

## 2026-10-27 14:47：训练与评测在服务器连续衔接

- 用户要求训练结束后立即启动下一轮。新增chain_lafan.sh，bash语法检查通过；会话sm-chain23-20261027运行 `bash scripts/experiments/chain_lafan.sh x2_lafan_dual23_20261027`，仅等待已有controller退出，不重复启动当前训练。
- 记录logs/x2_lafan_dual23_20261027_chain/。每秒检查当前训练退出码；成功后验证检查点有限，两张卡各评测原40段中的20段，保存分片清单、GPU/进程快照、命令/PID/日志/退出码、检查点SHA256，并合并结果及与首轮比较。
- 完整评测仍有失败且iteration<320000时，自动从最新检查点立即续训下一轮3小时（独立运行名x2_lafan_dual23_20261027_r1、r2等），随后重复完整评测。无提前终止或达到原步数上限即停止自动续训以分析质量，不等同收敛。
- 无额外短测，保留每rank峰值测量；遵守空闲核查，不用空转占位抢显存。只能缩短阶段间空档，不能保证其他用户不在间隙启动任务。30分钟监控已更新为查看chain，不额外启动重复任务。

## 2026-10-27：Webster 视频素材准备与开放模型路线核验

- 用户选定 `https://www.bilibili.com/video/BV1Xy4y1g7xA/`，授权准备其 X2 Ultra 仿真训练数据。原视频已在服务器下载；数据根目录为 `data/video/webster_BV1Xy4y1g7xA_20261027/`，相对于本项目专用大数据目录。
- `raw/BV1Xy4y1g7xA.mp4` 实测为1920×1080、25 Hz、7622视频帧、304.88秒视频流。原始来源信息、描述、SHA256和下载日志保留在同一目录。下载工具 `yt-dlp==2026.10.26` 从清华源安装在独立环境 `/home/2025-Long.xh/HHD/venvs/supermimic-media`，未修改主训练和SOMA环境。
- 新增 `scripts/data/inspect_video_reference.py`，服务器CPU生成带源时间戳的缩略图及可选无音轨片段；记录源SHA256、帧信息和完整转换命令，拒绝覆盖现有输出。已检查片头、132–144秒、250–260秒及256–258秒逐帧画面。片头混有慢放/第一视角，132–144秒包含手撑地分解练习，不能直接作为完整Webster训练轨迹。256.20–257.60秒为连续全身翻转与落地候选；切入时已在动作准备中，未包含充分的静止准备段，尚未证明三维恢复或物理可行性。
- 最初候选GVHMR的固定源码为 `ee960bb6e2ea2d381aa97f08e9b71ef320b624b1`，已在本地临时目录下载并核对接口。该路线需要另行获得SMPL-X人体模型，本项目数据目录内未找到相应模型；没有安装或运行GVHMR。
- 用户希望避免SMPL-X注册下载。核验替代候选[NVIDIA GEM-X](https://github.com/NVlabs/GEM-X)：固定源码 `20555befe2fbbe5b2015e1fff7ff214dc18ea9d6` 已下载到本地临时目录，支持视频时序推理和SOMA世界坐标动作。官方模型仓库 `nvidia/GEM-X` 修订 `4fce057963671fd95cd4154297944397e7b5a9a0` 的API报告 `gated=false`；配套 `nvidia/SOMA-X` 的 `v0.3.3` 对应 `d281db2d01553f7230c56e764cec00fe27ef23f7`，同样 `gated=false`。源码、权重和资产遵守各自许可；没有把人体模型本身当作视频动作恢复器。
- GEM-X当前仅完成源码/接口/公开资产可获取性核验，尚未安装独立环境或下载大模型权重，也未执行三维恢复、SOMA导出适配、X2重定向或Webster训练。其SOMA版本与现有重定向骨架仍需核对。
- 本轮SSH核验LAFAN1双卡续训正在GPU2、3运行，保留原任务；其他GPU也有计算负载。未启动额外GPU任务，未更改已有训练/评测链。

## 2026-10-27 16:18：Webster 已完成视频恢复、约束重定向与最终短训，正式训练排队

- GEM-X固定源码 `20555befe2fbbe5b2015e1fff7ff214dc18ea9d6` 已部署至服务器 `vendor/gem-x`；源码压缩包SHA256为 `926d28c0d3036d7c7898b0d097972018e18551e9a937eea0e597da30d5e71924`，本地/服务器一致。公开权重及SOMA资产共13个文件、7,239,901,706字节，已逐文件核验大小及SHA256并上传专用数据目录 `assets/gem-x/`。版本、来源和完整校验清单见 `configs/video/gem-x-assets.json`，服务器就绪标记为 `assets/gem-x/READY.json`。临时本地大模型副本已在远端验证后删除，保留来源清单与日志。
- 独立GEM环境为 `/home/2025-Long.xh/HHD/venvs/supermimic-gem`，`vendor/gem-x/.venv` 链接至此。Python3.12.13、Torch2.14.1+cu130、Warp1.18.0、Transformers5.3.0、py-soma-x0.3.3；69包依赖检查通过。Torch/Warp复用本项目既有Linux轮子，通用依赖走清华源；主训练和SOMA环境未新增依赖。安装日志保存在 `logs/webster-gem-setup-20261027/`；早期因未命中项目缓存的安装进程仅取消本任务PID后重启，版本未反复调整。
- 实际恢复使用所选视频256.20–257.60秒，保留真实时长，从25Hz规范化至30Hz，得到42帧。恢复在服务器CPU运行，关闭GPU可见性；原始输出、配置、环境快照、命令、PID和各阶段日志均在 `logs/webster_gem_20261027_r2/`。`raw_motion.pt` 的躯干向上分量最小为-0.947、末帧为0.997，说明存在倒立并恢复直立；不据此声称完整动作跟踪成功。
- 新增SOMA FK→BVH适配及真实导入器往返检查。两端均为78节点（含虚拟Root）；GEM的自父Root显式转换为BVH的-1父节点；BVH保持Y向上/厘米，由既有重定向器执行一次Y→Z转换。实际恢复42帧的导入往返最大位置误差约1.13e-6米、旋转矩阵误差约1.51e-6。合成整周旋转验证另存于setup目录，不能冒充视频结果。
- 默认世界速度积分产生明显高度漂移：骨盆末端相对起点下降约0.701米，直接官方接触修正后仍下降约0.554米。保留这两份结果，不作为训练参考。另生成相机空间根位置候选，使用背景估计的相对旋转（最大约0.015度）；明确假设此近固定镜头的相机平移可忽略，深度和尺度仍是估计。相关方法记录在 `camera_trajectory_candidate.pt` 元数据，未把规划或模型估计当成真实测量。
- 原视频逐帧接触证据保存在 `takeoff-contact.jpg`、`landing-contact.jpg`；离地边界在256.48–256.52秒，触地在256.96–257.00秒，误差约一源帧。注释见 `configs/video/webster-contacts.json`。以此在最终50Hz时钟上拟合X2质量模型的腾空质心抛物线，并在支撑阶段约束脚底；不是简单抬高整段轨迹。
- 复用既有 `retarget.py --robot x2_ultra --batch-size 1`，在CPU完成42帧候选重定向。全碰撞检查随后发现左腕 `left_wrist_roll_link/body_24_collision` 起跳前穿地约0.101米。正式r1队列在没有train.pid时被取消，保留 `logs/webster_formal_20261027_r1/cancelled.txt`；没有用这份候选启动正式训练。初次100步smoke仅用于验证流程，不能作为候选质量证明。
- 对左臂加入受限地面避障IK，保持根姿态和腿部姿态，关节改动上限0.6rad，实测最大0.3544rad；重新拟合质心并转换。最终数据为 `data/video/webster_BV1Xy4y1g7xA_20261027/npz_final_50hz_r1/webster.npz`，69帧/50Hz/1.36秒；采用独立清单 `manifests/webster.csv`，不混入LAFAN。转换没有额外关节修复或残余根抬升，仅统一地面偏移约-5mm。
- 最终 `reference-qc.json`：非脚碰撞几何最小离地约14.99mm；全碰撞几何最低约-3.33e-8米（浮点误差）；关节限位最大偏差约4.02e-6rad；腾空质心加速度相对[0,0,-9.81]的最大误差约5.60e-4m/s²。只验证几何、接触与质心约束，未证明角动量、执行器可行性、收敛或泛化。
- 最终数据short-run会话 `sm-webster-smoke-20261027-r2` 已完成100步、24次更新，退出码0；参数有限。记录目录 `logs/webster_smoke_20261027_r2/`，检查点位于同名 `_train/`。耗时174.89秒；本进程Torch峰值allocated/reserved为776.76/902MiB。GPU2启动前只有本项目LAFAN进程4089322、余量13138MiB；短训Torch分配限制为卡容量15%。该短测与本项目LAFAN共享已分配卡，设备采样12358MiB包含LAFAN，不能当作本短训独占显存或吞吐。
- 已启动持久正式队列 `sm-webster-formal-20261027-r2`，命令 `bash scripts/experiments/webster_formal.sh webster_formal_20261027_r2`。记录 `logs/webster_formal_20261027_r2/`，训练产物规划为 `logs/webster_formal_20261027_r2_train/`。该流程先验证最终数据short-run检查点和完整QC，再等待GPU0或1无计算进程且余量≥16GiB。16:18快照仍在等待，没有train.pid，正式训练尚未启动；不停止他人进程、不修改既有LAFAN训练/评测链。
- 首轮配置为单卡512环境、replay512、WM batch256×10、策略batch128、展开32步、关闭Torch编译，最多3小时/320000步。Torch分配上限为卡容量20%（不包括Warp/图形分配）。只从固定LAFAN19900步检查点warmstart世界模型，策略、critic与replay重新初始化。该正式配置的峰值和效果尚待实测，不能将小batch smoke峰值冒充正式峰值。结束后自动检查参数有限性，再在空闲卡上进行单片段完整评测和策略/参考对照录像；未生成的评测、录像均为待执行。
- 参考动画为 `logs/webster_gem_20261027_r2/x2-reference-final-r1.mp4`，明确标注REFERENCE ONLY，不是物理策略执行结果。代码、清单、接触标注和校验入口均已保留；所有三维恢复、重定向、训练与渲染实际运行在服务器。

## 2026-10-27 17:42：双卡首轮续训及评测完成，自动进入r1

- x2_lafan_dual23_20261027正常达到3小时时间预算，controller.exit=0，19900→36700步，累计14648次更新。两个rank的profile均completed，allocated/reserved峰值各8977.03/9760MiB（约8.77/9.53GiB）；GPU2设备采样峰值12358MiB包含同项目Webster短测，不当作LAFAN独占峰值。
- chain自动双卡各评测20段，两退出码均0，合并40段唯一片段。37段未提前终止、3段失败，帧完成率0.964642，mean_norm_return=0.778570，step_weighted_norm_return=0.787532，full_clip_weighted_return=0.759686。相较首轮31/40、92.04%、0.584666改善，但尚不能声称收敛。
- 剩余失败fallAndGetUp2_subject3、fightAndSports1_subject4、jumps1_subject2。结果与比较位于logs/x2_lafan_dual23_20261027_chain/eval-0/{results.json,comparison.json}。
- 17:29:12已自动在GPU2、3从36700步检查点恢复下一轮x2_lafan_dual23_20261027_r1，配置不变，独立记录与_train产物目录。进程39708/39709属2025-Long.xh，17:41日志37800步，检查点已落盘；预期20:30左右预算结束再自动完整评测。chain会话持续控制，无重复启动。

## 2026-10-27 19:52：Webster 首轮正式训练及评测完成，完整动作失败

- `webster_formal_20261027_r2` 在空闲GPU0于16:42启动，19:46达到3小时时间预算结束；3900步、1544次更新，profile耗时11033.86秒，controller.exit=0。Torch峰值allocated/reserved为945.58/1070MiB，设备采样峰值1656MiB。
- 完整单片段评测失败：17/69步，约0.34秒，completion_fraction=0.246377，norm_return=0.290423，full_clip_weighted_return=0.071554。训练流程完成不代表策略学会Webster，未完成翻转与落地。
- `logs/webster_formal_20261027_r2/policy.mp4` 已生成，9帧25fps；实体为物理策略，青色半透明为参考，终止后环境自动重置的画面不可解释为成功恢复。策略录像和评测已取回本地供查看。当前未启动新的Webster续训。

## 2026-10-27 19:57：Webster 失败后连续录像

- 用户要求观察失败后的动作。录像入口新增可选 `--continue-after-failure`：保留原终止检测与标志，在该录像进程抑制重置请求，继续执行原策略；参考片段结束后保持最后姿态。训练及正式评测的失败标准未变。
- 空闲GPU0执行，tmux `sm-webster-diagnostic-20261027-r1`，Python PID136014，记录 `logs/webster_diagnostic_20261027_r1/`。检查点SHA256 `6694ab2472257a30078280857bdd55c3956aa74517a63b794151d4ed02506d7a` 与正式录像一致。
- 实测150步/3秒、75帧25fps，退出码0，无自动重置。首次失败仍为第17步/0.34秒，触发项 `root_height`（实际与参考骨盆高度偏差）；后续实体机器人倒地，未完成翻转和落地。录像标注诊断模式与失败时刻，不代表完整动作成功。
- 服务器 `logs/webster_diagnostic_20261027_r1/policy.mp4` 已生成并取回本地展示。AST检查及实际连续录制通过。

## 2026-10-27：Webster 失败原因初查

- 确认直接失败项为实际/参考骨盆高度绝对偏差超过0.25m，在17步/0.34秒出现；参考最后支撑边界为15帧/0.30秒。连续录像随后倒地，因此并非仅录像截断。
- 实际运行配置已核对：policy_trainer.anchor_obs=True、goal_dim=729；config-preview中的False/720是Trainer初始化前的值，不是运行错误。
- `audit_airborne_dynamics.py` 在服务器CPU检查最终NPZ，以位姿中心差分速度计算MuJoCo整机关于质心的世界系角动量。排除起落接触边界，1帧/2帧差分半径的平均角动量模长约27.78/27.66 kg m²/s，距均值最大偏差约21.79/13.26 kg m²/s；两种差分均显示空中角动量变化显著。详细结果 `logs/webster_diagnostic_20261027_r1/airborne-dynamics.json`。数值受视频恢复误差和差分影响，提示参考空中旋转存在动力学不一致，不证明这是唯一失败原因，也不等于已验证执行器可行性。
- 此前只校验穿地、关节限位、接触和质心抛物线，未校验角动量及执行器；不应把可播放参考当成动态可执行目标。
- 3小时因关闭编译、保守batch仅完成3900环境步/1544次更新，策略从零学习；train_ret从约0.374到0.388，缺乏明显改善。return采样从片段中间初始化，训练平均回报不能代替从起点完整动作评测。尚未做对照试验，不能归结为单纯时间不足或承诺续训能解决。

## 2026-10-27 20:35：r1完成，评测改善并自动进入r2

- r1达到3小时时间预算，controller.exit=0，36700→53500步、累计21352次更新，两个rank profile completed，PyTorch峰值allocated/reserved各8977.03/9760MiB；设备峰值23906MiB不是本进程独占指标。
- eval-1合并40段完整评测：37段未提前终止、3段失败；帧完成率96.8813%，full_clip_weighted_return=0.799997，mean_norm_return=0.816377。相较eval-0的96.4642%/0.759686改善，但失败数量仍3段且身份改变，不视为收敛。
- 本轮失败fallAndGetUp1_subject1、fallAndGetUp1_subject5、jumps1_subject2；此前失败的fallAndGetUp2_subject3和fightAndSports1_subject4本轮未提前终止，新出现两段fallAndGetUp失败，需继续跟踪逐片段表现。
- 20:33:14已自动在GPU2、3启动r2，从53500步检查点恢复，20:34最新53600步；PID161012/161013。运行记录logs/x2_lafan_dual23_20261027_r2/，产物同名_train/。预计23:35前后达到3小时时间预算，随后由chain继续完整评测，无重复启动。

## 2026-10-27 21:07：Webster 修正与短训对照，尚未证明完整动作成功

- 用户要求修正起跳与空中旋转后短训。原视频、旧参考、旧模型均保留，未修改机器人质量、重力、电机限值或LAFAN训练任务。修正记录根目录为服务器 `logs/webster_repair_20261027_r1/`，数据位于既有视频目录的独立子目录。
- 已测得旧参考起跳处质心竖直速度相邻差约3.132m/s；旧接触处理仅对脚底高度对齐，存在支撑滑动和速度衔接问题。新增接触/质心约束、常角动量旋转积分及后续电机力矩优化。所有恢复、几何/动力学求解均在服务器CPU执行。
- `centroidal_ready_r3` 候选通过当时的几何/质心检查，起跳质心离散速度差范数约0.06857m/s，空中角动量相对最大偏差约0.00609（修正角速度坐标后）。这是分项修正，未获完整动力学可执行认证。
- 首次短训 `webster_repair_short_20261027_r1` 的总步数4100与恢复迭代3900使余弦学习率提前进入final_ratio=0.03；已明确取消，controller.exit=143，保留日志/检查点，不作为有效修复结论。修正短训使用lr_final_ratio=1，恒定2e-4。
- 有效对照 `webster_repair_short_20261027_r2` 在空闲GPU0完成3900→4100步，累计1544→1608次更新，耗时415.98秒；Torch峰值allocated/reserved为635.28/760MiB，退出码0。相同候选上训练前后norm_return 0.234384→0.249901，均在16/69步失败，未证明起跳或翻转成功。物理策略连续录像已生成并取回，不能用得分小幅提高宣称动作学会。
- 又发现数据/重置接口不一致：环境按0.9软限位裁剪关节，旧候选最大超出约0.113249rad、首帧约0.057475rad。`soft_aligned_r1` 按实际软限位修正并逐帧保持COM，最大根平移补偿约2.478mm；后续优化与QA均使用相同软限位。主环境重置逻辑未放宽。
- 早期逆动力学拟合使用中心一阶差分再求导，会掩盖相邻帧振荡，不能据此认定力矩合格；后续已改为同一姿态坐标内前/后边速度的直接二阶差分，并把离地固定边界帧纳入约束。旧 `force_repair_r1/r2` 等局部结果只保留作诊断，未经独立验证不得训练或宣称可行。
- 一致离散方式下，`flight_force_repair_r2` 对旧17～39帧的局部空中约束拟合，最大估计电机比例约0.9851、浮动基座力残差约0.40N、力矩残差约0.061Nm，关节在软限位内；这是数值分项检查，不是物理策略成功或落地验证。原节奏起跳拟合仍未通过整体约束。
- 用户随后明确允许延长准备阶段，优先仿真可执行，保留单腿起跳与完整前空翻。新候选 `retimed_seed_r2`：准备0.3→0.6秒，空中仍0.5秒，共84帧50Hz；离地COM上移0.06m并线性衔接到原落地高度，空中竖直速度改变-0.12m/s、重力不变。支撑腿初始化最大COM误差约8.85e-6m，接触误差约1.32e-6m。抬高0.15m的前一候选几何不可达，已拒绝，未训练。
- 当前联合起跳/空中优化会话 `sm-webster-retimed-20261027-r1`，PID184746，源码入口 `refine_webster_takeoff_forces.py`，takeoff=30、switch=10、landing=55、最多80次函数评估；输出规划 `retimed_force_r1`，完整命令见会话/日志。原模型力矩与软限位保持不变，脚底接触目标为地面0m。此时仍在求解，尚未通过独立QA，尚未启动延时候选的短训。


## 2026-10-27 22:24：Webster 延长准备与完整控制器迁移短训完成，起跳仍失败

- 前述21:07的联合求解状态已过时：后续全局力矩优化候选未通过整体可行性检查，所属求解进程已按PID核验并停止；失败输出保留，不作为训练数据或可行性证明。数值工具另完成自由落体与匀速平移不变性检查，日志见 `logs/webster_repair_20261027_r1/numerical-invariants.log`。
- 本轮实际数据为 `data/video/webster_BV1Xy4y1g7xA_20261027/npz_repaired_ease_r1/`：84帧、50Hz、1.66秒。准备阶段0.3→0.6秒，采用二次ease-in保留离地时间映射导数，空中0.5秒不变；未使用出现大幅髋关节跳变的IK候选。源于局部空中修正，软限位一致化后保持COM，最终将空中COM精确投影到重力抛物线。
- 实测起跳COM相邻速度差范数0.0387113m/s；空中角动量距均值最大相对偏差为0.0186186（差分半径1）/0.0244196（半径2）。全碰撞最低-1.406mm、非脚最低5.895mm、空中重力残差最大0.000295m/s²，硬/软限位违规为0。以上仅分项检查；起跳逆动力学估计仍有电机超限，`repair-validation.json` 明确 `motor_feasibility_passed=false`，不能称为整体动态可执行。
- 发现原Webster正式训练只迁移世界模型、策略从零学习。本轮改用已完成的LAFAN完整控制器/世界模型/critic检查点53500步，SHA256 `73c3bccbf594672974afae69d224910578c376bcccf6601c1fac00776753d519`；更换动作库后清空replay与采样统计。此实验同时改变初始化和参考，只能在同一新参考上比较短训前后，不能归因于单个修复。
- 会话 `sm-webster-transfer-short-20261027-r1`，独立记录 `logs/webster_transfer_short_20261027_r1/`，产物 `logs/webster_transfer_short_20261027_r1_train/`。启动前GPU0无计算进程，显式分配GPU0；保留LAFAN GPU2/3速度。256环境，WM batch256×2，policy batch64/horizon32，恒定学习率2e-4，关闭随机化与观测/动作/重置噪声，关闭编译，单次上限10分钟/新增200步。完整命令、启动脚本、输入SHA、配置、PID与profile均保留。
- 实际53500→53700步、21352→21416次更新，409.996秒，controller.exit=0；Torch峰值allocated/reserved为635.288/760MiB，设备采样峰值1276MiB。12个相关Python源码AST与两个shell入口语法检查通过。
- 同一新参考起点评测：短训前34/84步（0.68秒）、norm_return=0.327802、full_clip_weighted_return=0.132682；短训后35/84步（0.70秒）、norm_return=0.545534、full_clip_weighted_return=0.227306。两次均失败，未证明成功起跳、翻转或落地，不能因回报改善宣称修复成功。
- 训练后实体策略连续录像3秒/75帧25fps，0.70秒首次触发root_height，之后继续执行策略并倒地；参考在片段末保持末姿态，不自动重置。实体为真实闭环仿真，青色半透明仅为运动学参考。检查点SHA256 `709317d6f802d3fac083770d297f0d3cc86291b9bbe37899d226d0022ef8f316`。录像与metadata在本轮记录目录；本地展示副本为 `webster/repaired-transfer-policy.mp4`。
- 本轮完成分项参考修正与有界实际控制验证，但用户要求的实际起跳修复仍未达到成功条件。后续重点是支撑阶段可实现的起跳速度/角动量与电机约束联合设计；目前不启动数小时续训，不能承诺增加训练时间可解决此失败。

## 2026-10-27：用户授权 Webster GPU0 单卡6小时探索续训

- 用户随后要求尝试长训、指定物理GPU0并扩大环境数，明确时间预算为6小时。该授权更新了上一轮“不启动数小时续训”的状态；参考仍为 `npz_repaired_ease_r1`，起跳电机可行性问题未解决，不声称长训一定成功。
- 启动前核验主机 `amax-Super-Server`、本用户UID1006、GPU0无计算进程且显存仅16MiB；GPU1有其他用户任务，GPU2/3为既有LAFAN任务，均未干预。RTX3090 24GiB、驱动580.178.04；专用环境Python3.12.13、Torch2.14.1+cu130、MuJoCo/mujoco-warp3.16.0、Warp1.18.0，CUDA可用。HHD实际挂载点 `/mnt/local_DATA_HHD` 余量3.2T。
- 新入口 `scripts/experiments/webster_long.sh` 复用既有训练和显存测量入口，不使用20%分配限制的bounded入口。完整模型/优化器从 `logs/webster_transfer_short_20261027_r1_train/ckpt_latest.pt`（53700步、21416更新）恢复；容量测试与正式训练分别从同一检查点开始。保持上一轮无噪声/无随机化、恒定2e-4、WM每次2批、policy horizon32，扩大WM与policy batch至2048，replay capacity512，关闭编译。
- 两次独占GPU0容量测试均完成60个新增环境步、8次完整更新，控制退出码0，WM/policy/critic参数全部有限。8192环境：Torch allocated/reserved峰值10627.35/11432MiB，设备峰值14508MiB；16384环境：13691.62/14472MiB，设备峰值20172MiB。独立记录 `logs/webster_capacity_20261027_r1/`、`logs/webster_capacity_20261027_r2/`。前50步sim计时3.8→5.5秒、wm 2.9→2.7秒、policy 28.8→27.7秒；样本有限且含warmup，不能据此声称长期速度或收敛改善。
- 选择16384环境、WM/policy batch2048，预留约4GiB设备显存。正式命令 `bash scripts/experiments/webster_long.sh webster_long0_20261027_r1 16384 2048 6`，持久会话 `sm-webster-long0-20261027-r1`；独立控制/日志目录 `logs/webster_long0_20261027_r1/`，训练产物 `logs/webster_long0_20261027_r1_train/`。每50步记录、每500步保存，320000总步数上限，实际以6小时时间预算为先；时间限制在日志点检查，会有检查粒度超时。
- 控制目录保留启动脚本、完整命令、解释器/依赖快照、输入SHA256、来源提交记录及训练源码SHA256清单（当前上传目录没有Git元数据）。训练结束自动验证参数、完整单片段评测并录制3秒连续策略录像；这些结束结果均待执行，启动不代表收敛或动作完成。
- 用户随后明确改为8192环境。已核验并仅终止本轮16384环境训练PID247875，`webster_long0_20261027_r1/controller.exit=143`，未产生首个周期检查点，保留取消说明和全部日志。替代命令 `bash scripts/experiments/webster_long.sh webster_long0_20261027_r2 8192 2048 6`，会话 `sm-webster-long0-20261027-r2`，从原53700步短训检查点重新开始；6小时预算与其余参数不变。有效控制目录改为 `logs/webster_long0_20261027_r2/`，训练目录为同名 `_train/`，8192环境容量实测设备峰值14508MiB。
- 已确认替代任务Python PID249577进入实际更新：53750步、21420次累计更新，GPU0进程显存14484MiB，未出现退出标记。启动后的长期峰值、6小时最终结果与评测待运行结束后核验。


## 2026-10-27 22:46：并行于用户长训的Webster CPU接触诊断与局部优化

- 用户在另一侧准备六小时长训，要求本侧继续优化动作。核验GPU0进程242933为本项目 `webster_capacity_20261027_r1` 大规模试跑，GPU2/3仍有LAFAN；本侧仅在服务器以CUDA不可见、单线程、nice=10执行CPU求解与检查，未改动正在训练的NPZ、清单或模型，也未启动新GPU任务。
- 深化逆动力学审计：逐帧记录COM、角动量、根角速度、关节速度/加速度、足底高度、所需地面推力/压力中心。新增可选 `--exact-contact-wrench`，用非负摩擦射线线性规划，在严格满足浮动基座所需受力/力矩的前提下最小化电机峰值比例。此前NNLS会在基座受力残差和电机代价之间折中，不能单凭其峰值比例判断电机能力。解析小算例验证20Nm残差/10Nm限值=2，无法由竖直射线提供的水平推力被拒绝；日志 `logs/webster_takeoff_window_20261027_r2/wrench-invariants.json`。
- 当前参考19～30帧（0.38～0.60秒）在估计的12mm足部接触/摩擦系数1下，只有19帧能严格平衡所需地面wrench。25帧所需压力中心约[0.309,-0.029]m，估计接触点包围盒x=[0.558,0.606]、y=[0.133,0.199]m，明显落在脚外；30帧压力中心约[2.398,1.402]m，提示离地旋转衔接严重不一致。以上依赖有限差分和接触估计，证明当前参考在该检查下不自洽，不证明X2不能完成Webster。
- 局部窗口求解只优化全局19～32帧，33帧及其后姿态固定；输入切片及SHA保存在 `logs/webster_takeoff_window_20261027_r1/`。发现原优化器在足部patch模式下把虚拟脚尖中心用作固定支撑pivot，窗口起点实际脚跟更低，虚拟脚尖高约0.102m，导致不正确的锚点误差。新增显式 `--contact-anchor lowest` 选择实际初始最低碰撞顶点；旧toe默认保留以便历史命令复现。仅这一修正不能证明完整滚动接触正确。
- 两个有界局部求解均结束：r1为旧toe pivot40次评估、r2为lowest pivot80次评估。r2会话 `sm-webster-window-20261027-r2`，启动脚本 `scripts/experiments/webster_takeoff_window.sh`，最多600秒；controller.exit=0。源码快照、命令、PID、SHA与优化过程保存在同名logs目录，不自动启动训练。
- 新增 `assemble_motion_window.py` 将局部结果放回完整84帧并跨边界重新生成速度。组装输出 `data/video/webster_BV1Xy4y1g7xA_20261027/takeoff_window_assembled_r2/`，不覆盖长训输入；整段独立复查包含窗口两端，防止只查看局部拟合数据。
- r2全段审计结果：19～33帧估计峰值电机比例5.1894→3.5215，基座角力矩残差峰值767.74→324.17Nm，均未达标；19～30帧严格地面wrench可平衡帧仅20/21帧。几何最低-2.327mm、非脚最低1.652mm（低于5mm检查阈值），空中重力残差5.528m/s²（阈值0.02），硬/软限位无违规。候选被拒绝，`comparison.json` 明确approved_for_training=false。优化目标下降不能作为成功或数据替换依据。
- 本轮完成了接触点假设修正、独立压力中心约束诊断和有界候选验证，尚无新的可用Webster参考。后续需要联合设计支撑阶段重心、摆腿与角动量，不能继续把相机估计的COM和原摆腿末端速度当作固定可行目标。六小时长训的结果仍独立评估，本轮不把CPU候选变化归到该训练结果中。

## 2026-10-27 23:42：LAFAN1基线40段完整评测全部完成，停止自动续训与监控

- r2正常完成53500→70000步，累计27936次更新，controller.exit=0；两个rank profile completed，峰值allocated/reserved各8977.03/9760MiB，设备采样峰值各10987MiB。有效最终检查点logs/x2_lafan_dual23_20261027_r2_train/ckpt_latest.pt；chain进入评测前已验证模型参数有限。
- eval-2两卡各20段退出码均0，合并40段唯一片段：failed=0、completion_fraction=1.0，mean_norm_return=0.844981，step_weighted_norm_return/full_clip_weighted_return均0.849197。相比首轮31/40、帧完成92.04%、0.584666，及r1的37/40、96.8813%、0.799997均改善。
- 上轮失败的fallAndGetUp1_subject1、fallAndGetUp1_subject5、jumps1_subject2本轮均未提前终止。结果保存logs/x2_lafan_dual23_20261027_chain/eval-2/{results.json,comparison.json}，对应iteration.txt=70000。
- chain按预定failed=0条件正常退出（controller.exit=0），没有启动r3；GPU2、3当前无计算进程。已完成本监控范围的LAFAN1基线训练与完整片段评测，删除30分钟监控x2-ultra-lafan1。未提前终止不等于精确跟踪、泛化或收敛认证；后续质量检查或新实验需按用户需求开展，未为占卡添加空转任务，不影响独立Webster任务。


## 2026-10-28：Webster 重设计、CPU/GPU 搜索与首次 Git 回档点

- 用户要求先把单腿支撑和摆腿重设计到可执行，再短训，并保留另一侧六小时长训。六小时任务 `webster_long0_20261027_r2` 的GPU0进程249577、输入 `npz_repaired_ease_r1` 和检查点目录均未改动。本侧设计过程中仅使用服务器CPU，用户报告显卡释放后核验GPU2/3为空，再使用GPU2；不占用GPU0/1，不停止他人进程。
- 独立CPU前向模型尝试 `webster_forward_design_20261027_r1` 等均未获完整可执行认证：r1末期向上速度约1.66m/s、角动量Y约23.17kg m²/s，但软限位最大偏差约0.142rad；接入空中动作的probe累计根俯仰变化约4.98rad，落地失败。r2/r3按核验PID停止并保留原因；r4完成9720次前向评估，软限位可不违规，但未真实起跳。不得将这些局部或简化控制器结果当成训练策略成功。
- 首次确认原项目 `make_env(..., device="cpu")` 可直接运行。原53700步策略在同一参考上CPU完整评测为35/84步失败、norm_return约0.545534，与此前GPU结果一致。CPU后端探测保存在 `logs/webster_cpu_backend_probe_20261027_r1/`；此后以项目原环境设计，不修改机器人质量、重力、电机限制或底层驱动。
- 原输入首帧右踝pitch与左肘均距软限位约1e-5rad，原策略首20ms右踝已超出约0.03378rad。仅把首控制目标约束到内侧未解决初期问题，不能称为修复成功。新独立引导 `guide_neutral_r1` 从项目中立蹲姿出发，按真实碰撞面法向调整平脚，留0.15rad姿态余量、平滑进入准备段，头/腕/腰使用中立引导；原训练输入不覆盖。新引导CPU前0.2秒无关节软限位越界，但后续仍未起跳成功。该文件有 `GUIDE_ONLY.json`，明确禁止直接作为已验证训练数据。
- 新入口 `optimize_webster_policy_cpu.py` 用冻结53700步控制器、准备时长及有界残余位置命令搜索实际物理轨迹；当前已支持CPU/GPU及后续独立空中/落地阶段。关闭旧参考失败后的重置，仅限此设计进程；成败另按真实腾空、临近起跳的单左腿支撑、空中倒立和主要前转、完整旋转、落地稳定、全物理子步软限位/速度/力矩、足底及非脚碰撞几何检查。搜索不是神经网络短训，策略权重未更新，训练批准字段保持false。
- CPU设计 `webster_cpu_takeoff_20261027_r2` 在单个空闲CPU核、nice19、单线程运行，确定性完整重置重复qpos误差为0。用户要求加速后保存最好候选至 `gpu-migration-final.npz`，核验命令和进程所有者后仅停止本任务CPU PID294112，迁移记录在同目录 `gpu-migration.json`。
- GPU批量指标 `BatchedDesignMetrics` 使用同模型质量/惯量、姿态和速度计算COM与角动量，并检查全部碰撞凸包顶点高度。四个参考姿态与MuJoCo独立CPU计算对照，COM/速度/角动量最大误差约8.82e-6；结果 `logs/webster-gpu-metrics-check-r1.json`。GPU重复初态和首策略动作一致，但后续整段轨迹有数值差异，旧逐帧1mrad重复检查未通过，原失败记录保留；允许搜索不代表已认证，候选必须另做至少三次独立完整物理执行验证才能进入导出和训练。
- GPU16环境试测每16候选迭代约1.277秒，Torch峰值allocated/reserved约245.60/272MiB；这不是整卡显存。正式GPU2设计 `webster_gpu_takeoff_20261028_r1`，tmux `sm-webster-gpu-takeoff-20261028-r1`，工作PID303824，128环境、每轮256候选，160轮总41216次评估（含基线复测），526.90秒、controller.exit=0；每256候选迭代约3.17秒，Torch峰值allocated/reserved约245.60/402MiB，运行时整卡采样约850MiB。相较CPU每24候选约28秒，当前实际吞吐大幅提高，不能把它当作神经网络训练速度。
- 本轮GPU搜索最终仍失败：无有效腾空段，软限位最大偏差约0.05243rad、足底最低约-0.04675m，未通过可执行门槛，未启动后续短训。完整命令、初始权重/引导SHA、源码快照、候选轨迹、指标和重复性检查均在该独立运行目录。后续需要进一步重设计固定支撑脚的准备几何和起跳/收腿时序，不能继续靠增加同一搜索轮次宣称动作可执行。
- 用户随后明确要求Git回档和后续定期提交。核验本地与服务器源码原先均无Git仓库。本地现初始化main分支，当前源码、配置、来源锁、清单与实验记录将作为包含未完成研究的首次checkpoint保存；忽略数据、权重、环境、资产和训练产物，保留原先明确允许的项目演示素材。Git规范已追加至AGENTS.md；提交号和定期任务设置以实际完成结果补记。

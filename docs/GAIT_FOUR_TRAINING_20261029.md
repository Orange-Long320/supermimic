# 四段步态单卡一小时训练（2026-10-29）

## 用户最终配置

- 参考：IMG4884前进、同源合成后退、同源合成原地左转60°、用户确认的IMG4881小跑。
- 单卡GPU1（RTX3090），8192环境，世界模型batch2048，策略batch2048；replay256，世界模型每次更新10批，想象horizon32。正式优化一小时，容量检查耗时另计。
- 用户明确授权这四段试训；参考此前的动态稳定、脚滑和自碰撞未验证，授权试训不代表参考物理合格或策略成功。没有使用弧线左转替代原地左转。

## 数据与源码

- 来源和SHA256锁定：`configs/training/gait-four-1h.json`。四段204/204/204/131帧，总743帧，50Hz。独立复制到服务器 `/home/2025-Long.xh/HHD/data/supermimic/data/video/gait_four_final_train_20261029_r1/`，manifest记录来源和检查。有限值、关节顺序、采样率及硬限位检查通过；微小约4e-6rad数值误差保留，未裁剪。
- 最终运行源码快照 `aceb3ea`，隔离目录 `/home/2025-Long.xh/HHD/data/supermimic/logs/gait_four_final_1h_20261029_source/`，src固定；机器人资产与vendor仍链接项目现有固定资产，未改机器人或共享源码。独立环境解释器 `/home/2025-Long.xh/HHD/venvs/supermimic/bin/python`；Python3.12.13、Torch2.14.1+cu130、MuJoCo3.16.0、MuJoCo-Warp3.16.0、Warp1.18.0。
- 使用进程环境清除外部PYTHONPATH/PYTHONHOME和模型覆盖变量；CUDA_VISIBLE_DEVICES=1。无境外在线下载。启动前检查GPU占用与所有者；未停止其他任务。

## 容量检查与用户调整

- 最初1024环境batch512排程在用户要求提高并行度后取消，只终止本任务控制与其子进程；日志保留于 `logs/gait_four_1h_20261029_r1/`。未启动该计划的正式一小时训练。
- 后续4096环境batch512短训100步/24更新，峰值4857MiB；8192环境batch512累计200步/48更新，峰值7129MiB。用户随后固定8192环境并将两个batch依次要求1024、最终2048，停止本任务扩到16384的控制PID1737689及子PID1743386；未采用16384结果，未用batch1024启动正式训练。
- 最终8192环境batch2048从同数据容量检查点继续，短训完成累计300步/72更新（该阶段新增24更新），182.60秒。设备采样峰值12993MiB，Torch reserved9872MiB、allocated9086.26MiB；全部模型权重有限。容量检查点SHA256 `554138aee6a8cc03916283490f574ad8d956c76187a8d35cc2f424c7477879ea`。
- 这四段从新模型开始，容量阶段权重顺序延续；没有使用其他任务的预训练或运行中检查点。正式阶段从最终同配置容量检查点恢复；原始参考和旧运行均保留。

## 正式一小时：运行中

- 服务器运行目录 `/home/2025-Long.xh/HHD/data/supermimic/logs/gait_four_final_1h_20261029_r1/`。
- tmux `sm-gait-four-final-1h`，控制PID1744626，正式训练PID1747079，物理GPU1。
- 入口 `scripts/experiments/run_gait_four_fixed_hour.sh`；完整展开命令保存在运行目录capacity-command.sh和train-command.sh，环境与包版本在environment.txt和packages.txt，输入数据清单保存在dataset目录对应manifest。
- 正式TrainConfig、开始状态与最终profile写入 `train_1h/result.json`；进度 `train.log`；检查点 `train_1h/ckpt_latest.pt` 与每500步快照。
- 实际进入训练循环：北京时间2026-10-29 14:54:26；一小时预计约15:54–15:55结束。现有循环每20步检查墙钟预算，结束会保存最新检查点；启动/资源等待不算正式一小时。
- 当前仅确认训练循环已开始，最终完成、峰值、完整动作跟踪及泛化待后续核验，不能将短训回报当成完整成功率。
- 启动前完整Git bundle已备份本项目服务器logs；本记录作为运行中checkpoint另外提交并备份。

## 用户要求的中途检查点录像

- 使用当时最近保存的500步/136更新检查点，冻结副本位于 `logs/gait_four_midpreview_20261029_r1/checkpoint.pt`，SHA256 `72d70d9b012efb6da9ebb447637ef33101b7f8e1730951e3134562ae4c999c78`。不是采样时已经超过900步但尚未保存的内存权重。
- GPU3启动前空闲，独立tmux `sm-gait-four-mid-preview`，控制PID1771288。GPU1原训练未停止或切换权重。固定训练源码快照执行 `record_policy_states.py`，各段从第0帧、无动作/观测噪声和随机化执行，首个原始物理失败停止；没有自动reset、保护伺服、附加指令或参考动画替代。
- 从实际记录的50Hz状态渲染：前进0.96秒（参考进度49/204）、后退0.94秒（48/204）、原地左转0.94秒（48/204）、小跑0.94秒（48/131）；均触发原始失败，policy_alone=true。均未完成动作；多数尚在开头准备阶段，不以训练回报替代完整验收。
- 四格预览顺序左上前进、右上后退、左下原地左转、右下小跑。每格标注500步及停止时间；较短片段末帧定格用于比较，不是继续执行。输出1280×960、50fps、98帧、1.96秒，已目视检查联系表。
- 实际指令、轨迹、独立几何报告、渲染来源与汇总均保存在 `logs/gait_four_midpreview_20261029_r1/`；本地视频 `outputs/gait_four_midpreview_20261029_r1/four-gaits-policy.mp4`。预览入口 `preview_gait_four_checkpoint.py`，渲染入口只新增可选标签与镜头距离，不改变记录状态。

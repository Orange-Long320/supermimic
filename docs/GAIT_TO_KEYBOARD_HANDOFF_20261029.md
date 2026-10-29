# 五动作底座与后续键盘控制的衔接

用户要求先用LAFAN70000初始化五动作训练，之后接入[规划键盘行走模型训练](thread://01a18229-5cdd-7ed0-9053-e5e3931e9a95?hostId=local)。已读取该聊天的近期内容，并核对其工作树中的locomotion/config.py、policy.py和trainer.py；本次未向该聊天发消息、启动其训练或修改其工作树。

## 本次训练交付边界

- 五段参考：前进walk、后退backward、原地左转turn_left_inplace、原地右转turn_right_inplace、小跑jog。两种转向均固定骨盆水平位置，设计角度分别+60°/-60°；源为IMG4884风格，小跑为用户确认的IMG4881。
- 来源和哈希：服务器 `data/video/gait_five_train_20261029_r1/manifest.json`；源码配置 `configs/training/gait-five-lafan70000-dual1h.json`。
- 本次仍是motion-conditioned动作跟踪策略，包含motion encoder、共享decoder、世界模型和critic；不是键盘或速度指令策略。左右转与后退含合成运动学参考，不能据此声明物理执行合格。
- 待训练完成保留checkpoint、config、模型/数据哈希和从第0帧完整执行结果，作为后续候选底座。未取得完整执行结果前不标记“键盘可用”。

## 已核对的键盘方案接口

- 该工作树的VelocityPolicy继承现有Policy，保留motion encoder与共享decoder，另加`encoders['velocity']`，输入历史观测与三维速度指令。未来可用本次模型初始化兼容的世界模型、motion encoder、decoder；新增velocity encoder和速度critic需要专门训练。
- 接入前对比机器人资产、关节顺序、观测history、anchor_obs、future_frames、latent与decoder维度，严格检查加载参数，不能仅凭“都是X2”直接覆盖。
- 其当前VelocityCfg.forward为(0.0,0.8)，不覆盖负向速度；本次包含后退，后续需显式扩展负向范围并核对身体坐标系。偏航范围已为双向；原地转向要分别验收平移接近0和偏航方向/速度，不能只看身体转过角度。
- 源视频恢复的平移和速度均为单目估计。速度指令监督应从实际机器人参考的身体坐标速度与偏航速度核对后建立，不能直接把动作名字映射成未经测量的固定m/s或rad/s。
- 该聊天已有“前进→左转→停止”的连续录像和停止修复记录。后续保留其修复代码与可回退检查点，不能用本次tracking checkpoint覆盖已有velocity encoder。
- 停止需要从实际运动状态测试零指令，包含前进/后退/左右转/小跑到停止，以及速度和方向切换。五段跟踪训练没有独立证明停止能力；需要保留参考监督、站立/零速训练和连续不重置验收。

## 本次运行

- 服务器记录根目录：`/home/2025-Long.xh/HHD/data/supermimic/logs/gait_five_lafan70000_dual1h_20261029_r1/`。
- 候选训练输出：`train/ckpt_latest.pt`，尚待运行完成核验。固定LAFAN源副本`lafan70000.pt`，70000步/27936更新。
- 双卡GPU0、2，每卡8192环境与两个batch2048，共16384环境、全局两个batch4096。仅迁移原LAFAN三模块权重，优化器、回放、采样统计、迁移计数重新初始化；两rank入口权重已独立逐项哈希核对与LAFAN70000完全一致。
- 正式一小时从双卡容量及同步验证通过后计时；当前状态以controller.json、dual-proof.json、train/rank-*.json及日志为准。

## 2026-10-29 15:45：五动作双卡正式训练已启动

- 先前四动作双卡验证目录 `logs/gait_lafan70000_dual1h_20261029_r1/` 保留，其控制排程因用户增加右转而取消；没有使用四动作容量证明替代五动作证明。最终五动作完整源码快照 `8761f19`，隔离目录 `logs/gait_five_lafan70000_dual1h_20261029_source/`。
- 新增原地右转由既有源动作在服务器CPU以--angle-deg -60 --in-place生成，204帧，骨盆水平漂移0，最大脚部目标误差0.393毫米、残余地面校正4.44毫米、硬限位越界约4.02e-6rad；NPZ SHA256 `aaaa641356b0f2364aa1265ec163946da324de8ae8fe78bde3a3084b1260fa48`。原始四段不变。未将该运动学检查作为动态成功证明。
- 原LAFAN70000固定副本SHA256 `dc8f02f29f6e03064e9e3c9f4de3cb831a241181636c012dbbb558ee681377a8`；两rank的世界模型、策略、critic入口张量已独立计算哈希，均与原检查点逐项一致。只加载这三个模块的权重，优化器和迁移计数为新状态，没有加载旧70000步的优化器或采样历史。
- 双卡容量100步/24更新完成；两rank更新后模型哈希相同，NCCL同步验证通过，检查点权重有限。设备峰值13028/13029MiB，容量checkpoint SHA256 `fd4a3200d8bc70eb5f8f04093e1627d5b0847e842fdb863b57eedf0d86ea91e7`。dual-proof.json明确绑定五段数据manifest SHA256 `3759116728f8e76e96033649566d64597e0ffc34d054646facd60fd40ad84094`。
- 正式阶段从该五动作容量检查点100步/24更新继续；迁移步数单独计数，LAFAN源步数70000另行记录。实际每优化器LR2e-4：入口除以sqrt2，抵消既有Trainer双卡缩放，避免重复放大学习率；恒定日程。WM每次10批、policy horizon32，关闭Torch编译，与本轮单卡学习设置一致。
- 每卡8192环境、WM/策略batch各2048，GPU0与2，共16384环境、两个全局batch4096。tmux `sm-gait-five-lafan70000-dual1h`，控制PID1793601，正式rank PID1796063/1796064。15:45:49北京时间开始，预计16:46左右完成；容量时间不计入正式一小时，每20步检查预算，结束保存检查点。
- 当前状态为正式训练运行中，不代表五动作均已学会。日志train.log，逐rank状态train/rank-*.json，输出train/ckpt_latest.pt；全部位于服务器 `logs/gait_five_lafan70000_dual1h_20261029_r1/`。源码bundle、固定源权重、原始数据、命令与容量证明均保存，尚未执行未来键盘速度适配。

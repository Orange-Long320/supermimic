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

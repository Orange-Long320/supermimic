# 后退根因与单项验证（整套键盘策略尚未验收）

## 已定位范围

- 数据符号正常：后退参考平滑身体vx均值约-0.2236m/s。相同当前底座，用motion教师在后退参考起态实测vx-0.2725，从站立起步-0.2905；改用velocity编码器则为+0.0922/+0.2590。故不是参考方向或测量符号错误，且解码器具备后退执行能力。
- `backward_probe_20261029_r1` 使用固定bootstrap候选，与之前66秒候选一致；`backward_sourceprobe_20261029_r1` 仅恢复原五动作motion编码器/decoder，velocity编码器保持不变，student从站立的vx变为-0.1897，但yaw约+0.3575rad/s。说明新接口和持续更新后的共享底座组合对未覆盖状态不稳健；不能将此恢复操作单独认作完整修复。
- 旧监督只在正常教师轨迹上配对，命令与轨迹历史高度相关。随机行拟合误差很小不能证明同一状态下修改命令会产生正确方向。实际评测发现候选常偏向同一种运动，必须同时测试正、负、零指令。

## 已否决尝试

1. `backward_dagger_20261029_r1`：当前底座，100环境教师/学生混合采集2040步，3000在线+1000离线编码器更新。25/66秒评测仍有向前与跌倒，否决。
2. r2：恢复已验证底座、强制站立重置、均衡后退样本；保存前torch.equal在CPU/CUDA间比较错误，未保存候选。检查已修复，r3完整重跑。
3. `backward_dagger_20261029_r3`：90400样本，3000在线+1000离线更新，逐项确认非velocity policy参数与原五动作源完全相同。25秒seed0连续测试8/8不倒，但后退阶段均值只有-0.0217m/s，部分环境近零/微正，前进也近零。不能认作有效指令控制。
4. `backward_physical_20261029_r1`：真实MuJoCo PPO仅训练后退指令；256环境、80轮×32步，冻结解码器。最终candidate_80.pt SHA256 f994ade74fba8dcbc7550ca70edf5f3528e0e37b731d7f93c42a7d343eddd932。从站立后退8秒无失败，vx均值-0.2347；25秒seed0后退阶段8环境均为负，范围[-0.1169,-0.0790]，8/8全程不倒。但前进指令阶段vx=-0.0732，零指令也后移，说明学到后退偏置，否决作为键盘策略。不能只依据后退阶段平均速度宣布修复。此反例验证了单一指令训练无法保证条件控制。

## 混合指令训练尝试（已结束，未通过）

`train_backward_physical.py --conditional`：相同物理状态分布内，每64步（1.28秒）随机重采样前进+0.3、后退-0.2、零速，观测和命令共同输入同一个velocity encoder。PPO更新使用每条转移实际的指令，按真实身体vx误差计算密集奖励，冻结原五动作decoder/motion编码器。不再用单指令优化或随机行拟合误差代替闭环方向验收。

有界运行 `logs/directional_physical_20261029_r1`，GPU1、tmux `sm-directional-physical`，256环境，240轮×32步，固定每20轮保存，不覆盖前述失败记录。最终仍需正/负/零指令同条件实测。原双卡30分钟训练已按原预算结束，未延长或热改。

## 最终定位与后退单项修复

混合指令训练240轮仍未通过：seed0/7前进阶段vx约-0.168/-0.165，后退约-0.305/-0.300，零速也后移，不能因后退为负就采用。所有失败候选保留，均未替换既有键盘权重。

对80轮检查点的同条件实测做纵向输入输出拟合：请求vx为+0.3、0、-0.2时，实际阶段均值分别-0.06083、-0.12257、-0.22437。响应斜率为正但明显偏小，且存在后移偏置。故问题不是负号被颠倒，而是训练后的速度编码器到共享decoder的映射没有按真实速度标定；教师轨迹内拟合、单指令训练或模型内奖励不能保证请求速度的零点和增益正确。状态与指令的覆盖缺陷进一步使闭环响应依赖历史。

`calibrate_velocity_input.py` 由上述实测拟合增益3.1495896、偏置0.4614334。内部纵向查询u=gain*requested_vx+bias，变换严格折入velocity编码器第一层纵向输入列和bias；不增加推理模块，不改变用户指令，不更改机器人、motion编码器或decoder，无动作覆盖。此处输入标定是经物理干预验证的修复，不能泛化成未验证指令范围的保证。

验证权重：`logs/directional_calibration_20261029_r1/backward_validated.pt`（与candidate.pt内容相同），SHA256 `5c720b00790c52ae3f60eeb9fdc9146e20d7e692ee8053a4c3c618568b36eff9`。来源80轮权重SHA256 `e292857364fdc7d57bd59d9cb211bf9696e9ea5590acc06f096b17c25d36aa48`；固定五动作底座SHA256 `c4ebad3cb5686a3d237591b4400fc32eb1c65d56c5af028ab07a63ede5bb554b`。模型仅用于仿真评测，不可直接resume旧优化器。

### 物理验证结果

25秒连续流程：0–3s零速、3–10s前进+0.3、10–15s零速、15–22s后退-0.2、22–25s零速。除各段首1秒，按原评测日志5Hz取均值；保留全部失败/重置记录。

| 测试 | 前进vx均值 | 后退vx均值 | 后退各环境vx均值范围 | 无跌倒/重置 |
| --- | --- | --- | --- | --- |
| seed0，8环境 | +0.32652 | -0.20628 | [-0.23827,-0.17378] | 8/8 |
| seed7，8环境 | +0.28788 | -0.22945 | [-0.26848,-0.19714] | 8/8 |

后退阶段平均侧向速度-0.0017/-0.0048m/s、偏航+0.0489/+0.0483rad/s。另以backward参考库从站立直接施加-0.2m/s，8秒全部无失败，去首秒平均vx=-0.19545、vy=+0.00616、yaw=+0.00179。参考起态直接后退8秒vx=-0.20778，无失败。详见 `logs/directional_calibration_probe_20261029_r1/probe.json`。

停止仍未达标：seed0/7各环境停止段平均平面速度约0.113–0.168m/s，不能用平均vx接近零掩盖往复晃动。前进存在约-0.19至-0.20rad/s偏航。左右转、小跑、扩大速度范围、噪声/扰动泛化未回归；未将此后退验证权重替换为整套键盘模型的默认权重。通过的范围是上述后退方向与速度响应，非完整键盘策略验收或步态风格验收。

### 文件和复现

- 服务器统一前缀：`/home/2025-Long.xh/HHD/data/supermimic/`。最终权重/两次25秒评测/summary/标定参数在 `logs/directional_calibration_20261029_r1/`，独立后退probe在同级 `directional_calibration_probe_20261029_r1/`。
- 源码入口：`scripts/experiments/bootstrap_keyboard_gaits.py`、`train_backward_physical.py`、`calibrate_velocity_input.py`；诊断入口 `scripts/evaluation/probe_backward.py`。运行使用键盘快照的PYTHONPATH，未修改原键盘工作树。
- 有界运行会话依次为sm-backward-probe、sm-backward-sourceprobe、sm-backward-dagger、sm-backward-dagger-v2/v3、sm-backward-physical、sm-directional-physical、sm-directional-calibration-test、sm-calibration-seed7、sm-calibration-standback；全部已结束。GPU1用于采集/训练/probe，GPU3用于评测，未停止其他任务。r2未产出权重，等待它的两个本任务会话已精确取消。
- 所用环境/依赖/资产均沿用专用环境和既有离线快照，未安装依赖或修改系统。失败记录与每20轮保存点均保留。

## 2026-10-29 18:03：验证权重连续录像

- 用户要求视频。固定同SHA的backward_validated.pt，GPU1，tmux `sm-backward-video`，PID1866413；运行 `logs/backward_video_20261029_r1/`，完整argv与PID见launch.json，入口as-run.py对应renderer提交748d67e。
- 录像25秒，seed0、模拟2环境展示env0，前进/停止/后退/停止；仅速度编码器驱动，无未来参考姿态输入、无reset，无动作覆盖。625帧、25fps、1280×720，结束原因requested_duration，无首次失败。没有挑选种子或剪掉停止晃动。
- 视频backward-verified.mp4，同名JSON保存速度/命令/轨迹与权重SHA；overview.jpg已查看1/6/12/18/21/24秒关键帧，ffprobe确认25秒625帧。本地副本在outputs/backward_video_20261029_r1/。

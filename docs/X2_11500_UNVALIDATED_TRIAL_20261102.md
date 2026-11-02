# 11500 实机准备姿态试跑（2026-11-02）

用户授权以**未验证人工监督方式**试跑，随后明确“先只切初始姿态，暂不接管策略”。机器人当时吊挂防跌、双脚承重，用户在急停旁。本记录区分软件检查、只读测量和实际电机动作，未把任何一项写成通过。

## 软件改动

- `scripts/deploy/x2_hardware_publisher.py` 新增 `safety_mode='manual_supervised_unvalidated_trial'`：要求 readiness 证据显式 `explicitly_unvalidated_trial=true`、四项验证（准备接管、命令合同、状态链、ready）**逐项为 false**、`unvalidated_items` 非空，consent 额外要求 `acknowledges_unvalidated_software=true`。字段为 true 反而拒绝，因此该通道不能伪装成已验证部署。`native_mc_stopped_verified` 允许按实记录。
- `src/supermimic/deploy/early_handover.py` 新增可选 `max_tilt_delta`：首个样本定义参考倾角，判据改为相对偏差；`max_tilt` 保留为硬上限。默认不启用，原有绝对判据行为不变。
- `scripts/deploy/x2_11500_worker.py` 透传 `max_tilt_delta`，并把“状态生产者/发布者数量”失败改成带实际计数的报错。
- 测试：发布器门禁 5/5 通过（含“未验证试验不能被改标成已验证”）；控制器的 6 项测试在 `lxh` 内用 `python-cpu` 通过（含静态倾斜参考）。本机 Python 无 numpy，控制器测试不在本机运行。

## 只读实测

- 面板退出后重新启动在线定位（`online_lio_20261102_hwclock_cpu045_r12`，无 relay、无 CPU 绑定），45 秒 2242 条：49.8 Hz，状态年龄 p99 25.8 ms、最大 51.2 ms，到达间隔 33 次 >30 ms、1 次 >100 ms、最大 244 ms，单会话、时间戳单调、`motor_output_enabled=false`。此前 r12 的 13 Hz 失败发生在面板占用 CPU6/7 期间，本次条件不同，不能互相替代。
- 躯干 IMU 独立测量倾角 0.3548 rad（约 20°），在线定位同窗中位 0.3225、最大 0.4152。**吊挂姿态本身即倾斜约 20°**，绝对倾角判据不适用于当前支撑。

## 试跑结果：未完成准备姿态

计划固定 11500 导出（`policy.pt` `4424f231…4c`）、NumPy 后端、SDK 准备增益、策略增益取自 profile，命令与测量边界取设备 URDF 范围（命令合同未验证，已写入 `unvalidated_items`），`max_tick_gap=0.30`、`max_target_step=0.15`（按 244 ms 缺口放宽）、30 秒硬上限；准备时长 10 秒。

| 轮次 | 结果 |
| --- | --- |
| r1 | 只读目标监控器自己订阅了控制话题，worker 的唯一订阅者检查失败；无电机命令 |
| r2 | 状态生产者挂了 r3 源码路径，`supermimic.deploy.online_inertial` 缺失而退出；无电机命令 |
| r3 | 首次发出准备目标后控制器立即锁存 `Tilt or angular speed outside contract` |
| r4 | 加入相对倾角判据后，控制器构造即拒绝：实测姿态越界 |
| r5 | 运行脚本的补丁控制器路径指向运行目录，`early_handover.py` 在试验根目录，未启动 |
| r6 | 用户授权扩边后首次起步：准备轨迹以 0.335 rad/s 正常运动，随即锁存 `policy packet stale or future` |
| r7 | 把三个时效门限统一放宽到 0.35 s 后，发布器契约校验直接拒绝 `Invalid timeout`；零电机命令 |

r3 的实际动作：8.0 秒开始、10.7 秒结束，峰值 3.47 rad/s（左肘），末端与启动姿态的偏差从 0.951 rad 增大到 2.017 rad。这是故障锁存后发布器转入零刚度阻尼、手臂在吊挂下自行下垂，不是准备轨迹；准备轨迹的速度上限是 0.35 rad/s。角速度全程最大仅 1.09 rad/s，**故障来自倾角判据**：我把 `max_tilt` 设为 0.35，而测量稳态倾角是 0.3548，等于每帧都越限。这是门限设置错误，不能记作策略或动力学结论。

r4 拒绝原因（实测关节值对照）：`left_knee_joint=-0.0659`，URDF 与模型下限均为 0.0；`waist_pitch_joint=0.3479`，URDF 上限 0.314（模型上限 0.4887）。膝关节读数超出所有已记录范围，属编码器零位/方向未验证。未放宽测量边界去容纳该读数。

## 用户授权扩边后的尝试（r5–r7）

用户选择放宽测量边界以容纳实测姿态。构建器新增 `--snapshot` 与 `--extend-margin`：只在越界关节上把边界扩到实测值外 0.05 rad，并把每个关节的原值、实测值和扩展值逐项写入计划与 `unvalidated_items`。r6 的扩展项为 `left_knee_joint` 下限与 `waist_pitch_joint` 上限。

r6 是本轮第一次真正起步的准备动作：关节以峰值 0.3356 rad/s 向启动姿态移动（准备轨迹上限 0.35 rad/s），行程约 0.085 rad，随后锁存 `policy packet stale or future`。原因是 worker 的包时效上限硬编码 50 ms，而实测状态链有 33 次 >30 ms、最大 244 ms 的到达缺口；`validate_policy_packet` 在进入控制器前就拒绝。

r7 把 `max_state_age`、`max_target_age`、`max_tick_gap` 统一设为 0.35 s 并改为计划驱动，但发布器门禁中仍硬编码 `0<max_state_age<=0.05`，两个进程都在契约校验阶段退出，**未发送任何电机命令**。该 50 ms 上限是传输层既有设计约束，放宽它需要修改门禁常量，本轮没有继续改。

## 当前状态与证据

四组命令话题发布者 0，无本项目进程，原生 MC `Idel`；关节在 4 秒内漂移 0.0。手臂当前偏离参考启动姿态约 2 rad，膝关节读数仍为负。

PC2 容器路径：`/home/agi/lxh-supermimic/trials/11500_unvalidated_20261102_r1/`（`run-prep-r2`–`r7`、`contract-dryrun`）；在线定位会话在各 `run-prep-r*/state/`；关节记录 `run-prep-r3/joint-recording.json`；版本化脚本：`scripts/deploy/build_x2_11500_trial.py`、`run_x2_11500_prep.sh`、`run_x2_11500_trial.sh`、`record_x2_joint_states.py`、`probe_x2_state_rate.py`、`check_x2_trial_contract.py`。运行前需把打过补丁的 `early_handover.py` 放到试验目录，脚本用 `PATCHED_CONTROLLER` 或 `$trial/early_handover.py` 覆盖到源码副本。

## 仍未解决

1. 编码器零位/方向与真实关节范围未验证，膝关节读数在文件范围外。
2. 准备姿态动力学仍未通过；两次电机尝试都没有完成从当前姿态到启动姿态的过渡。
3. 命令合同仍是设备 URDF 范围替代，HAL 实际 PD 目标允许范围未确认。
4. 策略接管未执行；状态链的人工监督路径未验收。
5. 准备动作能否在 50 ms 时效约束内连续运行，取决于那 33 次/45 s 的状态缺口；这是测量链问题，不是策略问题。

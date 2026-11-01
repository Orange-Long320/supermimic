# X2 11500 部署问题复核（2026-11-01）

状态：**未通过实机部署验收**。本轮没有发布真实关节命令、上电、切换模式或启动原生 MC。所有机载程序均在 PC2 的 `lxh` 内执行。

## 已确认与已修改

- 11500 输入是 10 帧真实观测历史，观测 990 维、参考 729 维、输出 31 维。时间缺口后不得复制上一帧填充历史。
- 核验并停止旧 r1–r11 的 21 个隔离测试残留进程：精确匹配私有路径、`ROS_DOMAIN_ID=197` 和隔离话题前缀；保留原日志。11 个旧测试发布器仍各有约 8%–16% CPU 的累计占用读数。其存在会增加负载，但本轮未做同窗对照，不能定量归因全部断流。
- 新增 `tests/run_isolated_x2_worker.py`，限定隔离域、最长 180 秒主动状态窗口，为测试子进程建立私有进程组，在成功、异常和中断后回收；子进程启动失败提前退出。它不能用于真实电机测试。
- 工作进程状态订阅队列由 1 改为 10，仍保留 50 ms 包年龄和显式 31 ms 历史间隔保护。新的 r3 合成源最大实际间隔 27.470 ms，但工作进程出现 40.003 ms 历史间隔，支持接收链漏样本，不能认定这次是源发布迟到。
- FK/参考加载和预热移到状态订阅之前，避免首回调阻塞期间积压输入。10 帧历史仍来自后续真实输入；预热不算历史。
- 新增只读 `audit_x2_live_state.py`，分别比较训练模型和设备 URDF 范围，记录时间间隔及拒绝，不裁剪读数、不加拟合零偏。

快照：`d1298e3`（隔离测试回收）、`c3d8fc6`（队列与只读审计）、`07e6504`（订阅前预热）。

## 关节读数复核

30.014 秒原子状态订阅共 1341 包，1323 包通过包校验，18 包因陈旧/未来时间被拒绝。范围统计也包括结构完整但陈旧的包，不能把全部统计样本视为新鲜输入。

| 关节 | 实测 rad | 训练模型范围 rad | 设备 URDF 范围 rad |
| --- | ---: | --- | --- |
| waist_pitch | 0.349100 | ±0.488692 | ±0.314 |
| left_shoulder_roll | -0.121087 | [-0.061087, 2.98451] | [-0.061, 2.993] |
| right_shoulder_roll | 0.098653 | [-2.98451, 0.061087] | [-2.993, 0.061] |
| left_wrist_pitch | -0.573167 | ±0.523599 | ±0.558 |

腰部在训练范围内，但超设备文件范围。此前已查到 HAL 解算上限为 0.3491，不能当作已标定机械角或扩大硬件控制范围的依据。双肩及左腕仍超两套范围。厂商[活动范围页](https://x2-aimdk.agibot.com/zh-cn/latest/about_agibot_X2/joint_name_and_limit.html)当前列旗舰版腰 pitch ±18°、腕 pitch ±30°；不是所有 URDF 限位都等同厂商保证活动空间。

本轮读取 PC2 的 `factory_calibration_params.yml`：顶层是 waist/chest IMU、相机和胸部雷达标定，没有关节编码器零位项。不能从这个文件得到所需关节零位修正。需要这台设备的厂商关节零位校验、原始电机反馈及映射依据；不能以拟合 IMU 的角度代替标定，也不重复应用 HAL 已有方向映射。

## 连续性复核

- 实机 30 秒只读窗口：最大状态源间隔 200 ms、发布间隔 184.657 ms、接收间隔 179.254 ms；仍失败。
- 隔离 r3：源最大间隔 27.470 ms、零次超过 30 ms；工作进程间隔故障 40.003 ms。保留失败证据。
- 队列修改后的 r4 因陈旧包失败；预热移出回调后的 r5 仍发生陈旧包锁存。r4/r5 源最大间隔分别为 22.908/27.381 ms，均没有源间隔超过 30 ms。改动没有证明长期连续通过，不能仅凭源正常宣称接收端正常。
- 上述隔离测试均在 domain 197 的测试话题，使用合成输入与合成限值；不是实机物理验收。新测试进程退出后已核查无遗留。

### 180 秒真实输入影子推理

`stream-followup-affinity-180s-r2` 在同一 r17 定位会话使用原 11500 NumPy FP32 权重，180.003 秒收到 8980 状态、8846 帧推理，12 次流停止/陈旧重置、3 次历史间隔超限。最大源间隔 120 ms、发布间隔 145.20 ms、接收间隔 145.70 ms、推理帧间隔 299.95 ms；单帧处理最大 10.31 ms、推理最大 8.18 ms。全部有效推理帧仍有上述四处文件范围差异。`fatal=null` 和进程正常退出不等于连续性通过。

本窗口临时把本任务关节转发器、IMU 转发器、Python 桥的主线程分别放在 CPU 2/3/4，影子推理在 CPU 5；未修改其他进程或系统配置。结束后核对三个主线程均恢复原 CPU 0–7 集合。它不是核心独占、实时调度或永久部署配置。

按本次首末推理状态时间核对同窗日志，桥的拒绝计数增加项为 `IMU acquisition gap=48`。这是重复尝试的拒绝计数，不是 48 次独立源断点。**C++ 原生接收器在进入 Python 之前已记录两次头时间间隔 30.831466 ms 和 34.096846 ms**，对应接收年龄仅约 0.394 ms。这把剩余问题定位到 Python 策略之前的原生 IMU 发布/传输/接收链路；现有证据不能区分发布端缺样与 C++ 订阅端丢样，也不能认定传感器硬件故障。不可通过给这些缺口插造历史或放宽积分门限来宣布解决。

首轮影子检查因未包含 `stream_x2_policy_probe.py`/`deferred_policy_log.py` 的代码包而未开始有效采集；失败日志保留。补齐完整部署模块并核验实际入口后，本节 r2 才是实际 180 秒测量。最终相关 CPU 单元测试 11 项通过（历史间隔故障锁存与原子包验证），含 31.489 ms 缺口必须保持故障的回归测试；它们不构成物理验收。

机载证据根目录：`/home/agi/lxh-supermimic/readiness_work_20261031_r1`。本地小报告：`outputs/x2_readiness_20261101_followup_r1/`。当前定位会话为 `online_lio_20261101_r17/live-r1`，session `6ba900fb997a4661a9c72fad102e7dfd`；状态仍标记 `control_ready=false`、`motor_output_enabled=false`。

## 当前可以运行的只读复核指令

在 Mac 执行；订阅已有原子状态，不启动电机发布器：

```bash
ssh x2-robot 'podman exec -i lxh bash -s' <<'SH'
set -eo pipefail
source /opt/ros/humble/setup.bash
source /home/agi/aimdk/install/setup.bash
base=/home/agi/lxh-supermimic
run=$base/readiness_work_20261031_r1
export PYTHONPATH="$run/state-shadow-followup-r1/src:$base/python-cpu:${PYTHONPATH:-}"
/usr/bin/python3 "$run/state-shadow-followup-r1/scripts/deploy/audit_x2_live_state.py" \
  --release "$base/releases/startup_710c43e_20261031_r1" \
  --out "$run/live-audit-$(date +%Y%m%dT%H%M%S).json" --seconds 180
SH
```

## 电机部署命令的条件

目前没有可核准的实际限值/增益计划、连续性通过证据、独立硬件失联保护或当前上电确认，因此**不存在本次已验证可执行的实机 11500 启动命令**。不能使用 `synthetic-plan.json`、`synthetic-evidence.json`、`synthetic-consent.json`，不能将其中布尔值改成 true 充当实机证据。

用户已确认没有关节零位校验结果或校准流程。需要厂商提供：本机型/固件的关节零位只读自检及校准流程；腰部原始反馈到虚拟关节的解算/限幅依据；双肩和左腕超文件范围的判定；原生雷达 IMU 头时间缺口的发布端诊断。未自行修改 protected 标定、HAL 参数或重启厂商驱动。

接口顺序为：`x2_hardware_publisher.py --plan PLAN --consent CONSENT` → `x2_11500_worker.py --plan PLAN --consent CONSENT` → 同一 worker 入口增加 `--request-policy`。此处只记录已核对的接口，不是现场执行指令；真实 PLAN/CONSENT 尚未生成。原生 MC 必须保持关闭，最终上电仍需用户对具体现场方案确认。

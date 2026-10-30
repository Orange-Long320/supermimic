# SuperMimic 站立策略 EMA 同条件对照

用户要求站立与停止训练沿用 SuperMimic 世界模型与可微想象更新，禁止继续 PPO；增加一份仅用于评测的策略 EMA，与同次训练的在线权重比较。

## 接入范围

- 复用 `train_supermimic_stop.py` 与 `supermimic_stop_imagination.py`：真实物理采集、32步监督拟合动力学、独立环境模型误差门检、冻结世界模型后的32步可微想象/BPTT、critic 的 GAE 目标拟合。没有 PPO 概率比或裁剪策略目标。
- 移动底座保持冻结；仅训练站立 expert 的 encoder、decoder、6维反馈。数据采集和想象始终使用在线策略，EMA 不参与采样、模型拟合、奖励或 critic。
- 默认 `--ema-decay .99`。只在参数实际变化且动作变化检查接受的 BPTT 更新后更新一次。首次有效更新直接复制，之后 `ema = .99 * ema + .01 * online`；被拒绝或无效更新不计数。尺度等 buffers 直接复制，不平均。
- 只复制站立模块，不占用随机数序列。在线检查点保留 EMA 状态与计数；`train/ema/candidate_N.pt` 是普通策略加载器可读的评测导出，不包含新的优化器续训状态。
- 在线 `.ready` 在完整两份文件保存完后才发布。控制器核对配对 SHA256、迭代/来源/配置/更新计数，检查世界模型及所有非站立参数逐张量相等。至少两次有效更新才进行 EMA 对照。
- 每份策略都完成 full0/full7（66秒）、hold20、cold0/cold7、extended42、direct58 七组相同测试；不因另一份失败而省略场景。核对初始物理状态和策略观测 SHA256，以及种子、场景、时长、指令序列、数值精度和关节顺序。保留每组原始轨迹、物理门限和完整启动命令。
- 输出 `comparison.json`，两份权重分别按原有严格门限判断；不自动选赢家或替换键盘策略。只有某份全部通过时才生成对应 `accepted_online.pt` 或 `accepted_ema.pt`。

## 已验证与待执行

本地只执行静态编译与 diff 检查。服务器专用主环境运行 `tests/test_standing_ema.py`，9项通过，覆盖平均数值、拒绝更新跳过、首更新复制、buffer处理、随机数/梯度/在线参数隔离、导出保存加载，以及冻结参数或配对来源被篡改时拒绝比较。该测试不证明策略效果。pytest 仅有 `/dev` 缓存目录不可写警告，不影响测试结果。

服务器 Ubuntu 主机 `amax-Super-Server` 已核验；2026-10-30 本次连接可见4张RTX3090/驱动580.178.04。专用解释器及HHD真实路径沿用已核验主环境，未变更依赖或系统设置。上一轮非EMA容量试验的20轮训练实际完成5次actor更新，完整物理验收未通过；它不是本轮EMA对照结果。

计划首轮有界验证：GPU2，128环境，20轮训练上限、训练最多0.25小时，EMA=.99。该轮主要检查真实模型更新、反传、EMA与配对验收全流程；有效actor更新可能远少于20次。EMA半衰期约69次有效更新，不以这轮短试证明收敛或EMA长期优劣。

- 独立服务器源码：`/home/2025-Long.xh/HHD/data/supermimic/logs/keyboard_supermimic_stop_ema_20261030_source_r1`
- 规划输出：`/home/2025-Long.xh/HHD/data/supermimic/logs/keyboard_supermimic_stop_ema_20261030_r1`
- 规划会话：`sm-supermimic-stop-ema-r1`
- 部署差异：复制既有 `keyboard_supermimic_stop_20261030_source_r2` 的完整服务器runtime（对应d20a66b及其as-run SHA清单），再显式上传本轮训练/控制器/EMA helper/同条件评测入口。本地src缺少该隔离runtime的locomotion模块，不能声称整份本地src直接部署。原隔离源码和产物不热改。来源与部署文件SHA保留在本轮source/controller记录。

启动前保存本地Git快照并再次检查GPU占用及进程所有者；实际会话、PID、提交号、训练与对照结果随后补录。此处规划不代表已经启动或完成。

# 11500 骨盆输入与实机部署前核对（2026-10-31）

后续修订：PD 目标角与实测机械角范围不能直接等同；本页的目标越界数字保留为历史几何统计，不能单独用于判定策略异常。状态/目标范围分离、启动姿态及设备 FK 修正见 [后续处理记录](X2_STARTUP_ALIGNMENT_20261031.md)。

用户要求接入骨盆位置并部署 11500 checkpoint，随后明确要求：**实机部署之前必须取得用户同意**。本轮未向机器人写入新文件、启动容器或控制程序、切换控制权、发送电机命令。只读访问 PC2 文件，并在项目服务器核对已有录包；后续不能把本次部署请求当作已获得最终部署同意。

## 骨盆输入的实际状态

已有 `scripts/evaluation/shadow_x2_localization.py` 使用 `pelvis_from_imu` 将定位和同期腰部 FK 转换为骨盆位姿，再调用选定源码 `ac6c755` 的 `SkillGoal.goal(position, quat, joints, 'stand')`。位置经 FK 进入目标末尾 9 维锚点特征中的位置误差；不是给 990 维观测追加 3 维。原模型输入仍为 990 维观测、729 维目标，输出 31 维。本轮没有修改权重、观测合同、零位、参考或限位，也没有增加在线定位发布器。

这条链路目前只有服务器离线录包验证，没有完成 PC2 在线定位到控制的闭环。详见 [骨盆转换和影子推理记录](X2_SHADOW_LOCALIZATION_20261031.md)。

## 本轮独立核对

- 本地修改前源码为 `b5d8385`，工作区干净。新增 `scripts/evaluation/audit_x2_deployment_inputs.py`，只读文件和计算 FK，不推进仿真或创建机器人通信接口。
- 重新读取 PC2 的 `policy.pt`，SHA256 仍为 `4424f231fc1896da098b2fca366e8522df6791596638b2b9257ba682520bbe4c`，对应选定 11500 checkpoint `ff1f91d33631530d9d7f90f9f0675359fa1bff18445df43449312b9cb85db34b`。
- 读取 PC2 现有 `/home/agi/aimdk/extra/mc-rl/x2_urdf/X2_URDF-v1.3.0/x2_ultra.urdf`，SHA256 `f84fcd22187d7f3d9730f36b06b39e49fb6a59cd8b1d26bcb5a817dfd300b40a`。这是设备上的模型文件，不等于经过实物测量的硬件安全限位。
- 与训练模型在骨盆坐标系下的零关节姿态比较：最大关节轴方向差 0.3424°、最大关节锚点位置差 5.19 mm、限位最大差 0.27015 rad。该文件比较不能证明实机编码器零位和方向已经校准。
- 用同一批 723 帧已有诊断输出重新按 PC2 URDF 核对：实测腰俯仰、双肩侧摆和左腕俯仰超出该 URDF 范围。feedback 和 zero 两种反事实动作历史的输出各有 723/723 帧包含越界目标。
- feedback 最严重目标为左踝俯仰 2.56374 rad，PC2 URDF 范围 `[-0.803, 0.453]`，超出上限 2.11074 rad；zero 最严重为右踝俯仰 2.41190 rad，超出上限 1.95890 rad。没有裁剪或下发这些输出。

仅只读查看了 PC2 现有 mc-rl 的状态回调和启动流程，未运行该程序。它是另一种 29 动作/500 维观测策略；不能将其状态机、增益、权重、标定或 README 中的实机结果自动视为本项目 31 动作策略的验证。其回调按名称直接读取 `JointState.position`，这不足以证明 SuperMimic 的物理零位和方向正确。

## 运行与复现

服务器独立目录：

```text
/home/2025-Long.xh/HHD/data/supermimic/logs/x2_pelvis_deployment_gate_20261031_r1
```

使用主项目专用 Python，`CUDA_VISIBLE_DEVICES=''`，未使用 GPU。脚本第一次检查发现 URDF 含 `base_link→pelvis` 固定祖先，初版假定 pelvis 为整棵树根而退出，空目录 `audit/` 保留。修正为从真实根遍历并统一转换至 pelvis 坐标后，完整 723 帧核对成功，报告为 `audit-r2/report.json`；该成功仅指审计执行完成，部署门槛未通过。

复现参数（输出须使用新目录）：

```bash
CUDA_VISIBLE_DEVICES='' /home/2025-Long.xh/HHD/venvs/supermimic/bin/python \
  audit_x2_deployment_inputs.py \
  --profile /home/2025-Long.xh/HHD/data/supermimic/assets/robots/x2_ultra/profile.json \
  --urdf robot-pc2.urdf \
  --frames ../x2_shadow_readonly_20261031_r1/shadow_diagnostic_final/frames.jsonl \
  --shadow-report ../x2_shadow_readonly_20261031_r1/shadow_diagnostic_final/report.json \
  --out NEW_OUTPUT_DIRECTORY
```

本地小型报告保留于 `outputs/x2_pelvis_deployment_gate_20261031_r1/report.json`，原始大记录仍在服务器。脚本绑定模型哈希和关节顺序，并记录输入文件 SHA256。

## 尚未完成

当前实测姿态与训练参考存在明显差异，不能据此直接断定是零位、方向还是启动姿态问题；需分别核对。骨盆在线输入、实时延迟、硬件标定和独立故障处理也未通过验收。不能以接入骨盆位置或输出有限为由放行电机。准备好具体部署内容、启动姿态、控制和退出流程后，先交用户审阅并等待明确同意，再进行实机部署。

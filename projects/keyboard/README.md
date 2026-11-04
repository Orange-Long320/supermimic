# X2 键盘项目

范围：站立、前进、后退、慢跑、原地转向、停止，以及对应状态估计和明确授权的实机准备。
保留已有世界模型和可微想象训练路线，不新增 PPO 训练。

## 代码与入口

| 内容 | 所属工作区 / 位置 |
| --- | --- |
| 当前参考动作键盘与部署 | `keyboard`：`src/supermimic/control/`、`src/supermimic/deploy/`、`scripts/deploy/` |
| 速度接口与训练实现 | `keyboard-training`：`src/supermimic/locomotion/`、对应训练循环 |
| 历史五动作训练控制脚本 | `keyboard`：`scripts/experiments/*keyboard*`；须读取 `--source` 指向的实际版本 |
| 评测与回放 | 各工作区的 `scripts/evaluation/`；按检查点格式选入口 |

不搬迁现有运行入口，不修改模型、控制参数或正在运行的服务器包。
训练入口不能只因存在同名文件就认定可运行；先检查源码版本、实际导入路径和依赖。

## 当前结果入口

- [实机记录与限制](../../docs/X2_HARDWARE_DEPLOYMENT_20261103.md)
- [当前 IMU 容器接入记录](../../docs/X2_IMU_HARDWARE_STAGING_20261104.md)
- [部署基础](../../docs/X2_DEPLOYMENT.md)

保留用户报告及原始日志中的区别：目前不能据此前实机试验声明连续控制与安全验收通过。
源码分区不会改善或改变这些策略结果。

## 禁止跨入的工作

ACP、楼梯地形、楼梯接触目标与登顶奖励均在 `stairs` 工作区开发。
楼梯依赖升级不能安装到现有键盘环境，楼梯权重不能覆盖键盘发布物。

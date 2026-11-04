# 常用 X2 Ultra 模型与五动作参考

此目录保存项目常用的 X2 Ultra 配置、11500 步训练检查点，以及该次训练使用的五动作和收步参考。网格、检查点和动作数组通过 Git LFS 分发：克隆前安装 Git LFS，或在已有克隆中执行 `git lfs install` 和 `git lfs pull`。

| 用途 | 文件 |
| --- | --- |
| 常用机器人配置 | `model/profile.json`，31 关节 X2 Ultra；`xml_path` 指向同目录 `robot.xml` |
| 仿真模型 | `model/robot.xml`、`model/source.urdf` 和 `model/meshes/` |
| 常用检查点 | `checkpoint/ckpt_0011500.pt`，SHA256 `ff1f91d33631530d9d7f90f9f0675359fa1bff18445df43449312b9cb85db34b` |
| 动作参考 | `motions/`：前进、后退、左转、右转、小跑及各自收步，共 10 个 NPZ |
| 来源与核对 | `motions/manifest.json` 为原始运行清单；`SHA256SUMS` 记录此目录分发文件的哈希 |

模型来自服务器 `assets/robots/x2_ultra/`，基于 [NVIDIA SOMA retargeter](https://github.com/NVIDIA/soma-retargeter) 固定版本 `cbdb2b494a2af82137bf9b6f15122d60167281c4` 的 AgiBot X2 描述与网格。为便于迁移，分发副本将 XML 和 URDF 的网格路径及 profile 的源 URDF 路径改为相对路径；机器人参数未因此改变。原始运行清单中的服务器路径仅用于记录来源，不是可移植的文件路径。相关许可证保存在 `licenses/`，项目根目录 MIT 许可不替代第三方许可。

检查点和参考来自服务器运行 `five_gait_stops_5h_20261030_r2`。这些是实验材料；参考动作的运动学检查不证明物理稳定，检查点也不代表实机部署已经通过验收。实机操作仍需遵守项目的授权和验收要求。参考 [实验记录](../../docs/FIVE_GAIT_STOP_5H_20261030.md) 和 [部署记录](../../docs/X2_DEPLOYMENT.md)。

# 脚接触约束和因果未来窗口试验（2026-10-30）

## 方法及范围

继续使用冻结五动作跟踪 checkpoint `c4ebad3cb5686a3d237591b4400fc32eb1c65d56c5af028ab07a63ede5bb554b` 与参考 `8e7813254aa415246ef1a895bdc89350e2b1c134dab73114e7da55fa8f07dd63`。无训练、无 PPO、无辅助力和中途重置。此为单环境、无扰动探索，不证明泛化，不接入现有键盘或训练。

参考生成使用 `prepare_skill_phase_candidate.py`，周期起点70、终点110、循环两次、连接12帧，松键偏移0/14/28帧，松键后先保留10帧已发布的行走计划（200 ms），再75帧插值制动/收步，站立1000帧（20秒）。三份参考在最早松键前的关节及身体位置完全一致。`project_skill_contacts.py` 从双脚网格相对高度估计支撑，固定支撑脚踝目标及朝向，约束原脚网格地面高度，使用关节机械限位内的 IK；投影是运动学拟合，推断的支撑不代表真实接触力，未验证转矩/动力学可行性。原始参考的脚碰撞网格确实在站立附近距地面约7 cm，已核对几何和模型；不存在仅因脚几何名称映射导致的该高度读数。

首版仅锚定脚踝，因脚掌朝向变化导致收步期最大穿地约4.9 cm，已保留为失败候选 `logs/skill_contact_project110_20261030_r1/`。第二版约束脚掌朝向和网格高度后，三个候选的离线最大脚网格穿地分别小于0.001 mm、小于0.001 mm、约0.50 mm，最大硬限位超出约0.000004 rad。第一个和第二个支撑脚目标最大位置误差约0.72 mm和0.003 mm；第三个收步帧327约52.7 mm，根部高度调整已触及相对原参考 -0.18 m 边界，说明既定支撑目标和状态约束不能共同满足。第三个候选不合格；前两者也只通过这些有限几何检查，不作为动力学或训练验收。

## 未来窗口

评测期间松键之前使用继续行走分支，松键之后使用先保留10帧行走、再制动的分支。原有 anchor 观测沿用当前参考，策略不提前看到未来按键；已发出的200 ms目标先执行完，带来明确的停止响应延迟。三次实测中，松键前后针对同一时刻的9帧重叠未来目标最大绝对差均为 `5.96e-8`。此结论限于这三条预生成轨迹和该10帧接口，不是通用在线键盘调度器；不同按键、再次按下与任意时相尚未实现或验证。

## 实际单卡闭环

2026-10-30，服务器 GPU0 在启动前无计算进程，GPU1由其他用户占用。明确选择 `CUDA_VISIBLE_DEVICES=0`，服务器专用环境 `/home/2025-Long.xh/HHD/venvs/supermimic/bin/python`，50 Hz 控制、5 ms 物理步、原始根高/倾斜终止，噪声及域随机化关闭。评测在首个原始终止时停止，不重置。三个松键时相均到达约19.98秒站立，没有触发原始终止。

| 松键偏移 | 松键前沿初始身体朝向行进 m | 松键至站立位移 m | 站立净漂移 m | 站立平均水平速度 m/s | 末5秒平均水平速度 m/s | 回放双脚最小几何高度 m |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 0帧 | 1.160 | 0.447 | 0.263 | 0.0230 | 0.00009 | -0.0323 / -0.0279 |
| 14帧 | 1.230 | 0.209 | 0.0776 | 0.0106 | 0.00800 | -0.0259 / -0.0407 |
| 28帧 | 1.362 | 0.177 | 0.0820 | 0.0166 | 0.01481 | -0.0199 / -0.0278 |

回放脚底高度是基于50 Hz记录状态，在独立 CPU 原始网格模型做的地面几何重算；不等于引擎实际接触力或5 ms峰值。原网格的厘米级几何穿地和长站漂移未解决，不能将无终止称作稳定或物理接触成功。50 Hz采样的实际关节硬限位超出均为0，不能据此断言5 ms过程无超限。第一组后5秒几乎静止，但已经在站立期间累计移动约26 cm。

## 复现记录

服务器根 `/home/2025-Long.xh/HHD/data/supermimic/logs/`。原始候选 `skill_contact_raw110_20261030_r1/`、`skill_contact_rawoff14_20261030_r1/`、`skill_contact_rawoff28_20261030_r1/`；投影结果依次 `skill_contact_project110_20261030_r2/`、`skill_contact_projectoff14_20261030_r1/`、`skill_contact_projectoff28_20261030_r1/`；物理记录和汇总 `skill_contact_eval110_20261030_r1/`、`skill_contact_evaloff14_20261030_r1/`、`skill_contact_evaloff28_20261030_r1/`。源副本与日志在 `skill_contact_work_20261030_r1/`；CPU投影进程有已保留SSH会话，首轮 PID 2301495，另两项为顺序会话 PID 2302292 和 2304065。生成及评测使用项目代码路径 `PYTHONPATH=/home/2025-Long.xh/SSD/supermimic/src`、`SM_ROBOT=x2_ultra`，物理评测设置 `CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl`，缓存 `/home/2025-Long.xh/HHD/data/supermimic/cache`。项目主服务器源码未替换。

例如第一组物理命令（另外两组仅候选和输出目录替换）：

```sh
/home/2025-Long.xh/HHD/venvs/supermimic/bin/python /home/2025-Long.xh/HHD/data/supermimic/logs/skill_contact_work_20261030_r1/eval_skill_phase_candidate.py --candidate /home/2025-Long.xh/HHD/data/supermimic/logs/skill_contact_project110_20261030_r2 --checkpoint /home/2025-Long.xh/HHD/data/supermimic/logs/gait_five_midpreview_20261029_r1/checkpoint.pt --out /home/2025-Long.xh/HHD/data/supermimic/logs/skill_contact_eval110_20261030_r1
```

继续前需要重新设计第三时相的落脚与支撑转换，而不是放宽根高界限；还需追踪实际接触力、脚底穿透与滑动的原因，改善循环中相邻帧最大关节位置RMS变化约0.070 rad的突变，并以多种初始状态、动作方向和扰动验证。当前没有成功或训练输入验收。

# UniLab X2 Ultra 后端短测（2026-10-29）

本轮没有发现迁移到 UniLab CPU 仿真路径的速度收益。未执行完整 UniLab 训练，不能判断 SAC/FlashSAC 收敛速度或其他后端表现。

## 来源与环境

- https://github.com/unilabsim/UniLab ，固定提交 `fa577b265fcc185ee5615e0abe7c06da067aad35`，声明1.5.3。本地忽略目录 `outputs/unilab-eval-20261029/source` 保留源码。
- 当前源码tasks注册列表只有Go2 locomotion、G1 locomotion和G1 motion tracking，配置/资产未找到X2。先前在线API文档列出X2，不代表当前源码可直接启动X2任务。
- 服务器 `amax-Super-Server`，双路Xeon Silver4216，32物理核/64线程、4×RTX3090。GPU0/2键盘训练PID1889469/1889470始终保留，只使用空闲GPU3。
- 新独立环境 `/home/2025-Long.xh/HHD/venvs/supermimic-unilab`，Python3.12.13，按上游版本安装mjbatch-uni0.2.5、MuJoCo3.11.0，通用依赖使用清华源。没有安装完整UniLab学习器，未改主环境。完整包清单在r1目录cpu-environment.txt。
- 本项目GPU环境MuJoCo/MuJoCo Warp3.16.0。不同版本不保证数值行为等价。

## 方法与结果

入口 `scripts/experiments/probe_unilab_physics.py` 使用本项目make_env_cfg/Scene.compile编译X2 Ultra平地模型，38qpos/37qvel/31执行器，5ms物理步。profile默认关节位置、站立高度和同一控制目标。

相同XML、初态和无损物理参数侧文件供两端加载；质量、惯量、关节范围、armature、执行器和碰撞等指定数组在导出重载时校验。没有声称全部模型数组或跨版本动力学等价。XML SHA256 `d238509fdcd06af59577e945e6e20c02eae5ee765df167d9422027dac278d4e8`。

预热后10次计时，每次10个物理子步，报告中位数。每次恢复qpos/qvel，GPU同步，排除编译和初始化。GPU用10步CUDA graph，CPU用Batch.step(nstep=10)。未计入策略、观测、奖励、数据传输、学习、随机化和完整环境重置。warmstart等内部状态未每轮清零。

| 环境数 | CPU线程 | CPU耗时/10步 | GPU耗时/10步 | CPU物理步/秒 | GPU物理步/秒 | GPU/CPU |
| --- | --- | --- | --- | --- | --- | --- |
| 256 | 8 | 44.174ms | 7.241ms | 57,953 | 353,563 | 6.10 |
| 1024 | 16 | 93.718ms | 10.128ms | 109,264 | 1,011,046 | 9.25 |

两端输出有限。末端第一环境根高度CPU0.635249811m、GPU0.635039270m，差0.211mm，未验证跨后端动力学等价。只覆盖50ms站立片段，不能代表完整动作接触负载。未测最大CPU线程配置、8192环境、显存峰值或训练达标时间。

## 运行、失败与回档

服务器日志父目录 `/home/2025-Long.xh/HHD/data/supermimic/logs/`：

- `unilab_probe_20261029_r1/`：初始源码快照cbb62a1及bundle；独立环境清单。mj_saveLastXML因MjSpec.compile没有填充legacy LastXML失败，改为所属spec导出。导出空嵌套default导致两端加载均失败，并非UniLab独有错误。
- `unilab_probe_20261029_r2/`：删除无属性、无内容的空嵌套default后，重载校验发现XML六位有效数字截短armature，最大差2.5e-8。未视为速度结果。
- `unilab_probe_20261029_r3/`：保留无损物理参数后完成四组测量，四个日志及JSON分别为cpu-256、cpu-1024、gpu-256、gpu-1024。实际脚本保存在r1/probe-r3.py。本地结果在忽略目录 `outputs/unilab-eval-20261029/results/`。

实际CPU命令（另测1024环境/16线程）：

```sh
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 timeout 90 uv run --no-project --python /home/2025-Long.xh/HHD/venvs/supermimic-unilab/bin/python python /home/2025-Long.xh/HHD/data/supermimic/logs/unilab_probe_20261029_r1/probe-r3.py cpu --envs 256 --threads 8 --out /home/2025-Long.xh/HHD/data/supermimic/logs/unilab_probe_20261029_r3
```

实际GPU命令（另测1024环境）：

```sh
CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 timeout 180 /home/2025-Long.xh/SSD/supermimic/.venv/bin/python /home/2025-Long.xh/HHD/data/supermimic/logs/unilab_probe_20261029_r1/probe-r3.py gpu --envs 256 --out /home/2025-Long.xh/HHD/data/supermimic/logs/unilab_probe_20261029_r3
```

四组进程退出0，GPU3恢复17MiB/无计算负载，原键盘训练仍运行。未启动新长训或替换现有训练输入/策略。最终脚本和文档另行提交、bundle备份服务器。本轮脚本与文档为全部部署差异；主项目服务器源码和环境未变更。

后续若要比较完整训练，需先移植X2任务并核对语义，再比较同等动作质量的总耗时。目前不支持替换主线。

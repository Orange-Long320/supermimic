# 同算法 UniLab / mjlab 对照：准备与模型核查

目标是保留同一套 SuperMimic 世界模型、策略、BPTT和训练参数，对比两种环境实现。此前FlashSAC实验不回答这个问题。接口适配、状态/动作/观测/奖励/终止一致性验证和速度对照均尚未完成。

2026-10-29已核验服务器：GPU0/2有本项目进程，GPU3有其他用户进程，GPU1空闲。未停止任何进程，也未修改主环境或UniLab环境。

使用同一份准备好的X2场景（SHA256 `0c1fe840b648844a03ec2c3182da0c9da2c2b456c9d851c0d9370a454f433978`），分别由独立主环境MuJoCo3.16和UniLab环境MuJoCo3.11编译。两者均为38 qpos、37 qvel、31执行器、41刚体；时间步0.005秒、solver 2、integrator 3、迭代10/线搜索20一致。记录的质量、惯量、关节、执行器及接触参数数组差值均为0；但`geom_quat`最大差9.1611e-5，`geom_size`最大差6.5406e-6，涉及mesh碰撞几何，其中最大差来自`robot/body_25_collision`。

这些字段是编译后表示，不能仅凭字段差异判定实际碰撞网格或动力学不同；也不能宣称两版本的物理行为等价。下一步应核查编译后的网格在身体坐标系中的实际形状，并用相同状态和动作做连续转移对照，再实现/校验SuperMimic的观测、回放、奖励、终止和学习接口。

入口 `scripts/experiments/unilab_x2/audit_model_versions.py`；本地快照2afb570，bundle和完整数组/差异JSON保存在服务器 `logs/unilab_supermimic_parity_20261029_r1/`。该核查只读取模型、未积分仿真，不能代替实际训练与性能测试。

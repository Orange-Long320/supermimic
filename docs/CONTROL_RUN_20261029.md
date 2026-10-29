# 五动作双卡独立实验

用户要求先启动占用两张GPU的实验，随后要求先开一个。采用中性且真实的运行名 `control_r1_20261029`，不修改进程身份或伪造工作负载。

用户进一步明确目的是今晚保留两张卡供训练。启动计划：服务器 `2025-Long-robot`，GPU0/1，tmux `sm-control-r1`，运行目录 `/home/2025-Long.xh/HHD/data/supermimic/logs/control_r1_20261029`。独立真实训练暂定至北京时间2026-10-30 08:00，时长在启动时计算，训练计时器可能略超；结束执行66秒五动作及42秒停止评测。后续切换任务时需明确接替；不覆盖或恢复已暂停的原六小时主训练，不从其剩余预算扣除本次实验。持续运行不等于共享服务器的独占预约。

复用已验证的隔离源码 `keyboard_frozen_expert_20261029_source`（部署来源d97f8c7），固定motion expert，velocity horizon32、WM每更新20批、每卡8192环境及batch2048、持续配对监督、quiet-stop代价及原固定控制教师。起点为 `keyboard_rootfix_continue_20261029_r2/capacity/ckpt_latest.pt`，SHA256 `0844d0beb1c2967d117555987a01abae2c183478d2155e727221f8ea8dba5996`，使用既有双卡容量证据；该起点移动回归通过，但初始停止仍未达标。

启动前已核验四张RTX3090、专用解释器链接及数据挂载点；最近一次GPU0/1无计算进程。启动器须再次核对占用、固定输入和源码SHA，保存完整命令、环境版本、PID、日志与检查点。此记录为启动前checkpoint，尚未声明已运行或效果改善；实际状态以服务器controller/rank记录为准。

实际状态更新：用户随后认为站立训练可计作已有的一张卡，并要求只新增一个进程。复核站立任务已于21:46保存report.json/candidate_240.pt并退出，四张GPU均无计算进程，tmux无会话；未启动上述双卡计划。已向用户说明现状并请求选择单卡进程或双卡实验；未获得选择前不执行依赖该选择的启动。服务器未发现scontrol/squeue/qsub/bsub命令，未建立正式独占预约。启动前源码快照7ad7ee7已bundle备份到服务器logs，SHA256 `fd213e8f7ca803f80d375bb7c3d303701f009c739c6426bb0df9187d35c2b3cc`。

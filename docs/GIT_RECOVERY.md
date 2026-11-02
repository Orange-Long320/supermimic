# Git 快照与回档

本地源码仓库：`/Users/orange/project/supermimic`，开发分支为 `main`。

快照包含源码、配置、来源版本锁、数据清单和实验记录。数据、环境、资产、缓存、模型权重与训练产物留在服务器原目录，不放进 Git。原有明确允许的演示媒体保留。

完成实质工作阶段后提交；高风险实验性修改或长任务开始前先留一个回档点。每小时另检查一次，有稳定改动才提交，不创建空提交。未完成或未验证的研究使用 `checkpoint` 标记，不能当作已修复版本。

查看版本及某次变更：

```sh
git log --oneline --decorate -15
git show --stat <提交号>
git diff <提交号> -- <需要检查的文件>
```

需要恢复时先确定提交号和文件范围，保留当前改动，再恢复指定源码文件。回档不会自动回退服务器训练检查点，也不要在未明确范围时覆盖正在运行的服务器源码。

完整 Git 历史另外打包为 bundle，并备份至本项目服务器专用目录：

```text
/home/2025-Long.xh/HHD/data/supermimic/code-backups/git/
```

Bundle 经本地验证和传输后 SHA256 校验。即使原源码目录损坏，也可以从备份恢复到一个新的空目录，避免覆盖当前项目：

```sh
git clone /备份位置/已校验的文件.bundle supermimic-recovered
git -C supermimic-recovered log --oneline -10
```

已有暂存内容、Git 锁或正在编辑的文件由原操作继续管理。

首次回档点为 `ef72095`，标签为 `checkpoint-2026-10-28-before-support-redesign`。初始完整 bundle 为 `supermimic-initial-20261028.bundle`。服务器另保存 `server-deployment-ef72095.json` 与 `server-overlay-ef72095.tar.gz`，记录部署目录相对本地快照的差异和仅服务器存在的源码，避免把两处状态误当成完全一致。

曾创建每小时检查任务“SuperMimic Git 回档检查”（`supermimic-git`）；使用前请核实其当前状态。

# AutoDL 云服务器小白全流程教程（AE-CS 项目）

本教程面向第一次用 AutoDL 的同学，目标是：
1. 在 AutoDL 租到可用 GPU 实例
2. 拉取你的项目并运行 `run_cloud.sh`
3. 自动完成验证、训练、评估、打包
4. 把关键结果复制给我，我帮你分析

适用仓库：`AE-CS-`（当前已把脚本合并到 `main`）

## 0. 你将得到什么

运行完成后会自动生成：
- 训练日志：`artifacts/run_时间戳/logs/train.log`
- 验证日志：`artifacts/run_时间戳/logs/verify.log`
- 评估日志：`artifacts/run_时间戳/logs/eval.log`
- 指标文件：`results/eval_v9/metrics.json`
- 特征指标：`results/eval_v9/feature_performance.csv`
- 打包文件：`artifacts/run_时间戳/ae_cs_v9_时间戳.tar.gz`

你只要把这些内容中的关键部分发给我，我就能定位问题并给调参建议。

## 1. 在 AutoDL 租实例

1. 注册并登录 AutoDL，进入「我的实例」
2. 点击「租用新实例」
3. 选择配置：
- 计费方式：按量（先试跑最稳妥）
- GPU：建议先 `3090/4090/A5000` 这类 24GB 显存卡
- 镜像：优先选带 Python 的常用深度学习镜像
- 地区：优先网络稳定、价格合适的区
4. 创建后等待状态变为「运行中」

注意：实例「运行中」就计费，不用时记得关机。

## 2. 连接你的云服务器

你有两种常见方式：

1. 网页端 JupyterLab（最简单）
- 在实例页直接点「JupyterLab」
- 在 JupyterLab 里打开 Terminal

2. SSH（更稳定）
- 在实例页面复制 SSH 登录命令，格式类似：
```bash
ssh -p 端口 root@主机
```
- 在你本地终端粘贴执行，输入密码登录

如果你用 SSH 长时间训练，建议用 `tmux`/`screen` 防断线。

## 3. 拉取最新 main 分支代码

进入你希望放代码的目录后执行：

```bash
git clone https://github.com/luoyangxi0912/AE-CS-.git
cd AE-CS-
git checkout main
git pull origin main
```

如果你之前已经克隆过：

```bash
cd AE-CS-
git checkout main
git pull origin main
```

## 4. 上传或准备数据文件

你的默认数据文件名是：
- `hangmei_90_拼接好的.csv`

放置建议：
- 直接放到项目根目录（与 `train_cloud.py` 同级）

可选上传方式：
- JupyterLab 上传（简单）
- SCP / FileZilla（适合大文件或文件夹）

## 5. 一键跑验证 + 训练 + 评估

在项目根目录执行：

```bash
chmod +x run_cloud.sh
./run_cloud.sh
```

该脚本会自动做 6 件事：
1. 安装依赖
2. 记录环境信息
3. 检查 GPU
4. 跑 `verify_v9.py`
5. 跑 `train_cloud.py`
6. 跑 `evaluate.py` 并打包产物

## 6. 如何判断是否跑成功

重点看这几类信息：

1. 训练日志末尾（`train.log`）
- 是否出现 early stopping / best epoch
- `Val total loss` 是否整体下降

2. 评估日志末尾（`eval.log`）
- 是否显示 `Evaluation finished`
- 是否输出 `metrics.json`、`feature_performance.csv`

3. 结果文件是否存在

```bash
ls -lh results/eval_v9/
ls -lh artifacts/
```

## 7. 把结果发给我（最关键）

请按这个清单回传：

1. 训练日志最后 200 行
```bash
tail -n 200 artifacts/run_*/logs/train.log
```

2. 评估日志最后 200 行
```bash
tail -n 200 artifacts/run_*/logs/eval.log
```

3. 指标文件内容
```bash
cat results/eval_v9/metrics.json
```

4. 特征表现前 20 行
```bash
head -n 20 results/eval_v9/feature_performance.csv
```

5. 如果有报错，贴完整 traceback（从第一行到最后一行）

你把以上内容直接粘贴到对话里，我会给你：
- 结果解读（是否过拟合、欠拟合、掩码学习是否正常）
- 下一轮参数建议（如 `p_drop`、`learning_rate`、`epochs`）
- 必要时给你可直接替换的命令

## 8. 常见问题快速处理

1. 没检测到 GPU
- 先确认实例是否真的租的是 GPU
- 在终端执行：
```bash
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

2. 数据文件找不到
- 确保 `hangmei_90_拼接好的.csv` 在项目根目录
- 或改 `train_cloud.py` / 命令行参数里的 `--data_path`

3. SSH 断开导致任务停了
- 用 `tmux`/`screen` 跑脚本

4. 磁盘空间不足
- 清理旧 `artifacts/run_*` 目录
- 大文件迁移到网盘/本地备份

## 9. 可选：后台运行（推荐）

```bash
nohup ./run_cloud.sh > cloud_run.out 2>&1 &
tail -f cloud_run.out
```

这样即使断开 SSH，训练也继续。

## 10. 关机省钱（结束后务必做）

训练和结果下载完成后，回 AutoDL 控制台关机实例，避免继续计费。

---

## 参考（AutoDL 官方文档）

- 快速开始：https://www.autodl.com/docs/quick_start/
- GPU 选型：https://www.autodl.com/docs/gpu/
- 上传数据（SCP/FileZilla/JupyterLab）：https://www.autodl.com/docs/scp/
- 下载数据：https://www.autodl.com/docs/down/
- 环境与目录说明：https://www.autodl.com/docs/env/
- SSH 连接示例：https://api.autodl.com/docs/ssh/

# AutoDL 云服务器小白全流程教程（AE-CS 项目）

本教程面向第一次用 AutoDL 的同学，目标是：
1. 在 AutoDL 租到可用 GPU 实例
2. 拉取你的项目并运行 `run_cloud.sh`
3. 自动完成训练、评估、打包
4. 把关键结果复制给我，我帮你分析

适用仓库：`AE-CS-`（当前分支：`v10-tuning`）

## 0. 你将得到什么

运行完成后会自动生成：
- 训练日志：`artifacts/run_时间戳/logs/train.log`
- 评估日志：`artifacts/run_时间戳/logs/eval.log`
- 指标文件：`results/eval_v11/metrics.json`
- 特征指标：`results/eval_v11/feature_performance.csv`
- 打包文件：`artifacts/run_时间戳/ae_cs_v11_时间戳.tar.gz`

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

## 3. 拉取 V11 代码

**首次克隆：**

```bash
git clone https://github.com/luoyangxi0912/AE-CS-.git
cd AE-CS-
git checkout v10-tuning
git pull origin v10-tuning
```

**之前已克隆过：**

```bash
cd AE-CS-
git checkout v10-tuning
git pull origin v10-tuning
```

> V11 的代码修复在 `v10-tuning` 分支上。push 后再拉取。

## 4. 上传数据文件

数据文件名：`hangmei_90_拼接好的.csv`

放到项目根目录（与 `train_cloud.py` 同级）。

上传方式：
- JupyterLab 左侧文件面板拖入（简单）
- `scp -P 端口 hangmei_90_拼接好的.csv root@主机:~/AE-CS-/`

## 5. 一键训练 + 评估

```bash
chmod +x run_cloud.sh
nohup ./run_cloud.sh > cloud_run.out 2>&1 &
tail -f cloud_run.out
```

> `nohup` + `&` 确保断开 SSH 也继续跑。用 `tail -f` 实时看输出，`Ctrl+C` 退出查看（不会停训练）。

脚本会自动执行 5 步：
1. 安装依赖（tensorflow 2.10、faiss-cpu 等）
2. 记录环境信息、检查 GPU
3. 训练 (`train_cloud.py`，checkpoints_v11，100 epochs，patience=20)
4. 评估 (`evaluate.py`，输出到 results/eval_v11)
5. 打包全部产物为 tar.gz

## 6. 如何判断是否跑成功

### 训练日志（最重要）

看 `cloud_run.out` 或 `artifacts/run_*/logs/train.log`：

```
Epoch 1/100  LR=0.001000
  Train: total=0.6454  recon=0.4111  consist=0.4489  space=0.0191  time=0.0005
  Val:   total=0.1841  recon=0.1734  consist=0.0202  space=0.0009  time=0.0001
  [BEST] val_recon=0.1734
```

**V11 关键验证点：**
- `space` 应在 0.001~0.05 范围（V10 修复前 ≈0.0006）
- `time` 应 > 0（V10 修复前 ≡ 0.0000）
- 如果 space/time 仍为零，说明代码没有更新到 V11

### 评估日志

看是否输出了 `Evaluation finished` 和 metrics.json。

### 结果文件

```bash
ls -lh results/eval_v11/
ls -lh artifacts/
```

## 7. 把结果发给我

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
cat results/eval_v11/metrics.json
```

4. 特征表现前 20 行
```bash
head -n 20 results/eval_v11/feature_performance.csv
```

5. 如果有报错，贴完整 traceback（从第一行到最后一行）

你把以上内容直接粘贴到对话里，我会给你：
- 结果解读（是否过拟合、欠拟合、掩码学习是否正常）
- 下一轮参数建议
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
- 或改命令行参数 `--data_path`

3. SSH 断开导致任务停了
- 用 `nohup` 或 `tmux`/`screen` 跑脚本

4. 磁盘空间不足
- 清理旧 `artifacts/run_*` 目录
- 大文件迁移到网盘/本地备份

## 9. 关机省钱（结束后务必做）

训练和结果下载完成后，回 AutoDL 控制台关机实例，避免继续计费。

---

## 参考（AutoDL 官方文档）

- 快速开始：https://www.autodl.com/docs/quick_start/
- GPU 选型：https://www.autodl.com/docs/gpu/
- 上传数据（SCP/FileZilla/JupyterLab）：https://www.autodl.com/docs/scp/
- 下载数据：https://www.autodl.com/docs/down/
- 环境与目录说明：https://www.autodl.com/docs/env/
- SSH 连接示例：https://api.autodl.com/docs/ssh/

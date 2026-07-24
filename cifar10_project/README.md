# CIFAR-10 图像分类项目

这是一个完整的 CIFAR-10 图像分类项目，包含数据下载、数据处理、模型定义、训练、测试、预测、网页演示、checkpoint 保存和快速自测。

本次重新训练计划固定跑 `50` 轮，输出目录为 `checkpoints_50/`。模型仍使用 `WideResNet-28-10`，训练策略包括 `RandAugment`、`RandomErasing`、`MixUp`、`CutMix`、`label smoothing`、`SGD Nesterov`、`warmup + cosine` 学习率和 CUDA AMP。

## 详细文档

- `docs/project_overview.md`：项目整体说明
- `docs/model_and_parameters.md`：模型结构和参数
- `docs/data_training_testing.md`：数据处理、训练和测试流程

## 目录结构

```text
cifar10_project/
  cifar10_classifier/
    augment.py        MixUp、CutMix、label smoothing、soft cross entropy
    config.py         命令行参数和默认超参数
    data.py           数据下载、数据增强、DataLoader、镜像地址
    demo_server.py    浏览器点击预测演示
    download.py       单独下载数据集
    engine.py         训练、测试、checkpoint 保存和加载
    evaluate.py       加载 checkpoint 测试并保存混淆矩阵
    inference.py      推理阶段公共模型加载逻辑
    main.py           正式训练入口
    model.py          WideResNet 模型
    predict.py        单张图片预测
    scheduler.py      warmup + cosine 学习率
    utils.py          指标、随机种子、CSV 日志、参数量统计
    visualize.py      样本图、训练曲线、混淆矩阵
  data/               CIFAR-10 数据集，保留
  docs/               中文说明文档
  outputs/            可视化输出
  tests/              快速自测
```

## 运行环境

```bash
cd /home/zkf/pytorch-env/something/hw1/cifar10_project
```

当前虚拟环境已经有 PyTorch、torchvision、Matplotlib 和 Pillow。如需重新安装依赖：

```bash
/home/zkf/pytorch-env/bin/python -m pip install -r requirements.txt
```

## 数据集

数据已经下载并解压：

```text
data/cifar-10-python.tar.gz
data/cifar-10-batches-py/
```

如需重新下载，推荐 SJTU 镜像：

```bash
/home/zkf/pytorch-env/bin/python -m cifar10_classifier.download \
  --data-dir ./data \
  --mirror sjtu
```

## 重新训练 50 轮

```bash
/home/zkf/pytorch-env/bin/python -B -m cifar10_classifier.main \
  --epochs 50 \
  --batch-size 128 \
  --test-batch-size 512 \
  --workers 4 \
  --output-dir ./checkpoints_50 \
  --tta \
  --target-acc 90.0
```

输出文件：

```text
checkpoints_50/cifar10_wrn_best.pt
checkpoints_50/cifar10_wrn_last.pt
checkpoints_50/history.csv
checkpoints_50/training_curves.png
```

断点继续训练到第 50 轮：

```bash
/home/zkf/pytorch-env/bin/python -B -m cifar10_classifier.main \
  --resume ./checkpoints_50/cifar10_wrn_last.pt \
  --epochs 50 \
  --output-dir ./checkpoints_50 \
  --tta \
  --target-acc 90.0
```

## 测试模型

```bash
/home/zkf/pytorch-env/bin/python -m cifar10_classifier.evaluate \
  --checkpoint ./checkpoints_50/cifar10_wrn_best.pt \
  --tta \
  --output-dir ./checkpoints_50
```

测试后会保存：

```text
checkpoints_50/confusion_matrix.png
```

## 单张图片预测

```bash
/home/zkf/pytorch-env/bin/python -m cifar10_classifier.predict path/to/image.png \
  --checkpoint ./checkpoints_50/cifar10_wrn_best.pt
```

## 浏览器点击预测

```bash
/home/zkf/pytorch-env/bin/python -m cifar10_classifier.demo_server \
  --checkpoint ./checkpoints_50/cifar10_wrn_best.pt \
  --data-dir ./data \
  --port 8008
```

浏览器打开：

```text
http://127.0.0.1:8008
```

## 快速自测

这个测试不下载 CIFAR-10，只用随机张量检查模型前向、loss、反向传播、优化器和测试流程。

```bash
/home/zkf/pytorch-env/bin/python -B -m tests.smoke_test
```

# CIFAR-10 项目说明

这个项目用于训练 CIFAR-10 图像分类模型。代码按功能拆开，包含数据下载、增强、模型、训练、测试、预测和网页演示。

## 训练结果

已完成 100 轮训练，结果保存到：

```text
results/cifar10_100_epochs/
```

结果：

```text
第 100 轮测试准确率：96.87%
最佳测试准确率：96.96%
最佳轮次：第 89 轮
```

## 数据集

CIFAR-10 一共有 60,000 张 `32 x 32` 彩色图片，分为 10 个类别：

```text
airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
```

划分方式：

```text
训练集：50,000 张
测试集：10,000 张
```

本地数据目录：

```text
data/
```

数据没有上传到 GitHub，需要时用下载脚本重新获取。

## 主要代码

```text
cifar10_classifier/data.py       数据下载、增强和 DataLoader
cifar10_classifier/model.py      WideResNet-28-10
cifar10_classifier/engine.py     训练、测试和 checkpoint
cifar10_classifier/main.py       训练入口
cifar10_classifier/evaluate.py   测试入口
cifar10_classifier/predict.py    单张图片预测
cifar10_classifier/demo_server.py 浏览器演示
```

## 常用命令

进入目录：

```bash
cd assignments/03_cifar10_classifier
```

下载数据：

```bash
/home/zkf/pytorch-env/bin/python -m cifar10_classifier.download --mirror sjtu
```

训练 100 轮：

```bash
/home/zkf/pytorch-env/bin/python -B -m cifar10_classifier.main \
  --epochs 100 \
  --batch-size 128 \
  --test-batch-size 512 \
  --workers 4 \
  --output-dir ./checkpoints_50 \
  --tta \
  --target-acc 90.0
```

测试最佳模型：

```bash
/home/zkf/pytorch-env/bin/python -m cifar10_classifier.evaluate \
  --checkpoint ./checkpoints_50/cifar10_wrn_best.pt \
  --tta \
  --output-dir ./checkpoints_50
```

快速自测：

```bash
/home/zkf/pytorch-env/bin/python -B -m tests.smoke_test
```

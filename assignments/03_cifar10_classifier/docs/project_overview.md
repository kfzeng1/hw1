# CIFAR-10 项目说明

这个项目用于训练 CIFAR-10 图像分类模型。代码按功能拆开，训练、测试、预测和指标输出都有独立入口。

## 数据集

CIFAR-10 有 60,000 张 `32 x 32` 彩色图片，共 10 类：

```text
airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
```

划分：

```text
训练集：50,000 张
测试集：10,000 张
```

数据目录是 `data/`。数据不上传到 GitHub，需要时用下载脚本重新获取。

## 模型

模型使用 `WideResNet-28-10`。它是残差网络的一种变体，深度为 28，宽度系数为 10，适合 CIFAR-10 这类小尺寸图像分类任务。

## 主要代码

```text
cifar10_classifier/config.py      参数配置
cifar10_classifier/data.py        数据下载、增强和 DataLoader
cifar10_classifier/model.py       WideResNet-28-10
cifar10_classifier/engine.py      训练、测试和 checkpoint
cifar10_classifier/main.py        训练入口
cifar10_classifier/evaluate.py    测试入口，保存混淆矩阵和分类指标
cifar10_classifier/metrics.py     accuracy、precision、recall、F1
cifar10_classifier/predict.py     单张图片预测
cifar10_classifier/demo_server.py 浏览器演示
```

## 结果

已保留一份 100 轮训练结果快照：

```text
results/cifar10_100_epochs/
```

结果：

```text
第 100 轮测试准确率：96.87%
最佳测试准确率：96.96%
最佳轮次：第 89 轮
```

旧 `.pt` 权重已经删除。重新训练 50 轮时会在 `checkpoints_50/` 里生成新的权重、日志和训练曲线。

## 常用命令

进入目录：

```bash
cd assignments/03_cifar10_classifier
```

下载数据：

```bash
/home/zkf/pytorch-env/bin/python -m cifar10_classifier.download --mirror sjtu
```

训练 50 轮：

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

测试并保存分类指标：

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

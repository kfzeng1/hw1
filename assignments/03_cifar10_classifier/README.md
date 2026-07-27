# 作业 3：CIFAR-10 图像分类

要求：使用 PyTorch 自主设计并搭建 CNN 模型，完成 CIFAR-10 分类训练，测试集准确率达到 90% 以上，并输出多种分类评价指标。

## 代码结构

```text
cifar10_classifier/
  augment.py        MixUp、CutMix、label smoothing
  config.py         命令行参数
  data.py           数据下载、增强和 DataLoader
  engine.py         训练、测试、checkpoint
  evaluate.py       测试入口，保存混淆矩阵和分类指标
  metrics.py        accuracy、precision、recall、F1
  model.py          WideResNet-28-10
  main.py           训练入口
  predict.py        单张图片预测
  scheduler.py      warmup + cosine 学习率
  visualize.py      样本图、训练曲线、混淆矩阵
```

## 本次 50 轮训练结果

正式结果保存在：

```text
results/cifar10_50_epochs/
```

结果摘要：

```text
第 50 轮测试准确率：96.04%
最佳测试准确率：96.04%
最佳轮次：第 50 轮
macro F1：96.03%
weighted F1：96.03%
```

checkpoint 权重保存在 `checkpoints_50/`，体积较大，不上传到 GitHub。

## 下载数据

```bash
cd assignments/03_cifar10_classifier
/home/zkf/pytorch-env/bin/python -m cifar10_classifier.download \
  --data-dir ./data \
  --mirror sjtu
```

## 重新训练 50 轮

```bash
cd assignments/03_cifar10_classifier
/home/zkf/pytorch-env/bin/python -B -m cifar10_classifier.main \
  --epochs 50 \
  --batch-size 128 \
  --test-batch-size 512 \
  --workers 4 \
  --output-dir ./checkpoints_50 \
  --tta \
  --target-acc 90.0
```

训练输出：

```text
checkpoints_50/cifar10_wrn_best.pt
checkpoints_50/cifar10_wrn_last.pt
checkpoints_50/history.csv
checkpoints_50/training_curves.png
```

## 测试并输出指标

```bash
/home/zkf/pytorch-env/bin/python -m cifar10_classifier.evaluate \
  --checkpoint ./checkpoints_50/cifar10_wrn_best.pt \
  --tta \
  --output-dir ./results/cifar10_50_epochs
```

测试输出：

```text
results/cifar10_50_epochs/confusion_matrix.png
results/cifar10_50_epochs/metrics.json
results/cifar10_50_epochs/classification_report.csv
```

`metrics.json` 包含 accuracy、macro precision、macro recall、macro F1、weighted F1 等指标。

## 快速自测

```bash
/home/zkf/pytorch-env/bin/python -B -m tests.smoke_test
```

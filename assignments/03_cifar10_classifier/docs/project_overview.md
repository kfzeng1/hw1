# CIFAR-10 项目说明

本项目用于完成课程小作业中的 CIFAR-10 图像分类任务。代码按训练、评估、预测、可视化和网页演示拆分。正式结果保存在 `results/cifar10_100_finetune/`。

## 数据集

CIFAR-10 一共有 60,000 张 `32 x 32` RGB 图片，分为 10 个类别：

```text
airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
```

划分使用官方版本：

```text
训练集：50,000 张
测试集：10,000 张
```

本项目没有额外切分验证集。训练过程中的 `test_acc` 和最终评估结果都来自官方测试集。

## 模型

模型采用 `WideResNet-28-10`，包含残差连接和较宽的卷积通道。默认参数量为 `36,479,194`，用于保证模型容量足够达到 90% 以上测试准确率。

## 主要代码

```text
cifar10_classifier/config.py          命令行参数
cifar10_classifier/data.py            数据下载、增强、训练/测试 DataLoader
cifar10_classifier/augment.py         MixUp、CutMix、label smoothing
cifar10_classifier/model.py           WideResNet-28-10
cifar10_classifier/scheduler.py       warmup + cosine 学习率
cifar10_classifier/engine.py          训练、测试、checkpoint
cifar10_classifier/main.py            训练入口
cifar10_classifier/evaluate.py        测试入口，保存混淆矩阵和分类指标
cifar10_classifier/metrics.py         accuracy、precision、recall、F1
cifar10_classifier/result_summary.py  结果摘要打印
cifar10_classifier/predict.py         单张图片预测
cifar10_classifier/demo_server.py     浏览器演示
```

## 正式结果

本次最终结果保存在：

```text
results/cifar10_100_finetune/
```

结果摘要：

```text
最佳测试准确率：97.30%
最佳轮次：第 96 轮
第 100 轮测试准确率：97.25%
macro F1：97.30%
weighted F1：97.30%
测试集正确数：9730 / 10000
```

权重文件保存在本地 `checkpoints_100_finetune/`，体积较大，不上传到 GitHub。

## 常用命令

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
  --output-dir ./checkpoints_100_finetune \
  --tta \
  --target-acc 90.0
```

断点继续训练：

```bash
/home/zkf/pytorch-env/bin/python -B -m cifar10_classifier.main \
  --resume ./checkpoints_100_finetune/cifar10_wrn_last.pt \
  --epochs 150 \
  --lr 0.006 \
  --min-lr 1e-5 \
  --warmup-epochs 0 \
  --batch-size 128 \
  --test-batch-size 512 \
  --workers 4 \
  --output-dir ./checkpoints_100_finetune \
  --history-file history.csv \
  --tta \
  --target-acc 90.0
```

评估并保存指标：

```bash
/home/zkf/pytorch-env/bin/python -m cifar10_classifier.evaluate \
  --checkpoint ./checkpoints_100_finetune/cifar10_wrn_best.pt \
  --tta \
  --output-dir ./results/cifar10_100_finetune
```

查看结果摘要：

```bash
/home/zkf/pytorch-env/bin/python -m cifar10_classifier.result_summary \
  --result-dir ./results/cifar10_100_finetune
```

快速自测：

```bash
/home/zkf/pytorch-env/bin/python -B -m tests.smoke_test
```

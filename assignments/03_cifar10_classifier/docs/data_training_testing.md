# 数据处理、训练和测试

## 数据下载

项目使用 `torchvision.datasets.CIFAR10` 读取数据。本地没有数据时会下载，已有数据时直接读取。

支持的下载源：

```text
official, sjtu, oneflow, baidu, brainchip
```

推荐命令：

```bash
/home/zkf/pytorch-env/bin/python -m cifar10_classifier.download --mirror sjtu
```

数据目录：

```text
data/cifar-10-python.tar.gz
data/cifar-10-batches-py/
```

数据集不上传。

## 数据增强

训练集使用：

```text
RandomCrop(32, padding=4)
RandomHorizontalFlip()
RandAugment(num_ops=2, magnitude=14)
ToTensor()
Normalize(mean=(0.4914, 0.4822, 0.4465),
          std=(0.2470, 0.2435, 0.2616))
RandomErasing(p=0.25, scale=(0.02, 0.15), ratio=(0.3, 3.3))
```

测试集只做 `ToTensor()` 和 `Normalize()`。

训练循环中还使用：

```text
MixUp alpha = 0.2
CutMix alpha = 1.0
Label smoothing = 0.1
```

因此训练 loss 使用 soft-target cross entropy。

训练日志中的 `train_acc` 是在增强后的 batch 上按原始 hard label 粗略统计的。由于训练阶段启用了 MixUp、CutMix 和 RandomErasing，这个数值主要用于观察训练是否在推进，不适合和测试准确率直接比较。

## 默认训练参数

```text
epochs = 100
batch_size = 128
test_batch_size = 512
optimizer = SGD
learning_rate = 0.1
min_learning_rate = 1e-5
warmup_epochs = 5
momentum = 0.9
weight_decay = 5e-4
nesterov = True
grad_clip = 5.0
AMP = CUDA 上启用
seed = 42
```

学习率策略：

```text
前 5 个 epoch 线性 warmup
之后使用 cosine decay
```

训练命令：

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

训练输出：

```text
checkpoints_100_finetune/cifar10_wrn_best.pt
checkpoints_100_finetune/cifar10_wrn_last.pt
checkpoints_100_finetune/history.csv
checkpoints_100_finetune/training_curves.png
```

## 断点继续训练

模型已经达到较高准确率后，继续训练时不适合把学习率直接恢复到初始的 `0.1`。继续微调可以使用较小学习率，例如 `0.006`，并把 `--epochs` 设置成新的总轮数。

续训命令：

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

续训输出：

```text
checkpoints_100_finetune/cifar10_wrn_best.pt
checkpoints_100_finetune/cifar10_wrn_last.pt
checkpoints_100_finetune/history.csv
checkpoints_100_finetune/training_curves.png
```

## 测试和评价指标

测试使用 CIFAR-10 官方测试集，共 10,000 张图片。

测试命令：

```bash
/home/zkf/pytorch-env/bin/python -m cifar10_classifier.evaluate \
  --checkpoint ./checkpoints_100_finetune/cifar10_wrn_best.pt \
  --tta \
  --output-dir ./results/cifar10_100_finetune
```

测试输出：

```text
results/cifar10_100_finetune/confusion_matrix.png
results/cifar10_100_finetune/metrics.json
results/cifar10_100_finetune/classification_report.csv
```

`metrics.json` 保存整体 accuracy、macro precision、macro recall、macro F1、weighted F1。`classification_report.csv` 保存 10 个类别各自的 precision、recall、F1 和 support。

查看摘要：

```bash
/home/zkf/pytorch-env/bin/python -m cifar10_classifier.result_summary \
  --result-dir ./results/cifar10_100_finetune
```

## 本次结果

最终正式结果保存在：

```text
results/cifar10_100_finetune/history.csv
results/cifar10_100_finetune/training_curves.png
results/cifar10_100_finetune/confusion_matrix.png
results/cifar10_100_finetune/metrics.json
results/cifar10_100_finetune/classification_report.csv
results/cifar10_100_finetune/cifar10_samples.png
```

结果：

```text
第 100 轮测试准确率：97.25%
最佳测试准确率：97.30%
最佳轮次：第 96 轮
macro F1：97.30%
weighted F1：97.30%
测试集正确数：9730 / 10000
```

# 数据处理、训练和测试

## 数据下载

项目使用 `torchvision.datasets.CIFAR10` 读取数据。本地没有数据时会下载，已有数据时直接读取。

支持的下载源：

```text
official, sjtu, oneflow, baidu, brainchip
```

推荐 SJTU 镜像：

```bash
/home/zkf/pytorch-env/bin/python -m cifar10_classifier.download --mirror sjtu
```

数据目录：

```text
data/cifar-10-python.tar.gz
data/cifar-10-batches-py/
```

数据集不上传到 GitHub。

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

测试集只做 `ToTensor()` 和 `Normalize()`，不使用随机增强。

训练循环中还会使用：

```text
MixUp alpha = 0.2
CutMix alpha = 1.0
Label smoothing = 0.1
```

所以训练 loss 使用 soft-target cross entropy。

## 训练参数

本次结果对应的主要参数：

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
  --output-dir ./checkpoints_50 \
  --tta \
  --target-acc 90.0
```

## 输出文件

训练过程会保存到本地：

```text
checkpoints_50/cifar10_wrn_best.pt
checkpoints_50/cifar10_wrn_last.pt
checkpoints_50/history.csv
checkpoints_50/training_curves.png
```

其中 `.pt` 权重不上传。提交到 GitHub 的结果快照在：

```text
results/cifar10_100_epochs/history.csv
results/cifar10_100_epochs/training_curves.png
results/cifar10_100_epochs/cifar10_samples.png
```

## 测试方式

测试使用 CIFAR-10 官方测试集，共 10,000 张图片，不使用训练集。

使用 `--tta` 时，测试会平均两次预测：

```text
原图预测
水平翻转图预测
```

测试命令：

```bash
/home/zkf/pytorch-env/bin/python -m cifar10_classifier.evaluate \
  --checkpoint ./checkpoints_50/cifar10_wrn_best.pt \
  --tta \
  --output-dir ./checkpoints_50
```

本次训练结果：

```text
第 100 轮测试准确率：96.87%
最佳测试准确率：96.96%
最佳轮次：第 89 轮
```

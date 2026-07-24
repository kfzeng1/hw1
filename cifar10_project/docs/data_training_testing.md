# 数据处理、训练和测试

## 数据下载

项目使用 `torchvision.datasets.CIFAR10` 读取数据，代码里设置了 `download=True`。如果本地没有数据，会自动下载；如果已经下载并解压，就直接读取本地文件。

支持这些下载源：

```text
official, sjtu, oneflow, baidu, brainchip
```

当前机器实测最快的是 SJTU：

```bash
/home/zkf/pytorch-env/bin/python -m cifar10_classifier.download --mirror sjtu
```

2026-07-20 在这台机器上的测速：

```text
SJTU:                 约 3.3 MB/s，约 50 秒
OneFlow OSS:          约 1.3 MB/s，约 2 分钟
Baidu BOS:            约 1.1 MB/s，约 2.5 分钟
BrainChip:            约 0.33 MB/s，约 9 分钟
torchvision 默认源:   18-31 KB/s，约 1.5-2.6 小时
```

数据已经准备好：

```text
data/cifar-10-python.tar.gz
data/cifar-10-batches-py/
```

## 训练集处理

训练时使用的数据增强：

```text
RandomCrop(32, padding=4)
RandomHorizontalFlip()
RandAugment(num_ops=2, magnitude=14)
ToTensor()
Normalize(mean=(0.4914, 0.4822, 0.4465),
          std=(0.2470, 0.2435, 0.2616))
RandomErasing(p=0.25, scale=(0.02, 0.15), ratio=(0.3, 3.3))
```

作用说明：

- `RandomCrop`：先补边，再随机裁剪，增加平移鲁棒性。
- `RandomHorizontalFlip`：随机水平翻转，适合 CIFAR-10 这类自然图像。
- `RandAugment`：随机组合图像增强操作。
- `Normalize`：按 CIFAR-10 常用均值和方差归一化。
- `RandomErasing`：随机遮挡局部区域，减少模型只记局部纹理的情况。

## 测试集处理

测试集不做随机增强，只做：

```text
ToTensor()
Normalize(mean=(0.4914, 0.4822, 0.4465),
          std=(0.2470, 0.2435, 0.2616))
```

这样测试结果比较稳定。

## Batch 级增强

训练循环里还会对一个 batch 做：

```text
MixUp alpha = 0.2
CutMix alpha = 1.0
Label smoothing = 0.1
```

因此 loss 不是普通的 hard-label cross entropy，而是 soft-target cross entropy。这样可以同时处理 MixUp、CutMix 和 label smoothing 产生的软标签。

## 默认训练参数

正式训练默认参数：

```text
epochs = 50
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

## 正式训练命令

当前监督训练使用的命令：

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

这次重新训练固定跑完 `50` 轮，不设置 `--stop-at-target`。

## 训练输出

正式训练会保存：

```text
checkpoints_50/cifar10_wrn_best.pt
checkpoints_50/cifar10_wrn_last.pt
checkpoints_50/history.csv
checkpoints_50/training_curves.png
```

`history.csv` 记录每个 epoch 的：

```text
epoch, lr, train_loss, train_acc, test_loss, test_acc, best_acc, seconds
```

`training_curves.png` 会画出 loss 和 accuracy 曲线。

## 测试方式

测试使用 CIFAR-10 官方测试集，共 10,000 张图片，不使用训练集。

如果加上 `--tta`，测试时会平均两次预测：

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

输出包括：

```text
test_loss
test_acc
checkpoints_50/confusion_matrix.png
```

## 训练目标

这次主要目标是完整训练 `50` 轮。`target_acc` 仍保留为参考值：

```text
test_acc >= 90.0%
```

这个准确率以 CIFAR-10 测试集为准，使用训练过程中保存的最佳 checkpoint 评估。

# 模型结构和参数

## 模型选择

本项目使用 `WideResNet-28-10`。

WideResNet 是 ResNet 的一种变体。它没有把网络堆得特别深，而是增加每一层的通道数。CIFAR-10 图片只有 `32 x 32`，这种“较宽”的卷积网络通常比很浅的 CNN 更适合冲高准确率。

## 输入和输出

输入是一张 RGB 图片：

```text
3 x 32 x 32
```

输出是 10 个类别的 logits：

```text
airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
```

训练和测试时取 logits 最大的位置作为预测类别。

## 网络结构

Stem：

```text
3x3 Conv: 3 -> 16 channels
```

三个残差阶段：

```text
Stage 1: 4 个 residual block，160 channels，stride 1
Stage 2: 4 个 residual block，320 channels，stride 2
Stage 3: 4 个 residual block，640 channels，stride 2
```

分类头：

```text
BatchNorm
ReLU
Global Average Pooling
Linear: 640 -> 10
```

## Residual Block

每个残差块使用 pre-activation 结构：

```text
BatchNorm -> ReLU -> 3x3 Conv
BatchNorm -> ReLU -> Dropout -> 3x3 Conv
Residual shortcut
```

如果输入输出通道数不同，或者特征图尺寸要缩小，shortcut 使用：

```text
1x1 Conv
```

## 默认模型参数

当前正式训练使用：

```text
depth = 28
widen_factor = 10
drop_rate = 0.3
num_classes = 10
```

训练启动时打印出的参数量：

```text
trainable parameters = 36,479,194
```

也就是约 `3648 万` 个可训练参数。


本项目用的是：

```text
WideResNet-28-10 + 强数据增强 + MixUp/CutMix + cosine 学习率
```

这是为了让模型有足够能力，同时减少过拟合。

## 代码位置

模型代码在：

```text
cifar10_classifier/model.py
```

主要类：

```text
BasicBlock
NetworkBlock
WideResNet
```

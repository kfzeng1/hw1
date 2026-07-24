# 作业 3：CIFAR-10 图像分类

本作业训练 CIFAR-10 图像分类模型。项目包含数据下载、数据处理、模型定义、训练、测试、单张图片预测、网页演示和结果保存。

## 当前结果

100 轮训练结果已整理到：

```text
results/cifar10_100_epochs/
```

结果摘要：

```text
第 100 轮测试准确率：96.87%
最佳测试准确率：96.96%
最佳轮次：第 89 轮
```

结果目录中的文件：

```text
history.csv           每轮训练日志
training_curves.png  训练曲线
cifar10_samples.png  CIFAR-10 原始样本图
README.md            结果说明
```

模型权重 `.pt` 没有上传到 GitHub，保留在本地 `checkpoints_50/` 中。

## 模型和训练策略

模型使用 `WideResNet-28-10`。训练中使用：

```text
RandAugment
RandomErasing
MixUp
CutMix
label smoothing
SGD Nesterov
warmup + cosine 学习率
CUDA AMP
```

详细说明：

```text
docs/model_and_parameters.md
docs/data_training_testing.md
docs/project_overview.md
```

## 目录结构

```text
03_cifar10_classifier/
  cifar10_classifier/
    augment.py
    config.py
    data.py
    demo_server.py
    download.py
    engine.py
    evaluate.py
    inference.py
    main.py
    model.py
    predict.py
    scheduler.py
    utils.py
    visualize.py
  docs/
  results/cifar10_100_epochs/
  tests/
  requirements.txt
```

## 运行环境

```bash
cd assignments/03_cifar10_classifier
```

安装依赖：

```bash
/home/zkf/pytorch-env/bin/python -m pip install -r requirements.txt
```

## 下载数据

推荐使用 SJTU 镜像：

```bash
/home/zkf/pytorch-env/bin/python -m cifar10_classifier.download \
  --data-dir ./data \
  --mirror sjtu
```

## 训练

重新训练 100 轮：

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

断点继续训练：

```bash
/home/zkf/pytorch-env/bin/python -B -m cifar10_classifier.main \
  --resume ./checkpoints_50/cifar10_wrn_last.pt \
  --epochs 100 \
  --output-dir ./checkpoints_50 \
  --tta \
  --target-acc 90.0
```

## 测试

```bash
/home/zkf/pytorch-env/bin/python -m cifar10_classifier.evaluate \
  --checkpoint ./checkpoints_50/cifar10_wrn_best.pt \
  --tta \
  --output-dir ./checkpoints_50
```

## 单张图片预测

```bash
/home/zkf/pytorch-env/bin/python -m cifar10_classifier.predict path/to/image.png \
  --checkpoint ./checkpoints_50/cifar10_wrn_best.pt
```

## 浏览器演示

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

这个测试不需要下载 CIFAR-10，只检查模型前向、loss、反向传播、优化器和测试流程。

```bash
/home/zkf/pytorch-env/bin/python -B -m tests.smoke_test
```

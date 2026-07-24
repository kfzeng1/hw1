# CIFAR-10 项目说明

这个项目用于训练 CIFAR-10 图像分类模型。代码已经按功能拆开，包含数据下载、数据处理、模型定义、训练、测试、单张图片预测、训练曲线和混淆矩阵。

## 目标

重新训练 CIFAR-10 分类模型，固定训练 `50` 轮，并记录训练过程中达到的最佳测试准确率。

## 数据集

CIFAR-10 一共有 60,000 张彩色图片：

- 图片大小：`32 x 32`
- 类别数：10
- 训练集：50,000 张
- 测试集：10,000 张

类别名称：

```text
airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
```

数据已经下载并解压到：

```text
data/cifar-10-python.tar.gz
data/cifar-10-batches-py/
```

压缩包 MD5：

```text
c58f30108f718f92721af3b95e74349a
```

这个值和 CIFAR-10 官方数据一致。

## 目录结构

```text
cifar10_project/
  cifar10_classifier/
    augment.py       MixUp、CutMix、label smoothing、soft cross entropy
    config.py        命令行参数和默认超参数
    data.py          数据下载、数据增强、DataLoader、镜像地址
    demo_server.py   浏览器点击预测演示
    download.py      单独下载数据集
    engine.py        训练、测试、checkpoint 保存和加载
    evaluate.py      加载 checkpoint 做测试，并保存混淆矩阵
    inference.py     推理阶段公共模型加载逻辑
    main.py          正式训练入口
    model.py         WideResNet 模型
    predict.py       单张图片预测
    scheduler.py     warmup + cosine 学习率调度
    utils.py         准确率、随机种子、CSV 日志、参数量统计
    visualize.py     样本图、训练曲线、混淆矩阵
  docs/
    project_overview.md
    model_and_parameters.md
    data_training_testing.md
  tests/
    smoke_test.py
  data/
  requirements.txt
  README.md
```

## 常用命令

进入项目目录：

```bash
cd /home/zkf/pytorch-env/something/hw1/cifar10_project
```

下载数据：

```bash
/home/zkf/pytorch-env/bin/python -m cifar10_classifier.download --mirror sjtu
```

正式训练：

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

测试最佳模型：

```bash
/home/zkf/pytorch-env/bin/python -m cifar10_classifier.evaluate \
  --checkpoint ./checkpoints_50/cifar10_wrn_best.pt \
  --tta \
  --output-dir ./checkpoints_50
```

启动浏览器演示：

```bash
/home/zkf/pytorch-env/bin/python -m cifar10_classifier.demo_server \
  --checkpoint ./checkpoints_50/cifar10_wrn_best.pt \
  --data-dir ./data \
  --port 8008
```

快速检查代码能不能跑：

```bash
/home/zkf/pytorch-env/bin/python -B -m tests.smoke_test
```

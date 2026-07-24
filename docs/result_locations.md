# 四个作业结果位置

这里列的是提交后最需要看的文件。模型权重、数据集、API key 和实习报告没有上传。

## 1. 多项式逼近 sin(x)

代码：

```text
assignments/01_poly_sin/poly_sin_regression.py
```

结果图：

```text
assignments/01_poly_sin/poly_sin_regression.png
```

运行：

```bash
cd assignments/01_poly_sin
/home/zkf/pytorch-env/bin/python poly_sin_regression.py
```

## 2. 牛顿法最小化 x^2 + y^2

代码：

```text
assignments/02_newton_min/newton_square_xy.py
```

结果图：

```text
assignments/02_newton_min/newton_square_xy.png
```

运行：

```bash
cd assignments/02_newton_min
/home/zkf/pytorch-env/bin/python newton_square_xy.py
```

## 3. CIFAR-10 图像分类

项目目录：

```text
assignments/03_cifar10_classifier/
```

100 轮训练结果：

```text
assignments/03_cifar10_classifier/results/cifar10_100_epochs/
```

结果文件：

```text
assignments/03_cifar10_classifier/results/cifar10_100_epochs/history.csv
assignments/03_cifar10_classifier/results/cifar10_100_epochs/training_curves.png
assignments/03_cifar10_classifier/results/cifar10_100_epochs/cifar10_samples.png
assignments/03_cifar10_classifier/results/cifar10_100_epochs/README.md
```

结果摘要：

```text
第 100 轮测试准确率：96.87%
最佳测试准确率：96.96%
最佳轮次：第 89 轮
```

权重文件 `.pt` 没有上传。需要复现时，进入 `assignments/03_cifar10_classifier/`，按该目录的 `README.md` 下载数据并训练。

## 4. 豆包 API 生成 3D 视频

代码：

```text
assignments/04_doubao_3d_video/generate_3d_model.py
assignments/04_doubao_3d_video/generate_video.py
assignments/04_doubao_3d_video/render_turntable.py
assignments/04_doubao_3d_video/view_model.html
```

输入图片：

```text
assignments/04_doubao_3d_video/inputs/met_bulul_DP320246.jpg
```

视频结果：

```text
assignments/04_doubao_3d_video/results/turntable.mp4
```

`assignments/04_doubao_3d_video/.env` 没有上传。需要运行时，复制 `.env.example` 后填入自己的 `ARK_API_KEY`。

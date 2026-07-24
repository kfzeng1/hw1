# 四个作业结果位置

这份文件只说明结果在哪里看。模型权重、数据集和密钥文件没有上传。

## 作业 1：多项式逼近 sin(x)

代码：

```text
hw_poly_sin/poly_sin_regression.py
```

结果图：

```text
hw_poly_sin/poly_sin_regression.png
```

运行命令：

```bash
cd hw_poly_sin
/home/zkf/pytorch-env/bin/python poly_sin_regression.py
```

## 作业 2：牛顿法最小化 x^2 + y^2

代码：

```text
hw_newton_min/newton_square_xy.py
```

结果图：

```text
hw_newton_min/newton_square_xy.png
```

运行命令：

```bash
cd hw_newton_min
/home/zkf/pytorch-env/bin/python newton_square_xy.py
```

## 作业 3：CIFAR-10 图像分类

项目目录：

```text
cifar10_project/
```

100 轮训练结果：

```text
cifar10_project/results/cifar10_100_epochs/
```

里面的文件：

```text
cifar10_project/results/cifar10_100_epochs/history.csv
cifar10_project/results/cifar10_100_epochs/training_curves.png
cifar10_project/results/cifar10_100_epochs/cifar10_samples.png
cifar10_project/results/cifar10_100_epochs/README.md
```

结果：

```text
第 100 轮测试准确率：96.87%
最佳测试准确率：96.96%
最佳轮次：第 89 轮
```

模型权重没有上传。需要复现时，先下载 CIFAR-10 数据，再按 `cifar10_project/README.md` 里的命令训练。

## 作业 4：豆包 API 生成 3D 视频

代码：

```text
api/3d.py
api/vedio.py
api/render_turntable.py
api/view_model.html
```

参考图片：

```text
api/met_bulul_DP320246.jpg
```

视频结果：

```text
api/turntable.mp4
```

API key 文件 `api/.env` 没有上传。需要运行时，复制 `api/.env.example` 并填入自己的 key。

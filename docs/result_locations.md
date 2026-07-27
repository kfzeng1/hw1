# 四个作业结果位置

这份文件只说明结果在哪里看。数据集、模型权重、API key 和实习报告不上传。

## 作业 1：多项式逼近 sin(x)

代码：

```text
assignments/01_poly_sin/poly_sin_regression.py
```

结果：

```text
assignments/01_poly_sin/results/fit_comparison.png
assignments/01_poly_sin/results/loss_convergence.png
assignments/01_poly_sin/results/coefficients.txt
```

运行：

```bash
cd assignments/01_poly_sin
/home/zkf/pytorch-env/bin/python poly_sin_regression.py
```

## 作业 2：牛顿法最小化 x^2 + y^2

代码：

```text
assignments/02_newton_min/newton_square_xy.py
```

结果：

```text
assignments/02_newton_min/results/newton_path.png
assignments/02_newton_min/results/iteration_process.png
assignments/02_newton_min/results/iteration_log.txt
```

运行：

```bash
cd assignments/02_newton_min
/home/zkf/pytorch-env/bin/python newton_square_xy.py
```

## 作业 3：CIFAR-10 图像分类

项目目录：

```text
assignments/03_cifar10_classifier/
```

已有 100 轮结果：

```text
assignments/03_cifar10_classifier/results/cifar10_100_epochs/history.csv
assignments/03_cifar10_classifier/results/cifar10_100_epochs/training_curves.png
assignments/03_cifar10_classifier/results/cifar10_100_epochs/cifar10_samples.png
```

结果摘要：

```text
第 100 轮测试准确率：96.87%
最佳测试准确率：96.96%
最佳轮次：第 89 轮
```

重新训练 50 轮后，新的 checkpoint、训练日志和训练曲线会保存到训练命令指定的 `--output-dir`。

## 作业 4：豆包 API 生成 3D 视频

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

结果视频：

```text
assignments/04_doubao_3d_video/results/turntable.mp4
```

`.env` 不上传。需要运行 API 时，复制 `.env.example` 后填自己的 `ARK_API_KEY`。

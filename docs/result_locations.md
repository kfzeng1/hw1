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
assignments/01_poly_sin/results/training_history.csv
assignments/01_poly_sin/results/coefficients.txt
```

运行：

```bash
cd assignments/01_poly_sin
/home/zkf/pytorch-env/bin/python poly_sin_regression.py
```

## 作业 2：牛顿法最小化 x1^2+2x2^2+x1x2

代码：

```text
assignments/02_newton_min/newton_quadratic_min.py
```

结果：

```text
assignments/02_newton_min/results/newton_path.png
assignments/02_newton_min/results/iteration_process.png
assignments/02_newton_min/results/iteration_log.txt
assignments/02_newton_min/results/iteration_history.csv
```

运行：

```bash
cd assignments/02_newton_min
/home/zkf/pytorch-env/bin/python newton_quadratic_min.py
```

## 作业 3：CIFAR-10 图像分类

项目目录：

```text
assignments/03_cifar10_classifier/
```

本次 50 轮训练正式结果：

```text
assignments/03_cifar10_classifier/results/cifar10_50_epochs/history.csv
assignments/03_cifar10_classifier/results/cifar10_50_epochs/training_curves.png
assignments/03_cifar10_classifier/results/cifar10_50_epochs/confusion_matrix.png
assignments/03_cifar10_classifier/results/cifar10_50_epochs/metrics.json
assignments/03_cifar10_classifier/results/cifar10_50_epochs/classification_report.csv
assignments/03_cifar10_classifier/results/cifar10_50_epochs/cifar10_samples.png
```

结果摘要：

```text
第 50 轮测试准确率：96.04%
最佳测试准确率：96.04%
最佳轮次：第 50 轮
macro F1：96.03%
weighted F1：96.03%
```

checkpoint 保存在本地 `assignments/03_cifar10_classifier/checkpoints_50/`，不上传到 GitHub。

## 作业 4：豆包 API 生成 3D 视频

代码：

```text
assignments/04_doubao_3d_video/generate_3d_model.py
assignments/04_doubao_3d_video/generate_video.py
assignments/04_doubao_3d_video/doubao_utils.py
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
assignments/04_doubao_3d_video/results/input_preview.png
assignments/04_doubao_3d_video/results/turntable_preview.png
```

`.env` 不上传。需要运行 API 时，复制 `.env.example` 后填自己的 `ARK_API_KEY`。

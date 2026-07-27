# 机器学习与深度学习作业

仓库按 4 个小作业整理。每个作业都有独立目录，目录里放代码、README 和结果文件。

## 目录

```text
assignments/
  01_poly_sin/             多项式逼近 sin(x)
  02_newton_min/           牛顿法最小化 x^2 + y^2
  03_cifar10_classifier/   CIFAR-10 图像分类
  04_doubao_3d_video/      豆包 API 生成 3D 视频
docs/
  project_structure.md
  result_locations.md
```

## 结果位置

| 作业 | 主要代码 | 结果 |
| --- | --- | --- |
| 1. 多项式逼近 `sin(x)` | `assignments/01_poly_sin/poly_sin_regression.py` | `assignments/01_poly_sin/results/` |
| 2. 牛顿法最小化 `x^2 + y^2` | `assignments/02_newton_min/newton_square_xy.py` | `assignments/02_newton_min/results/` |
| 3. CIFAR-10 图像分类 | `assignments/03_cifar10_classifier/` | `assignments/03_cifar10_classifier/results/cifar10_100_epochs/` |
| 4. 豆包 API 生成 3D 视频 | `assignments/04_doubao_3d_video/` | `assignments/04_doubao_3d_video/results/turntable.mp4` |

更详细的结果说明在：

```text
docs/result_locations.md
```

## 提交说明

仓库上传源码、文档和小体积结果文件。下面这些内容不上传：

```text
docx/
assignments/03_cifar10_classifier/data/
assignments/03_cifar10_classifier/checkpoints_*/
assignments/04_doubao_3d_video/.env
assignments/04_doubao_3d_video/output_model/
```

CIFAR-10 旧模型权重已经删除。后续重新训练 50 轮时，会重新生成新的 checkpoint。

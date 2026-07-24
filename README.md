# 机器学习与深度学习作业

本仓库整理了本次课程的 4 个作业，代码和结果文件已经分目录放好：

1. `hw_poly_sin/`：使用线性回归和多项式特征逼近 `sin(x)`，次数为 3、4、5。
2. `hw_newton_min/`：使用牛顿法求解 `f(x, y) = x^2 + y^2` 的最小值，并绘制迭代轨迹。
3. `cifar10_project/`：CIFAR-10 图像分类项目，包含数据下载、数据处理、模型训练、测试、预测和网页演示。
4. `api/`：调用豆包 API 生成 3D 模型和视频，并用脚本渲染旋转展示视频。

## 四个作业结果位置

| 作业 | 代码位置 | 结果位置 |
| --- | --- | --- |
| 多项式逼近 `sin(x)` | `hw_poly_sin/poly_sin_regression.py` | `hw_poly_sin/poly_sin_regression.png` |
| 牛顿法最小化 `x^2 + y^2` | `hw_newton_min/newton_square_xy.py` | `hw_newton_min/newton_square_xy.png` |
| CIFAR-10 图像分类 | `cifar10_project/` | `cifar10_project/results/cifar10_100_epochs/` |
| 豆包 API 生成 3D 视频 | `api/3d.py`、`api/vedio.py`、`api/render_turntable.py` | `api/turntable.mp4` |

CIFAR-10 的 100 轮结果中，`history.csv` 是训练日志，`training_curves.png` 是训练曲线，`cifar10_samples.png` 是样本图。最佳测试准确率为 96.96%。更详细的位置说明见 `RESULTS.md`。

## 注意

- `api/.env` 不上传，需要根据 `api/.env.example` 自行配置。
- CIFAR-10 原始数据、训练 checkpoint 和大模型生成的 `.glb` 文件体积较大，已通过 `.gitignore` 排除。
- CIFAR-10 数据可通过项目内脚本重新下载。
- `docx/` 不上传到 GitHub，本地报告文件保留在本机目录中。

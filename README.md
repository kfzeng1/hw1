# 机器学习与深度学习作业

本仓库整理了本次课程作业代码和实习报告，主要包括：

1. `hw_poly_sin/`：使用线性回归和多项式特征逼近 `sin(x)`，次数为 3、4、5。
2. `hw_newton_min/`：使用牛顿法求解 `f(x, y) = x^2 + y^2` 的最小值，并绘制迭代轨迹。
3. `cifar10_project/`：CIFAR-10 图像分类项目，包含数据下载、数据处理、模型训练、测试、预测和网页演示。

另外，`api/` 中保留了调用豆包 API 生成 3D 模型和视频的相关代码；`docx/` 中保留最终实习报告。

## 结果文件

- `hw_poly_sin/poly_sin_regression.png`：多项式逼近 `sin(x)` 结果图。
- `hw_newton_min/newton_square_xy.png`：牛顿法最小化 `x^2 + y^2` 的迭代轨迹图。
- `cifar10_project/results/cifar10_100_epochs/`：CIFAR-10 训练到 100 轮后的日志、训练曲线和样本图，最佳测试准确率为 96.96%。
- `api/turntable.mp4`：调用 API 和渲染脚本生成的 3D 展示视频结果。

## 注意

- `api/.env` 不上传，需要根据 `api/.env.example` 自行配置。
- CIFAR-10 原始数据、训练 checkpoint 和大模型生成的 `.glb` 文件体积较大，已通过 `.gitignore` 排除。
- CIFAR-10 数据可通过项目内脚本重新下载。

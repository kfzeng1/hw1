# 机器学习与深度学习作业

这个仓库按 4 个作业重新整理，代码、结果和说明分开放。根目录只保留总说明，具体内容在 `assignments/` 和 `docs/`。

## 目录结构

```text
.
├── assignments/
│   ├── 01_poly_sin/             多项式逼近 sin(x)
│   ├── 02_newton_min/           牛顿法最小化 x^2 + y^2
│   ├── 03_cifar10_classifier/   CIFAR-10 图像分类
│   └── 04_doubao_3d_video/      豆包 API 生成 3D 视频
├── docs/
│   ├── project_structure.md     仓库结构说明
│   └── result_locations.md      四个作业结果位置
└── README.md
```

## 四个作业

| 作业 | 代码位置 | 结果位置 |
| --- | --- | --- |
| 多项式逼近 `sin(x)` | `assignments/01_poly_sin/poly_sin_regression.py` | `assignments/01_poly_sin/poly_sin_regression.png` |
| 牛顿法最小化 `x^2 + y^2` | `assignments/02_newton_min/newton_square_xy.py` | `assignments/02_newton_min/newton_square_xy.png` |
| CIFAR-10 图像分类 | `assignments/03_cifar10_classifier/` | `assignments/03_cifar10_classifier/results/cifar10_100_epochs/` |
| 豆包 API 生成 3D 视频 | `assignments/04_doubao_3d_video/` | `assignments/04_doubao_3d_video/results/turntable.mp4` |

CIFAR-10 已训练到 100 轮。第 100 轮测试准确率为 `96.87%`，最佳测试准确率为 `96.96%`，最佳轮次是第 89 轮。

## 快速查看结果

详细结果位置见：

```text
docs/result_locations.md
```

仓库结构说明见：

```text
docs/project_structure.md
```

## 不上传的内容

以下文件保留在本机，但不放进 GitHub：

- `docx/`：实习报告 Word/PDF 文件
- `assignments/03_cifar10_classifier/data/`：CIFAR-10 原始数据
- `assignments/03_cifar10_classifier/checkpoints_*/`：训练权重
- `assignments/04_doubao_3d_video/.env`：API key
- `assignments/04_doubao_3d_video/output_model/`：生成的 `.glb` 模型文件

如果要复现训练或重新生成视频，按各作业目录里的 `README.md` 操作。

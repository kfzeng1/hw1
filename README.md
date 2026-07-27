# 机器学习与深度学习作业

本仓库按 4 个作业拆分。每个作业都有自己的代码、README 和结果目录；根目录只放总览文档，不混放训练数据和模型权重。

## 先看这里

四个作业的完成情况、运行入口、结果位置和本地 Word 文档位置见：

```text
docs/assignment_summary.md
docs/result_locations.md
```

## 目录

```text
assignments/
  01_poly_sin/                 作业 1：多项式逼近 sin(x)
    poly_sin_regression.py     主程序
    results/                   拟合图、损失曲线、训练记录和系数

  02_newton_min/               作业 2：牛顿法最小化二元函数
    newton_quadratic_min.py    主程序
    results/                   迭代路径图、过程图、日志和 CSV

  03_cifar10_classifier/       作业 3：CIFAR-10 图像分类项目
    cifar10_classifier/        训练、评估、预测、网页演示代码包
    docs/                      模型、参数、数据处理和测试说明
    results/cifar10_100_finetune/
                                100 轮正式测试结果
    tests/                     快速自测

  04_doubao_3d_video/          作业 4：豆包 API 生成 3D 视频
    generate_*.py              API 生成脚本
    render_turntable.py        本地转台视频渲染
    view_model.html            本地查看页面
    inputs/                    输入图片
    results/                   预览图和视频结果

docs/
  assignment_summary.md        四个作业完成情况
  project_structure.md         更详细的项目结构说明
  result_locations.md          所有结果文件位置
```

## 结果位置

| 作业 | 主要代码 | 正式结果 |
| --- | --- | --- |
| 1. 多项式逼近 `sin(x)` | `assignments/01_poly_sin/poly_sin_regression.py` | `assignments/01_poly_sin/results/` |
| 2. 牛顿法最小化 `x1^2+2x2^2+x1x2` | `assignments/02_newton_min/newton_quadratic_min.py` | `assignments/02_newton_min/results/` |
| 3. CIFAR-10 图像分类 | `assignments/03_cifar10_classifier/` | `assignments/03_cifar10_classifier/results/cifar10_100_finetune/` |
| 4. 豆包 API 生成 3D 视频 | `assignments/04_doubao_3d_video/` | `assignments/04_doubao_3d_video/results/turntable.mp4` |

更详细的结果说明在：

```text
docs/result_locations.md
```

## 提交说明

GitHub 上传源码、Markdown 文档和小体积结果文件。下面这些内容只保留在本地，不上传：

```text
*.docx
docx/
专业实习课程考核要求*.pdf
assignments/03_cifar10_classifier/data/
assignments/03_cifar10_classifier/checkpoints_*/
assignments/04_doubao_3d_video/.env
assignments/04_doubao_3d_video/output_model/
```

CIFAR-10 checkpoint 保存在本地 `assignments/03_cifar10_classifier/checkpoints_100_finetune/`，体积较大，不上传；可用代码重新训练生成。

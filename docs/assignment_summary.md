# 四个作业综合整理

本仓库按课程小作业要求拆成 4 个独立目录。每个目录都包含代码、README 和结果文件；Word 文档保存在本地对应目录，但不上传到 GitHub。

## 完成情况

| 作业 | 课程要求 | 当前实现 | 主要结果 |
| --- | --- | --- | --- |
| 作业 1 | 用 3、4、5 次多项式拟合 `sin(x)`，用梯度下降最小化 MSE | 手写全批量梯度下降，输出拟合图、损失曲线、训练 CSV 和多项式系数 | 5 次多项式最终 MSE 为 `0.0000204249` |
| 作业 2 | 用牛顿法最小化 `f(x1,x2)=x1^2+2x2^2+x1x2` | 计算梯度、Hessian 和牛顿更新，保存完整迭代过程 | 极小点 `(0,0)`，极小值 `0` |
| 作业 3 | 用 PyTorch CNN 训练 CIFAR-10，测试准确率达到 90% 以上，并输出多种分类指标 | WideResNet-28-10，先训练 50 轮，再继续微调到 100 轮，保存训练曲线、混淆矩阵、JSON/CSV 指标 | 最佳测试准确率 `97.30%`，macro F1 `97.30%` |
| 作业 4 | 对接豆包 API，实现物品 3D 展示交互效果 | 提供 3D 模型生成、图生视频、本地 turntable 渲染和网页查看代码 | 保存输入预览、3D 展示截图和 `turntable.mp4` |

## 目录对应关系

```text
assignments/
  01_poly_sin/             作业 1：多项式逼近 sin(x)
  02_newton_min/           作业 2：牛顿法最小化二元函数
  03_cifar10_classifier/   作业 3：CIFAR-10 图像分类
  04_doubao_3d_video/      作业 4：豆包 API 生成 3D 视频
```

## 本地 Word 文档

这些文件在本地已经生成，和代码放在同一个作业目录，符合课程要求。因为之前要求 `docx` 不上传，Git 中已忽略这些文件。

```text
assignments/01_poly_sin/作业1_多项式逼近sinx.docx
assignments/02_newton_min/作业2_牛顿法最小化.docx
assignments/03_cifar10_classifier/作业3_CIFAR10图像分类.docx
assignments/04_doubao_3d_video/作业4_豆包3D视频.docx
```

## GitHub 中保留的结果

```text
assignments/01_poly_sin/results/
assignments/02_newton_min/results/
assignments/03_cifar10_classifier/results/cifar10_50_epochs/
assignments/03_cifar10_classifier/results/cifar10_100_finetune/
assignments/04_doubao_3d_video/results/
```

## 不上传内容

```text
*.docx
专业实习课程考核要求*.pdf
assignments/03_cifar10_classifier/data/
assignments/03_cifar10_classifier/checkpoints_*/
assignments/04_doubao_3d_video/.env
assignments/04_doubao_3d_video/output_model/
```

这些文件要么包含本地环境或密钥，要么体积较大。需要复现实验时，按各作业 README 中的命令重新运行即可。

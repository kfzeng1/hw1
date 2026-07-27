# 作业 1：多项式逼近 sin(x)

要求：用 3、4、5 次多项式拟合 `sin(x)`，通过梯度下降更新权重，最小化均方误差。提交内容需要包含完整代码、损失收敛过程和拟合效果截图。

## 文件

```text
poly_sin_regression.py          训练、绘图和结果保存代码
results/fit_comparison.png      拟合效果对比图
results/loss_convergence.png    损失收敛曲线
results/coefficients.txt        学到的多项式系数
```

## 运行

```bash
cd assignments/01_poly_sin
/home/zkf/pytorch-env/bin/python poly_sin_regression.py
```

运行后会重新生成 `results/` 里的结果文件。

## 实现说明

代码把输入 `x` 转成 `[1, x, x^2, ..., x^n]` 的多项式特征，再用 `torch.nn.Linear` 做线性回归。训练目标是 MSE loss，优化器使用 Adam。默认训练 5000 轮，次数为 `3,4,5`。

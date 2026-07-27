# 作业 1：多项式逼近 sin(x)

要求：用 3、4、5 次多项式拟合 `sin(x)`，通过梯度下降更新权重，最小化均方误差。提交内容需要包含完整代码、损失收敛过程和拟合效果截图。

## 文件

```text
poly_sin_regression.py          训练、绘图和结果保存代码
results/fit_comparison.png      拟合效果对比图
results/loss_convergence.png    损失收敛曲线
results/training_history.csv    每个次数、每一轮的 MSE 记录
results/coefficients.txt        学到的多项式系数
```

## 运行

```bash
cd assignments/01_poly_sin
/home/zkf/pytorch-env/bin/python poly_sin_regression.py
```

运行后会重新生成 `results/` 里的结果文件。

## 实现说明

代码把输入 `x` 转成 `[1, x, x^2, ..., x^n]` 的多项式特征，再用线性模型表示多项式系数。训练目标是 MSE loss，参数更新使用手写的全批量梯度下降：

```text
w = w - lr * gradient
```

为避免高阶特征数值过大，程序会对每一阶特征按最大绝对值缩放，保存系数时再换算回原始 `x` 的多项式表达。默认训练 10000 轮，次数为 `3,4,5`。

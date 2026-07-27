# 作业 2：牛顿法最小化二元函数

要求：使用牛顿法优化二元函数

```text
f(x1, x2) = x1^2 + 2x2^2 + x1x2
```

求该函数的极小值。提交内容需要包含完整代码、完整迭代过程和最终极值结果截图。

## 文件

```text
newton_quadratic_min.py         牛顿法、迭代记录和绘图代码
results/newton_path.png         迭代轨迹图
results/iteration_process.png   完整迭代过程截图
results/iteration_log.txt       迭代过程文本记录
results/iteration_history.csv   迭代过程表格数据
```

## 运行

```bash
cd assignments/02_newton_min
/home/zkf/pytorch-env/bin/python newton_quadratic_min.py
```

运行后会重新生成 `results/` 里的结果文件。

## 实现说明

对 `f(x1,x2)=x1^2+2x2^2+x1x2` 有：

```text
gradient = [2x1 + x2, x1 + 4x2]
Hessian = [[2, 1], [1, 4]]
```

牛顿更新为：

```text
[x1, x2] = [x1, x2] - H^(-1) * gradient
```

默认初始点是 `(3, 4)`。程序会检查 Hessian 矩阵是否正定，并保存迭代表、等高线轨迹图、文本日志和 CSV 结果。该函数为凸二次函数，牛顿法一步即可到达极小点 `(0, 0)`，极小值为 `0`。

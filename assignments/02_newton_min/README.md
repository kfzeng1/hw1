# 作业 2：牛顿法最小化 x^2 + y^2

要求：使用牛顿法优化二元函数

```text
f(x, y) = x^2 + y^2
```

求该函数的极小值。提交内容需要包含完整代码、完整迭代过程和最终极值结果截图。

## 文件

```text
newton_square_xy.py             牛顿法、迭代记录和绘图代码
results/newton_path.png         迭代轨迹图
results/iteration_process.png   完整迭代过程截图
results/iteration_log.txt       迭代过程文本记录
```

## 运行

```bash
cd assignments/02_newton_min
/home/zkf/pytorch-env/bin/python newton_square_xy.py
```

运行后会重新生成 `results/` 里的结果文件。

## 实现说明

对 `f(x,y)=x^2+y^2` 有：

```text
gradient = [2x, 2y]
Hessian = [[2, 0], [0, 2]]
```

牛顿更新为：

```text
[x, y] = [x, y] - H^(-1) * gradient
```

默认初始点是 `(3, 4)`。这个函数是凸二次函数，因此一步就能到达极小点 `(0, 0)`。

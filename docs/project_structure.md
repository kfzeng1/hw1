# 项目结构说明

仓库只保留四类内容：作业代码、Markdown 说明、小体积结果文件、示例输入。数据集、模型权重、API key、Word 报告和课程 PDF 不进入 Git。

```text
assignments/
  01_poly_sin/
    README.md                     作业说明和运行命令
    poly_sin_regression.py        多项式拟合主程序
    results/
      coefficients.txt            3/4/5 次多项式系数
      fit_comparison.png          拟合效果图
      loss_convergence.png        损失下降图
      training_history.csv        训练记录

  02_newton_min/
    README.md                     作业说明和运行命令
    newton_quadratic_min.py       牛顿法主程序
    results/
      iteration_history.csv       迭代数据
      iteration_log.txt           迭代日志
      iteration_process.png       迭代过程图
      newton_path.png             等高线和迭代路径

  03_cifar10_classifier/
    README.md                     CIFAR-10 项目入口说明
    requirements.txt              Python 依赖
    cifar10_classifier/           Python 包
      augment.py                  MixUp、CutMix、label smoothing
      config.py                   训练参数
      data.py                     数据下载、增强和 DataLoader
      demo_server.py              浏览器演示
      download.py                 数据下载入口
      engine.py                   训练、评估、checkpoint
      evaluate.py                 测试集评估入口
      inference.py                加载模型和设备选择
      main.py                     训练入口
      metrics.py                  分类指标
      model.py                    WideResNet-28-10
      predict.py                  单张图片预测
      result_summary.py           结果摘要
      scheduler.py                warmup + cosine 学习率
      utils.py                    通用工具
      visualize.py                图像和曲线绘制
    docs/
      data_training_testing.md    数据处理、训练、测试说明
      model_and_parameters.md     模型结构和参数说明
      project_overview.md         项目说明
    results/
      cifar10_100_finetune/       100 轮正式结果
    tests/
      smoke_test.py               快速自测

  04_doubao_3d_video/
    README.md                     作业说明和运行命令
    .env.example                  环境变量模板
    doubao_utils.py               豆包 API 公共工具
    generate_3d_model.py          生成 3D 模型
    generate_video.py             图生视频
    render_turntable.py           本地转台视频渲染
    view_model.html               本地模型查看页面
    inputs/
      met_bulul_DP320246.jpg      示例输入图片
    results/
      input_preview.png           输入预览
      turntable_preview.png       视频预览
      turntable.mp4               最终视频

docs/
  assignment_summary.md           四个作业总览
  project_structure.md            当前文件
  result_locations.md             结果文件位置
```

## 目录规则

```text
根目录 README.md       只写总览和入口
assignments/*/README.md 只写单个作业怎么运行、结果怎么看
assignments/*/results/  只放可以上传的小体积结果
docs/                  放跨作业说明
```

作业 3 是完整项目，代码在 `cifar10_classifier/` 包内，文档在 `docs/`，正式结果在 `results/cifar10_100_finetune/`。

## 本地保留但不提交

```text
*.docx
docx/
专业实习课程考核要求*.pdf
assignments/03_cifar10_classifier/data/
assignments/03_cifar10_classifier/checkpoints_*/
assignments/04_doubao_3d_video/.env
assignments/04_doubao_3d_video/output_model/
```

这些文件要么包含本地信息，要么体积较大，不适合放进 GitHub。

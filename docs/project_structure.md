# 项目结构说明

仓库按小作业拆分，避免代码、图片和文档混在一起。

```text
assignments/
  01_poly_sin/
    README.md
    poly_sin_regression.py
    results/

  02_newton_min/
    README.md
    newton_quadratic_min.py
    results/

  03_cifar10_classifier/
    README.md
    cifar10_classifier/
    docs/
    results/
    tests/
    requirements.txt

  04_doubao_3d_video/
    README.md
    .env.example
    doubao_utils.py
    generate_3d_model.py
    generate_video.py
    render_turntable.py
    view_model.html
    inputs/
    results/

docs/
  assignment_summary.md
  project_structure.md
  result_locations.md
```

## 每个作业目录内的内容

```text
README.md        该作业说明和运行命令
*.py             代码入口或功能模块
results/         已生成的小体积结果文件
作业*.docx       本地 Word 报告，不上传 GitHub
```

作业 3 是完整项目，代码在 `cifar10_classifier/` 包内，文档在 `docs/`。50 轮结果在 `results/cifar10_50_epochs/`，继续训练到 100 轮后的最新结果在 `results/cifar10_100_finetune/`。

## 不提交的内容

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

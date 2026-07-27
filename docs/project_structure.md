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
    newton_square_xy.py
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
    generate_3d_model.py
    generate_video.py
    render_turntable.py
    view_model.html
    inputs/
    results/

docs/
  project_structure.md
  result_locations.md
```

## 不提交的内容

```text
docx/
专业实习课程考核要求*.pdf
assignments/03_cifar10_classifier/data/
assignments/03_cifar10_classifier/checkpoints_*/
assignments/04_doubao_3d_video/.env
assignments/04_doubao_3d_video/output_model/
```

这些文件要么包含本地信息，要么体积较大，不适合放进 GitHub。

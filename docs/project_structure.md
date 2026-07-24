# 项目结构说明

仓库按作业拆成 4 个目录。每个目录里保留代码、结果和本作业说明，根目录只做索引。

```text
assignments/
  01_poly_sin/
    README.md
    poly_sin_regression.py
    poly_sin_regression.png

  02_newton_min/
    README.md
    newton_square_xy.py
    newton_square_xy.png

  03_cifar10_classifier/
    README.md
    cifar10_classifier/
    docs/
    results/cifar10_100_epochs/
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
```

## 提交策略

上传到 GitHub 的内容包括源码、说明文档和较小的结果文件。下面这些内容不上传：

```text
docx/
assignments/03_cifar10_classifier/data/
assignments/03_cifar10_classifier/checkpoints_*/
assignments/04_doubao_3d_video/.env
assignments/04_doubao_3d_video/output_model/
```

这样仓库打开后能直接看到代码和结果，又不会把密钥、数据集和几百 MB 的模型权重放进远端。

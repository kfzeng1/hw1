# 作业 4：豆包 API 生成 3D 视频

本作业根据一张参考图片调用豆包相关 API，生成 3D 模型或视频，并用本地脚本渲染旋转展示视频。

## 文件

```text
generate_3d_model.py       根据参考图生成 3D 模型
generate_video.py          根据参考图调用图生视频接口
render_turntable.py        将 .glb 模型渲染成旋转 mp4
view_model.html            在浏览器中查看 .glb 模型
.env.example               API key 配置示例
inputs/met_bulul_DP320246.jpg
results/turntable.mp4
```

## 结果

视频结果在：

```text
results/turntable.mp4
```

参考图片在：

```text
inputs/met_bulul_DP320246.jpg
```

## 运行

先配置 API key：

```bash
cd assignments/04_doubao_3d_video
cp .env.example .env
```

然后在 `.env` 中填入 `ARK_API_KEY`。

生成 3D 模型：

```bash
/home/zkf/pytorch-env/bin/python generate_3d_model.py
```

调用图生视频：

```bash
/home/zkf/pytorch-env/bin/python generate_video.py
```

如果已经有 `.glb` 模型，可以渲染旋转视频：

```bash
/home/zkf/pytorch-env/bin/python render_turntable.py --out results/turntable.mp4
```

`.env` 和 `output_model/` 不上传到 GitHub。

# 作业 4：豆包 API 生成 3D 视频

要求：对接豆包应用程序开发接口，完成商品或物品的三维展示交互效果。这里使用一张木雕文物图片作为输入，生成 3D 模型或视频，并保存旋转展示结果。

## 文件

```text
generate_3d_model.py              调用接口生成 3D 模型
generate_video.py                 调用图生视频接口
doubao_utils.py                   API key、图片编码、任务轮询和下载工具
render_turntable.py               将 .glb 模型渲染成旋转 mp4
view_model.html                   浏览器查看 .glb 模型
.env.example                      API key 配置示例
inputs/met_bulul_DP320246.jpg     输入图片
results/input_preview.png         输入图片预览
results/turntable_preview.png     3D 展示截图
results/turntable.mp4             结果视频
```

## 运行前配置

```bash
cd assignments/04_doubao_3d_video
cp .env.example .env
```

然后在 `.env` 中填入：

```text
ARK_API_KEY=你的 key
```

`.env` 不上传到 GitHub。

## 生成 3D 模型

```bash
/home/zkf/pytorch-env/bin/python generate_3d_model.py \
  --image inputs/met_bulul_DP320246.jpg \
  --output-dir output_model \
  --task-json results/seed3d_task_result.json
```

生成的 `.glb` 文件会保存在 `output_model/`。这个目录体积较大，不上传。

## 生成视频

```bash
/home/zkf/pytorch-env/bin/python generate_video.py \
  --image inputs/met_bulul_DP320246.jpg \
  --output results/doubao_generated_video.mp4 \
  --task-json results/video_task_result.json
```

接口返回内容会保存为 JSON，视频文件会下载到 `results/`。

## 渲染旋转展示视频

如果本地已有 `.glb` 模型，可以生成旋转展示视频：

```bash
/home/zkf/pytorch-env/bin/python render_turntable.py \
  --out results/turntable.mp4
```

提交到仓库中的结果视频：

```text
results/turntable.mp4
```

## 说明

`.env` 保存真实 API key，不上传到 GitHub。`output_model/` 保存接口生成的 glb 模型，体积较大，也不上传；需要时可以按上面的命令重新生成。

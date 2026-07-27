import argparse
import base64
import os
import shutil
import time
import zipfile
from pathlib import Path

import requests
from dotenv import load_dotenv
from volcenginesdkarkruntime import Ark


DEFAULT_MODEL_ID = "doubao-seed3d-2-0-260328"
DEFAULT_PROMPT = (
    "Create a complete 3D model of the museum artifact in the reference image: "
    "a standing dark wooden human figure on a rectangular base. Preserve the full body, "
    "facial features, hands, legs, base blocks, aged wood texture, and museum-object proportions. "
    "No text, no watermark, no extra objects, no display stand, clean geometry. "
    "--subdivisionlevel medium --fileformat glb"
)


def image_to_data_uri(image_path):
    image_bytes = image_path.read_bytes()
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:image/jpeg;base64,{image_b64}"


def build_client(api_key):
    return Ark(base_url="https://ark.cn-beijing.volces.com/api/v3", api_key=api_key)


def create_task(client, model_id, prompt, image_path):
    return client.content_generation.tasks.create(
        model=model_id,
        content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_to_data_uri(image_path)}},
        ],
    )


def wait_for_task(client, task_id, poll_seconds):
    while True:
        result = client.content_generation.tasks.get(task_id=task_id)
        print(f"status: {result.status}", flush=True)
        if result.status == "succeeded":
            return result
        if result.status == "failed":
            raise RuntimeError(f"任务失败: {result.error}")
        time.sleep(poll_seconds)


def download_and_extract(file_url, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = output_dir.with_suffix(".zip")
    response = requests.get(file_url, timeout=120)
    response.raise_for_status()
    zip_path.write_bytes(response.content)

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(output_dir)
    zip_path.unlink()

    glb_files = sorted(output_dir.glob("**/*.glb"))
    if not glb_files:
        raise RuntimeError(f"解压后没有找到 glb 文件: {output_dir}")
    return glb_files[0]


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a 3D model from one image.")
    parser.add_argument("--image", default="inputs/met_bulul_DP320246.jpg", type=Path)
    parser.add_argument("--output-dir", default="output_model", type=Path)
    parser.add_argument("--model-id", default=os.getenv("ARK_3D_MODEL_ID", DEFAULT_MODEL_ID))
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--poll-seconds", default=30, type=int)
    return parser.parse_args()


def main():
    load_dotenv()
    args = parse_args()
    api_key = os.getenv("ARK_API_KEY") or os.getenv("API-KEY")
    if not api_key:
        raise RuntimeError("请在 .env 里配置 ARK_API_KEY 或 API-KEY")
    if not args.image.exists():
        raise FileNotFoundError(f"找不到图片: {args.image}")

    client = build_client(api_key)
    print("----- create 3D generation task -----", flush=True)
    task = create_task(client, args.model_id, args.prompt, args.image)
    print(f"task_id: {task.id}", flush=True)

    print("----- polling task status -----", flush=True)
    result = wait_for_task(client, task.id, args.poll_seconds)
    glb_path = download_and_extract(result.content.file_url, args.output_dir)
    print(f"模型文件: {glb_path.resolve()}", flush=True)
    print(f"解压目录: {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()

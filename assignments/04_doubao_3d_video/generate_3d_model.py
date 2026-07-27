import argparse
import json
import os
import shutil
import zipfile
from pathlib import Path

from doubao_utils import build_client, download_file, find_urls, image_to_data_uri, object_to_plain_data, wait_for_task


DEFAULT_MODEL_ID = "doubao-seed3d-2-0-260328"
DEFAULT_PROMPT = (
    "Create a complete 3D model of the artifact in the reference image. "
    "Keep the full body, facial features, hands, legs, rectangular base, dark aged wood texture, "
    "and original proportions. Use clean geometry, no text, no watermark, no extra objects. "
    "--subdivisionlevel medium --fileformat glb"
)


def create_task(client, model_id, prompt, image_path):
    return client.content_generation.tasks.create(
        model=model_id,
        content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_to_data_uri(image_path)}},
        ],
    )


def get_result_file_url(result):
    data = object_to_plain_data(result)
    content = data.get("content", {}) if isinstance(data, dict) else {}
    if isinstance(content, dict) and content.get("file_url"):
        return content["file_url"]

    urls = find_urls(data)
    if not urls:
        raise RuntimeError("任务成功，但返回结果中没有找到可下载链接。")
    return urls[0]


def extract_zip(zip_path, output_dir):
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(output_dir)

    glb_files = sorted(output_dir.glob("**/*.glb"))
    if not glb_files:
        raise RuntimeError(f"解压完成，但没有找到 glb 文件: {output_dir}")
    return glb_files[0]


def save_task_json(result, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(object_to_plain_data(result), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Use Doubao Seed3D to generate a GLB model from one image.")
    parser.add_argument("--image", default="inputs/met_bulul_DP320246.jpg", type=Path)
    parser.add_argument("--output-dir", default="output_model", type=Path)
    parser.add_argument("--model-id", default=os.getenv("ARK_3D_MODEL_ID", DEFAULT_MODEL_ID))
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--poll-seconds", default=30, type=int)
    parser.add_argument("--timeout-seconds", default=3600, type=int)
    parser.add_argument("--task-json", default="results/seed3d_task_result.json", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    client = build_client()

    print("create 3D generation task", flush=True)
    task = create_task(client, args.model_id, args.prompt, args.image)
    print(f"task_id: {task.id}", flush=True)

    result = wait_for_task(
        client=client,
        task_id=task.id,
        poll_seconds=args.poll_seconds,
        timeout_seconds=args.timeout_seconds,
    )
    save_task_json(result, args.task_json)

    file_url = get_result_file_url(result)
    zip_path = args.output_dir.with_suffix(".zip")
    download_file(file_url, zip_path)
    glb_path = extract_zip(zip_path, args.output_dir)
    zip_path.unlink(missing_ok=True)

    print(f"task result: {args.task_json.resolve()}", flush=True)
    print(f"model file: {glb_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()

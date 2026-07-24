import os
import base64
import shutil
import time
import zipfile
from pathlib import Path

from dotenv import load_dotenv
import requests
from volcenginesdkarkruntime import Ark


IMAGE_PATH = Path("inputs/met_bulul_DP320246.jpg")
OUTPUT_ZIP = Path("output_model.zip")
OUTPUT_DIR = Path("output_model")
DEFAULT_MODEL_ID = "doubao-seed3d-2-0-260328"
PROMPT = (
    "Create a complete 3D model of the museum artifact in the reference image: "
    "a standing dark wooden human figure on a rectangular base. Preserve the full body, "
    "facial features, hands, legs, base blocks, aged wood texture, and museum-object proportions. "
    "No text, no watermark, no extra objects, no display stand, clean geometry. "
    "--subdivisionlevel medium --fileformat glb"
)


def image_to_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def main():
    load_dotenv()
    api_key = os.getenv("ARK_API_KEY") or os.getenv("API-KEY")
    if not api_key:
        raise RuntimeError("请在 .env 里配置 ARK_API_KEY 或 API-KEY")
    if not IMAGE_PATH.exists():
        raise FileNotFoundError(f"找不到图片: {IMAGE_PATH}")

    image_b64 = image_to_base64(IMAGE_PATH)
    data_uri = f"data:image/jpeg;base64,{image_b64}"

    client = Ark(
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        api_key=api_key,
    )

    print("----- create 3D generation task -----", flush=True)
    model_id = os.getenv("ARK_3D_MODEL_ID", DEFAULT_MODEL_ID)
    create_result = client.content_generation.tasks.create(
        model=model_id,
        content=[
            {"type": "text", "text": PROMPT},
            {"type": "image_url", "image_url": {"url": data_uri}},
        ],
    )
    print(f"task_id: {create_result.id}", flush=True)

    print("----- polling task status -----", flush=True)
    task_id = create_result.id
    while True:
        get_result = client.content_generation.tasks.get(task_id=task_id)
        status = get_result.status
        print(f"status: {status}", flush=True)
        if status == "succeeded":
            break
        if status == "failed":
            raise RuntimeError(f"任务失败: {get_result.error}")
        time.sleep(30)

    file_url = get_result.content.file_url
    response = requests.get(file_url, timeout=120)
    response.raise_for_status()
    OUTPUT_ZIP.write_bytes(response.content)

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(exist_ok=True)
    with zipfile.ZipFile(OUTPUT_ZIP) as archive:
        archive.extractall(OUTPUT_DIR)

    OUTPUT_ZIP.unlink()
    glb_files = sorted(OUTPUT_DIR.glob("**/*.glb"))
    if not glb_files:
        raise RuntimeError(f"解压后没有找到 glb 文件: {OUTPUT_DIR}")

    print(f"模型文件: {glb_files[0].resolve()}", flush=True)
    print(f"解压目录: {OUTPUT_DIR.resolve()}", flush=True)


if __name__ == "__main__":
    main()

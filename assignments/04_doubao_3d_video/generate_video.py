import os
import base64
import time
from pathlib import Path

from dotenv import load_dotenv
from volcenginesdkarkruntime import Ark


IMAGE_PATH = Path("inputs/met_bulul_DP320246.jpg")
DEFAULT_MODEL_ID = "doubao-seedance-1-0-pro-250528"
PROMPT = (
    "Use the reference image as the exact subject. Create a clean museum-style turntable video "
    "of the wooden artifact slowly rotating 360 degrees around its vertical axis. Keep the figure "
    "upright, centered, full body visible, original dark aged wood texture, rectangular base, and "
    "neutral plain background. Camera stays fixed at eye level with smooth motion. "
    "No text, no captions, no watermark, no extra objects, no hands, no scene change."
)


def image_to_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


if __name__ == "__main__":
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

    print("----- create image-to-video task -----", flush=True)
    model_id = os.getenv("ARK_VIDEO_MODEL_ID", DEFAULT_MODEL_ID)
    resp = client.content_generation.tasks.create(
        model=model_id,
        content=[
            {"type": "text", "text": PROMPT},
            {"type": "image_url", "image_url": {"url": data_uri}},
        ],
        generate_audio=False,
        ratio="adaptive",
        duration=5,
        watermark=False,
    )
    print(f"task_id: {resp.id}", flush=True)

    while True:
        result = client.content_generation.tasks.get(task_id=resp.id)
        print(f"status: {result.status}", flush=True)
        if result.status == "succeeded":
            print(result)
            break
        if result.status == "failed":
            raise RuntimeError(f"任务失败: {result.error}")
        time.sleep(30)

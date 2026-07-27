import argparse
import base64
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from volcenginesdkarkruntime import Ark


DEFAULT_MODEL_ID = "doubao-seedance-1-0-pro-250528"
DEFAULT_PROMPT = (
    "Use the reference image as the exact subject. Create a clean museum-style turntable video "
    "of the wooden artifact slowly rotating 360 degrees around its vertical axis. Keep the figure "
    "upright, centered, full body visible, original dark aged wood texture, rectangular base, and "
    "neutral plain background. Camera stays fixed at eye level with smooth motion. "
    "No text, no captions, no watermark, no extra objects, no hands, no scene change."
)


def image_to_data_uri(image_path):
    image_b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:image/jpeg;base64,{image_b64}"


def build_client(api_key):
    return Ark(base_url="https://ark.cn-beijing.volces.com/api/v3", api_key=api_key)


def create_video_task(client, model_id, prompt, image_path, duration):
    return client.content_generation.tasks.create(
        model=model_id,
        content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_to_data_uri(image_path)}},
        ],
        generate_audio=False,
        ratio="adaptive",
        duration=duration,
        watermark=False,
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


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a short video from one image.")
    parser.add_argument("--image", default="inputs/met_bulul_DP320246.jpg", type=Path)
    parser.add_argument("--model-id", default=os.getenv("ARK_VIDEO_MODEL_ID", DEFAULT_MODEL_ID))
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--duration", default=5, type=int)
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
    print("----- create image-to-video task -----", flush=True)
    task = create_video_task(
        client=client,
        model_id=args.model_id,
        prompt=args.prompt,
        image_path=args.image,
        duration=args.duration,
    )
    print(f"task_id: {task.id}", flush=True)

    result = wait_for_task(client, task.id, args.poll_seconds)
    print(result)


if __name__ == "__main__":
    main()

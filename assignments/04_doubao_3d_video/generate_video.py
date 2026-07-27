import argparse
import json
import os
from pathlib import Path

from doubao_utils import build_client, download_file, find_urls, image_to_data_uri, object_to_plain_data, wait_for_task


DEFAULT_MODEL_ID = "doubao-seedance-1-0-pro-250528"
DEFAULT_PROMPT = (
    "Use the reference image as the exact subject. Create a museum-style turntable video of the "
    "wooden artifact slowly rotating 360 degrees around its vertical axis. Keep the figure upright, "
    "centered, full body visible, original dark aged wood texture, rectangular base, and plain neutral "
    "background. Fixed camera, smooth motion, no text, no watermark, no extra objects."
)


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


def choose_video_url(result):
    data = object_to_plain_data(result)
    urls = find_urls(data)
    if not urls:
        raise RuntimeError("任务成功，但返回结果中没有找到视频链接。")

    video_urls = [
        url for url in urls if any(token in url.lower().split("?")[0] for token in [".mp4", ".mov", ".webm"])
    ]
    return video_urls[0] if video_urls else urls[0]


def save_task_json(result, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(object_to_plain_data(result), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Use Doubao image-to-video API to generate a turntable video.")
    parser.add_argument("--image", default="inputs/met_bulul_DP320246.jpg", type=Path)
    parser.add_argument("--model-id", default=os.getenv("ARK_VIDEO_MODEL_ID", DEFAULT_MODEL_ID))
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--duration", default=5, type=int)
    parser.add_argument("--poll-seconds", default=30, type=int)
    parser.add_argument("--timeout-seconds", default=3600, type=int)
    parser.add_argument("--output", default="results/doubao_generated_video.mp4", type=Path)
    parser.add_argument("--task-json", default="results/video_task_result.json", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    client = build_client()

    print("create image-to-video task", flush=True)
    task = create_video_task(
        client=client,
        model_id=args.model_id,
        prompt=args.prompt,
        image_path=args.image,
        duration=args.duration,
    )
    print(f"task_id: {task.id}", flush=True)

    result = wait_for_task(
        client=client,
        task_id=task.id,
        poll_seconds=args.poll_seconds,
        timeout_seconds=args.timeout_seconds,
    )
    save_task_json(result, args.task_json)

    video_url = choose_video_url(result)
    download_file(video_url, args.output)
    print(f"task result: {args.task_json.resolve()}", flush=True)
    print(f"video file: {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()

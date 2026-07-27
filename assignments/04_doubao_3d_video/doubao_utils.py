import base64
import mimetypes
import os
import time
from pathlib import Path
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv
from volcenginesdkarkruntime import Ark


ARK_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"


def load_api_key():
    load_dotenv()
    api_key = os.getenv("ARK_API_KEY") or os.getenv("API-KEY")
    if not api_key:
        raise RuntimeError("请先在 .env 中配置 ARK_API_KEY。")
    return api_key


def build_client(api_key=None):
    return Ark(base_url=ARK_BASE_URL, api_key=api_key or load_api_key())


def image_to_data_uri(image_path):
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"找不到输入图片: {image_path}")

    mime_type = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
    image_b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{image_b64}"


def wait_for_task(client, task_id, poll_seconds=30, timeout_seconds=3600):
    start = time.time()
    while True:
        result = client.content_generation.tasks.get(task_id=task_id)
        status = getattr(result, "status", "")
        print(f"status: {status}", flush=True)

        if status == "succeeded":
            return result
        if status == "failed":
            raise RuntimeError(f"任务失败: {getattr(result, 'error', '')}")
        if time.time() - start > timeout_seconds:
            raise TimeoutError(f"任务等待超时: {task_id}")

        time.sleep(poll_seconds)


def object_to_plain_data(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {key: object_to_plain_data(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [object_to_plain_data(item) for item in value]
    if hasattr(value, "model_dump"):
        return object_to_plain_data(value.model_dump())
    if hasattr(value, "__dict__"):
        return {
            key: object_to_plain_data(item)
            for key, item in vars(value).items()
            if not key.startswith("_")
        }
    return str(value)


def find_urls(value):
    urls = []
    if isinstance(value, str):
        parsed = urlparse(value)
        if parsed.scheme in {"http", "https"}:
            urls.append(value)
    elif isinstance(value, dict):
        for item in value.values():
            urls.extend(find_urls(item))
    elif isinstance(value, list):
        for item in value:
            urls.extend(find_urls(item))
    return urls


def download_file(url, output_path, timeout=300):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with output_path.open("wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)
    return output_path

import argparse
import base64
import json
import random
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from urllib.parse import parse_qs, urlparse

import torch
from torchvision import datasets

from .data import CIFAR10_CLASSES, build_predict_transform
from .inference import get_device, load_model_from_checkpoint


HTML = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CIFAR-10 预测演示</title>
  <style>
    :root {
      --bg: #f6f7f9;
      --panel: #ffffff;
      --line: #d9dde4;
      --text: #151922;
      --muted: #626b7a;
      --accent: #1769e0;
      --accent-dark: #0f55b8;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text);
    }
    .app {
      width: min(980px, calc(100vw - 32px));
      margin: 32px auto;
    }
    h1 {
      margin: 0 0 18px;
      font-size: 26px;
      font-weight: 700;
    }
    .layout {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 18px;
      align-items: stretch;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 18px;
      min-height: 420px;
    }
    .title {
      margin: 0 0 12px;
      font-size: 17px;
      font-weight: 700;
    }
    .image-wrap {
      display: grid;
      place-items: center;
      height: 280px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #eef1f5;
      overflow: hidden;
    }
    img {
      width: 256px;
      height: 256px;
      image-rendering: pixelated;
      object-fit: contain;
    }
    button {
      width: 100%;
      height: 44px;
      margin-top: 16px;
      border: 0;
      border-radius: 7px;
      background: var(--accent);
      color: white;
      font-size: 15px;
      font-weight: 700;
      cursor: pointer;
    }
    button:hover { background: var(--accent-dark); }
    button:disabled {
      background: #9aa5b5;
      cursor: not-allowed;
    }
    .meta, .result {
      margin-top: 14px;
      color: var(--muted);
      line-height: 1.7;
      font-size: 14px;
    }
    .result strong {
      color: var(--text);
      font-size: 22px;
    }
    .prob {
      margin-top: 10px;
      height: 12px;
      border-radius: 999px;
      background: #e3e7ee;
      overflow: hidden;
    }
    .bar {
      width: 0%;
      height: 100%;
      background: var(--accent);
      transition: width 180ms ease;
    }
    @media (max-width: 760px) {
      .layout { grid-template-columns: 1fr; }
      .panel { min-height: 0; }
    }
  </style>
</head>
<body>
  <main class="app">
    <h1>CIFAR-10 图片预测演示</h1>
    <section class="layout">
      <div class="panel">
        <p class="title">随机图片</p>
        <div class="image-wrap">
          <img id="sample" alt="CIFAR-10 sample">
        </div>
        <button id="randomBtn">随机刷新一张测试图片</button>
        <div class="meta" id="meta">正在加载...</div>
      </div>
      <div class="panel">
        <p class="title">模型预测</p>
        <button id="predictBtn" disabled>预测当前图片</button>
        <div class="result" id="result">请先随机刷新一张图片。</div>
        <div class="prob"><div class="bar" id="bar"></div></div>
      </div>
    </section>
  </main>
  <script>
    let currentIndex = null;
    const sample = document.getElementById("sample");
    const meta = document.getElementById("meta");
    const result = document.getElementById("result");
    const bar = document.getElementById("bar");
    const randomBtn = document.getElementById("randomBtn");
    const predictBtn = document.getElementById("predictBtn");

    async function loadRandom() {
      randomBtn.disabled = true;
      predictBtn.disabled = true;
      result.textContent = "等待预测。";
      bar.style.width = "0%";
      try {
        const res = await fetch("/api/random");
        const data = await res.json();
        currentIndex = data.index;
        sample.src = data.image;
        meta.textContent = `测试集序号：${data.index} / 9999，真实类别：${data.label}`;
        predictBtn.disabled = false;
      } finally {
        randomBtn.disabled = false;
      }
    }

    async function predict() {
      if (currentIndex === null) return;
      predictBtn.disabled = true;
      result.textContent = "预测中...";
      try {
        const res = await fetch(`/api/predict?index=${currentIndex}`);
        const data = await res.json();
        const pct = (data.probability * 100).toFixed(2);
        result.innerHTML = `预测类别：<br><strong>${data.predicted}</strong><br>置信度：${pct}%`;
        bar.style.width = `${pct}%`;
      } finally {
        predictBtn.disabled = false;
      }
    }

    randomBtn.addEventListener("click", loadRandom);
    predictBtn.addEventListener("click", predict);
    loadRandom();
  </script>
</body>
</html>
"""


class DemoState:
    def __init__(self, data_dir, checkpoint):
        self.device = get_device()
        self.dataset = datasets.CIFAR10(root=data_dir, train=False, transform=None, download=False)
        self.transform = build_predict_transform()
        self.model, _, self.device = load_model_from_checkpoint(checkpoint, self.device)

    def sample_payload(self):
        index = random.randrange(len(self.dataset))
        image, target = self.dataset[index]
        return self.image_payload(index, image, target)

    def image_payload(self, index, image, target):
        buffer = BytesIO()
        image.resize((256, 256)).save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return {
            "index": index,
            "label": CIFAR10_CLASSES[target],
            "image": f"data:image/png;base64,{encoded}",
        }

    @torch.no_grad()
    def predict_payload(self, index):
        image, target = self.dataset[index]
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        logits = self.model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        prob, pred = probs.max(dim=0)
        return {
            "index": index,
            "true_label": CIFAR10_CLASSES[target],
            "predicted": CIFAR10_CLASSES[pred.item()],
            "probability": prob.item(),
        }


def make_handler(state):
    class DemoHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self.send_html(HTML)
            elif parsed.path == "/api/random":
                self.send_json(state.sample_payload())
            elif parsed.path == "/api/predict":
                query = parse_qs(parsed.query)
                index = int(query.get("index", ["0"])[0])
                if index < 0 or index >= len(state.dataset):
                    self.send_json({"error": "index out of range"}, status=400)
                else:
                    self.send_json(state.predict_payload(index))
            else:
                self.send_json({"error": "not found"}, status=404)

        def log_message(self, fmt, *args):
            return

        def send_html(self, body):
            data = body.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def send_json(self, payload, status=200):
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return DemoHandler


def parse_args():
    parser = argparse.ArgumentParser(description="Run CIFAR-10 browser demo.")
    parser.add_argument("--data-dir", default="./data", type=str)
    parser.add_argument(
        "--checkpoint",
        default="./checkpoints_100_finetune/cifar10_wrn_best.pt",
        type=str,
    )
    parser.add_argument("--host", default="127.0.0.1", type=str)
    parser.add_argument("--port", default=8008, type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    state = DemoState(args.data_dir, args.checkpoint)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(state))
    print(f"device: {state.device}", flush=True)
    print(f"loaded checkpoint: {args.checkpoint}", flush=True)
    print(f"open: http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()

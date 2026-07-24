#!/usr/bin/env python3
import argparse
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image


DEFAULT_MODEL = ""


def find_default_model():
    preferred = [
        Path("output_model/pbr/mesh_textured_pbr.glb"),
        Path("output_model/mesh_textured_pbr.glb"),
    ]
    for path in preferred:
        if path.exists():
            return path

    candidates = sorted(Path(".").glob("**/*.glb"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError("找不到 .glb 模型文件，请用 --model 指定。")
    return candidates[0]


def look_at(eye, target, up):
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)

    forward = target - eye
    forward /= np.linalg.norm(forward)

    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-8:
        up = np.array((0.0, 1.0, 0.0), dtype=np.float64)
        right = np.cross(forward, up)
    right /= np.linalg.norm(right)

    true_up = np.cross(right, forward)
    true_up /= np.linalg.norm(true_up)

    matrix = np.eye(4)
    matrix[:3, 0] = right
    matrix[:3, 1] = true_up
    matrix[:3, 2] = -forward
    matrix[:3, 3] = eye
    return matrix


def make_renderer(width, height):
    last_error = None
    for platform in ("egl", "osmesa", None):
        if platform:
            os.environ["PYOPENGL_PLATFORM"] = platform
        else:
            os.environ.pop("PYOPENGL_PLATFORM", None)

        try:
            import pyrender

            renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)
            print(f"OpenGL backend: {platform or 'default'}")
            return pyrender, renderer
        except Exception as exc:
            last_error = exc
            for module_name in list(sys.modules):
                if module_name.startswith(("OpenGL", "pyrender")):
                    sys.modules.pop(module_name, None)

    raise RuntimeError(
        "无法创建离屏渲染器。服务器可能缺少 EGL/OSMesa/OpenGL 运行库。"
    ) from last_error


def scene_bounds(scene):
    bounds = scene.bounds
    if bounds is None:
        raise RuntimeError("无法读取模型边界。")
    center = bounds.mean(axis=0)
    extents = bounds[1] - bounds[0]
    size = float(np.max(extents))
    return center, max(size, 1e-6)


def get_up_axis(extents, requested):
    if requested != "auto":
        return {"x": 0, "y": 1, "z": 2}[requested]
    return int(np.argmax(extents))


def render_frames(model_path, frame_dir, frames, width, height, distance_multiplier, up_axis, start_angle):
    import trimesh

    pyrender, renderer = make_renderer(width, height)

    loaded = trimesh.load(model_path, force="scene")
    if loaded.is_empty:
        raise RuntimeError(f"模型为空: {model_path}")

    center, size = scene_bounds(loaded)
    up_index = get_up_axis(loaded.extents, up_axis)
    up_vector = np.zeros(3, dtype=np.float64)
    up_vector[up_index] = 1.0
    horizontal = [axis for axis in range(3) if axis != up_index]
    print(f"up axis: {'xyz'[up_index]}")

    scene = pyrender.Scene.from_trimesh_scene(loaded, bg_color=(238, 238, 235, 255))

    camera = pyrender.PerspectiveCamera(yfov=math.radians(35.0), znear=0.01)
    camera_node = scene.add(camera, pose=np.eye(4))

    key_light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    fill_light = pyrender.DirectionalLight(color=np.ones(3), intensity=1.2)
    key_pos = center + up_vector * size * 2.0
    key_pos[horizontal[0]] -= size
    key_pos[horizontal[1]] -= size
    fill_pos = center + up_vector * size * 1.5
    fill_pos[horizontal[0]] += size
    fill_pos[horizontal[1]] += size
    scene.add(key_light, pose=look_at(key_pos, center, up_vector))
    scene.add(fill_light, pose=look_at(fill_pos, center, up_vector))
    scene.ambient_light = np.array([0.35, 0.35, 0.35, 1.0])

    radius = size * distance_multiplier
    vertical_offset = size * 0.12
    target = np.array(center, dtype=np.float64)
    target[up_index] += size * 0.03

    for index in range(frames):
        angle = math.radians(start_angle) + 2.0 * math.pi * index / frames
        eye = np.array(center, dtype=np.float64)
        eye[horizontal[0]] += math.sin(angle) * radius
        eye[horizontal[1]] -= math.cos(angle) * radius
        eye[up_index] += vertical_offset
        scene.set_pose(camera_node, pose=look_at(eye, target, up_vector))
        color, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        image = Image.fromarray(color, mode="RGBA").convert("RGB")
        image.save(frame_dir / f"frame_{index:04d}.png")
        if index == 0 or (index + 1) % 10 == 0 or index == frames - 1:
            print(f"rendered {index + 1}/{frames}")

    renderer.delete()


def encode_video(frame_dir, output_path, fps):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("找不到 ffmpeg，请先安装 ffmpeg。")

    cmd = [
        ffmpeg,
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(frame_dir / "frame_%04d.png"),
        "-vf",
        "format=yuv420p",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="把 .glb 3D 模型渲染成自动旋转 mp4。")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="输入 .glb 文件路径；不传则自动查找。")
    parser.add_argument("--out", default="turntable.mp4", help="输出 mp4 文件路径。")
    parser.add_argument("--frames", type=int, default=180, help="总帧数。")
    parser.add_argument("--fps", type=int, default=30, help="视频帧率。")
    parser.add_argument("--width", type=int, default=1080, help="视频宽度。")
    parser.add_argument("--height", type=int, default=1080, help="视频高度。")
    parser.add_argument("--distance", type=float, default=1.7, help="相机距离倍数，越小主体越大。")
    parser.add_argument("--up-axis", choices=("auto", "x", "y", "z"), default="auto", help="模型竖直轴。")
    parser.add_argument("--start-angle", type=float, default=180.0, help="起始水平角度，单位为度。")
    parser.add_argument("--keep-frames", action="store_true", help="保留中间 PNG 帧。")
    args = parser.parse_args()

    model_path = Path(args.model) if args.model else find_default_model()
    if not model_path.exists():
        raise SystemExit(f"模型文件不存在: {model_path}")

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.keep_frames:
        frame_dir = output_path.with_suffix("")
        frame_dir.mkdir(parents=True, exist_ok=True)
        cleanup = False
    else:
        frame_dir_obj = tempfile.TemporaryDirectory(prefix="turntable_frames_")
        frame_dir = Path(frame_dir_obj.name)
        cleanup = True

    try:
        render_frames(
            model_path,
            frame_dir,
            args.frames,
            args.width,
            args.height,
            args.distance,
            args.up_axis,
            args.start_angle,
        )
        encode_video(frame_dir, output_path, args.fps)
    finally:
        if cleanup:
            frame_dir_obj.cleanup()

    print(f"完成: {output_path.resolve()}")


if __name__ == "__main__":
    main()

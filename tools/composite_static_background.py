"""Compose an existing PhysGaussian rollout over an inpainted static input image."""
import argparse
from pathlib import Path
import subprocess

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--dynamic", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=float, required=True)
    args = parser.parse_args()

    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(args.image)
    hole = np.zeros(image.shape[:2], dtype=np.uint8)
    for name in args.dynamic:
        mask = cv2.imread(str(args.mask_dir / f"{name}.png"), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(args.mask_dir / f"{name}.png")
        hole |= cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    static = cv2.inpaint(image, hole, 5, cv2.INPAINT_TELEA)
    args.output.mkdir(parents=True, exist_ok=True)
    frames = sorted(args.frames.glob("*.png"))
    if not frames:
        raise FileNotFoundError(f"No rendered PNGs in {args.frames}")
    for index, frame_path in enumerate(frames):
        destination = args.output / f"{index:04d}.png"
        if destination.is_file():
            continue
        rendered = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if rendered is None:
            raise ValueError(f"Unreadable frame: {frame_path}")
        base = cv2.resize(static, (rendered.shape[1], rendered.shape[0]), interpolation=cv2.INTER_AREA)
        # ponytail: PhysGaussian currently emits RGB only; brighteness above its black clear color is the available coverage proxy.
        alpha = np.clip((rendered.max(axis=2).astype(np.float32) - 5.0) / 35.0, 0.0, 1.0)[..., None]
        composed = np.uint8(rendered * alpha + base * (1.0 - alpha))
        cv2.imwrite(str(destination), composed)
    video = args.output / "output.mp4"
    subprocess.run(["ffmpeg", "-y", "-framerate", str(args.fps), "-i", str(args.output / "%04d.png"), "-vf", "crop=trunc(iw/2)*2:trunc(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(video)], check=True)
    print(video)


if __name__ == "__main__":
    main()

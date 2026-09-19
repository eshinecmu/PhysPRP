"""Reuse REST3D instance masks to create GaussianProperty LMM inputs."""

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np


def normalize(value):
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--mask_dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--case_name", default="rest3d_scene")
    args = parser.parse_args()

    image = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(args.image)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    masks = {normalize(path.stem): path for path in Path(args.mask_dir).glob("*.png")}
    output = Path(args.output_root) / args.case_name / "gpt_input" / "01"
    output.mkdir(parents=True, exist_ok=True)

    for row in manifest:
        name = normalize(str(row.get("name", "")))
        candidates = [path for mask_name, path in masks.items() if name == mask_name or name in mask_name or mask_name in name]
        if len(candidates) != 1:
            raise ValueError(f"Expected one REST3D mask for {row.get('name')!r}, found {len(candidates)}")
        raw = cv2.imread(str(candidates[0]), cv2.IMREAD_UNCHANGED)
        if raw.shape[:2] != image.shape[:2]:
            raw = cv2.resize(raw, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        if raw.ndim == 3 and raw.shape[2] == 4:
            mask = raw[:, :, 3] > 0
        elif raw.ndim == 3:
            mask = np.any(raw[:, :, :3] > 8, axis=2)
        else:
            mask = raw > 0
        ys, xs = np.where(mask)
        if not len(xs):
            raise ValueError(f"Empty REST3D mask: {candidates[0]}")
        overlay = image.copy()
        overlay[mask] = (0.45 * overlay[mask] + 0.55 * np.array([0, 0, 255])).astype(np.uint8)
        part = np.zeros_like(image)
        part[mask] = image[mask]
        crop = part[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        crop = cv2.resize(crop, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_AREA)
        panel = np.concatenate((image, overlay, crop), axis=1)
        cv2.imwrite(str(output / f"{int(row['object_id']):04d}.png"), panel)
    print(f"wrote {len(manifest)} GaussianProperty inputs to {output}")


if __name__ == "__main__":
    main()

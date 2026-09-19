"""Frame-aligned rendered-video comparison without extra dependencies."""

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


def ssim(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu_a = cv2.GaussianBlur(a, (11, 11), 1.5)
    mu_b = cv2.GaussianBlur(b, (11, 11), 1.5)
    sigma_a = cv2.GaussianBlur(a * a, (11, 11), 1.5) - mu_a * mu_a
    sigma_b = cv2.GaussianBlur(b * b, (11, 11), 1.5) - mu_b * mu_b
    sigma_ab = cv2.GaussianBlur(a * b, (11, 11), 1.5) - mu_a * mu_b
    score = ((2 * mu_a * mu_b + c1) * (2 * sigma_ab + c2)) / (
        (mu_a * mu_a + mu_b * mu_b + c1) * (sigma_a + sigma_b + c2)
    )
    return float(np.mean(score))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True)
    parser.add_argument("--gt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_frames", type=int, default=0)
    args = parser.parse_args()

    pred, gt = cv2.VideoCapture(args.pred), cv2.VideoCapture(args.gt)
    if not pred.isOpened() or not gt.isOpened():
        raise FileNotFoundError("Could not open prediction or GT video")
    rows, previous_pred, previous_gt = [], None, None
    while not args.max_frames or len(rows) < args.max_frames:
        ok_pred, frame_pred = pred.read()
        ok_gt, frame_gt = gt.read()
        if not ok_pred or not ok_gt:
            break
        if frame_pred.shape[:2] != frame_gt.shape[:2]:
            frame_pred = cv2.resize(frame_pred, (frame_gt.shape[1], frame_gt.shape[0]), interpolation=cv2.INTER_AREA)
        difference = frame_pred.astype(np.float32) - frame_gt.astype(np.float32)
        mse = float(np.mean(difference * difference))
        temporal = None
        if previous_pred is not None:
            pred_motion = frame_pred.astype(np.float32) - previous_pred.astype(np.float32)
            gt_motion = frame_gt.astype(np.float32) - previous_gt.astype(np.float32)
            temporal = float(np.mean((pred_motion - gt_motion) ** 2))
        rows.append({"frame": len(rows), "mse": mse,
                     "psnr": 100.0 if mse == 0 else 10.0 * math.log10(255.0 ** 2 / mse),
                     "ssim": ssim(frame_pred, frame_gt), "temporal_mse": temporal})
        previous_pred, previous_gt = frame_pred, frame_gt
    pred.release(); gt.release()
    if not rows:
        raise RuntimeError("No aligned frames were read")
    valid_temporal = [row["temporal_mse"] for row in rows if row["temporal_mse"] is not None]
    result = {"frames": len(rows),
              "mean_psnr": float(np.mean([row["psnr"] for row in rows])),
              "mean_ssim": float(np.mean([row["ssim"] for row in rows])),
              "mean_mse": float(np.mean([row["mse"] for row in rows])),
              "mean_temporal_mse": float(np.mean(valid_temporal)) if valid_temporal else None,
              "per_frame": rows}
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "per_frame"}, indent=2))


if __name__ == "__main__":
    main()

"""Cheap pre-GPU validation for an adapted REST3D object."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from prepare_rest3d_object import read_obj_vertices


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--canonical-obj", type=Path, required=True)
    args = parser.parse_args()
    from plyfile import PlyData

    ply = args.model_path / "point_cloud" / "iteration_1" / "point_cloud.ply"
    vertex = PlyData.read(str(ply))["vertex"].data
    xyz = np.column_stack([vertex[name] for name in ("x", "y", "z")]).astype(float)
    scales = np.column_stack([vertex[name] for name in ("scale_0", "scale_1", "scale_2")]).astype(float)
    opacity = np.asarray(vertex["opacity"], dtype=float)
    mesh = read_obj_vertices(args.canonical_obj)
    if not np.isfinite(xyz).all() or not np.isfinite(scales).all():
        raise ValueError("non-finite Gaussian values")
    if len(xyz) == 0 or np.any(np.exp(scales) <= 0):
        raise ValueError("Gaussian count or physical scale is invalid")
    lo, hi = mesh.min(axis=0), mesh.max(axis=0)
    outside = np.logical_or(xyz < lo - 1e-5, xyz > hi + 1e-5).any(axis=1)
    if outside.mean() > 0.05:
        raise ValueError(f"{outside.sum()}/{len(xyz)} Gaussian centers fall outside the canonical mesh AABB")
    print({"status": "ok", "gaussians": len(xyz), "outside_aabb": int(outside.sum()), "positive_opacity": int((opacity > 0).sum()), "support_plane_y": float(lo[1])})


if __name__ == "__main__":
    main()

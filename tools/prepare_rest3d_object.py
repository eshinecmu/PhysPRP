"""Adapt one REST3D/SAM3D object into a PhysGaussian model directory.

REST3D's canonical OBJ is the source of truth for placement.  The matching
raw OBJ and SAM3D PLY are transformed with the same rigid transform, so the
PLY is never treated as a drop-in PhysGaussian checkpoint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def read_obj_vertices(path: Path) -> np.ndarray:
    vertices = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("v "):
            fields = line.split()
            vertices.append([float(fields[1]), float(fields[2]), float(fields[3])])
    result = np.asarray(vertices, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != 3 or len(result) < 3:
        raise ValueError(f"OBJ has fewer than three vertices: {path}")
    return result


def rigid_transform(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if src.shape != dst.shape:
        raise ValueError(f"OBJ vertex counts differ: {src.shape} vs {dst.shape}")
    src_center = src.mean(axis=0)
    dst_center = dst.mean(axis=0)
    u, _, vt = np.linalg.svd((src - src_center).T @ (dst - dst_center))
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(vt.T @ u.T)
    rotation = vt.T @ correction @ u.T
    translation = dst_center - rotation @ src_center
    return rotation, translation


def quaternion_from_matrix(rotation: np.ndarray) -> np.ndarray:
    """Return [w, x, y, z] for a proper 3x3 rotation matrix."""
    trace = np.trace(rotation)
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2
        return np.array([0.25 * s, (rotation[2, 1] - rotation[1, 2]) / s,
                         (rotation[0, 2] - rotation[2, 0]) / s,
                         (rotation[1, 0] - rotation[0, 1]) / s])
    axis = int(np.argmax(np.diag(rotation)))
    nxt = (axis + 1) % 3
    last = (axis + 2) % 3
    s = np.sqrt(1.0 + rotation[axis, axis] - rotation[nxt, nxt] - rotation[last, last]) * 2
    q = np.zeros(4)
    q[axis + 1] = 0.25 * s
    q[0] = (rotation[last, nxt] - rotation[nxt, last]) / s
    q[nxt + 1] = (rotation[nxt, axis] + rotation[axis, nxt]) / s
    q[last + 1] = (rotation[last, axis] + rotation[axis, last]) / s
    return q


def transform_ply(source: Path, target: Path, rotation: np.ndarray, translation: np.ndarray) -> int:
    from plyfile import PlyData, PlyElement

    data = PlyData.read(str(source))
    vertex = data["vertex"].data.copy()
    names = vertex.dtype.names or ()
    required = {"x", "y", "z"}
    if not required.issubset(names):
        raise ValueError(f"Gaussian PLY is missing xyz fields: {source}")
    xyz = np.column_stack([vertex[name] for name in ("x", "y", "z")]).astype(np.float64)
    vertex["x"], vertex["y"], vertex["z"] = (rotation @ xyz.T + translation[:, None]).astype(vertex["x"].dtype)

    # Rigid transforms preserve scale eigenvalues; compose the Gaussian frame.
    rot_q = quaternion_from_matrix(rotation)
    rot_fields = [name for name in ("rot_0", "rot_1", "rot_2", "rot_3") if name in names]
    if len(rot_fields) == 4:
        local_q = np.column_stack([vertex[name] for name in rot_fields]).astype(np.float64)
        world_q = np.column_stack([
            rot_q[0] * local_q[:, 0] - rot_q[1] * local_q[:, 1] - rot_q[2] * local_q[:, 2] - rot_q[3] * local_q[:, 3],
            rot_q[0] * local_q[:, 1] + rot_q[1] * local_q[:, 0] + rot_q[2] * local_q[:, 3] - rot_q[3] * local_q[:, 2],
            rot_q[0] * local_q[:, 2] - rot_q[1] * local_q[:, 3] + rot_q[2] * local_q[:, 0] + rot_q[3] * local_q[:, 1],
            rot_q[0] * local_q[:, 3] + rot_q[1] * local_q[:, 2] - rot_q[2] * local_q[:, 1] + rot_q[3] * local_q[:, 0],
        ], axis=1)
        world_q /= np.linalg.norm(world_q, axis=1, keepdims=True).clip(min=1e-12)
        for i, name in enumerate(rot_fields):
            vertex[name] = world_q[:, i].astype(vertex[name].dtype)
    target.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(vertex, "vertex")], text=False).write(str(target))
    return len(vertex)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-obj", type=Path, required=True)
    parser.add_argument("--canonical-obj", type=Path, required=True)
    parser.add_argument("--raw-gaussian", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--iteration", type=int, default=1)
    args = parser.parse_args()

    raw = read_obj_vertices(args.raw_obj)
    canonical = read_obj_vertices(args.canonical_obj)
    rotation, translation = rigid_transform(raw, canonical)
    output_ply = args.model_path / "point_cloud" / f"iteration_{args.iteration}" / "point_cloud.ply"
    count = transform_ply(args.raw_gaussian, output_ply, rotation, translation)
    bounds = np.column_stack([canonical.min(axis=0), canonical.max(axis=0)])
    manifest = {
        "source": {"raw_obj": str(args.raw_obj), "canonical_obj": str(args.canonical_obj), "raw_gaussian": str(args.raw_gaussian)},
        "coordinate_convention": "REST3D canonical scene; +Y is up and the support plane is y=0",
        "rigid_rotation": rotation.tolist(),
        "rigid_translation": translation.tolist(),
        "canonical_bounds_xyz": bounds.tolist(),
        "gaussian_count": count,
        "note": "PhysGaussian preprocessing still normalizes this object before MPM; this manifest records the pre-normalization alignment.",
    }
    args.model_path.mkdir(parents=True, exist_ok=True)
    (args.model_path / "rest3d_alignment.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output_ply), "gaussians": count, "min_xyz": canonical.min(axis=0).tolist(), "max_xyz": canonical.max(axis=0).tolist()}))


if __name__ == "__main__":
    main()

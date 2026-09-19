"""Small two-object MPM input with an explicit Gaussian render subset."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def make_scene(points_per_axis: int = 5, spacing: float = 0.08) -> dict[str, np.ndarray]:
    if points_per_axis < 3 or spacing <= 0 or (points_per_axis - 1) * spacing >= 0.36:
        raise ValueError("Need at least 3 points per axis and a positive, non-overlapping spacing")

    lattice = np.stack(
        np.meshgrid(*([np.arange(points_per_axis)] * 3), indexing="ij"), axis=-1
    ).reshape(-1, 3)
    local = (lattice - (points_per_axis - 1) / 2) * spacing
    surface = np.any((lattice == 0) | (lattice == points_per_axis - 1), axis=1)
    centers = np.array([[0.82, 1.0, 1.0], [1.18, 1.0, 1.0]])
    x = np.concatenate([local + center for center in centers]).astype(np.float32)
    object_id = np.repeat(np.arange(2), len(local)).astype(np.int64)
    material_id = (x[:, 2] > 1.0).astype(np.int64)
    density = np.array([900.0, 1200.0], dtype=np.float32)[material_id]
    E = np.array([2e4, 8e4], dtype=np.float32)[material_id]
    nu = np.array([0.35, 0.30], dtype=np.float32)[material_id]
    v = np.zeros_like(x)
    v[:, 0] = np.where(object_id == 0, 0.1, -0.1)

    # ponytail: Uniform voxel volume suffices here; use occupancy-based volumes for reconstructed scenes.
    return {
        "x": x,
        "v": v,
        "F": np.broadcast_to(np.eye(3, dtype=np.float32), (len(x), 3, 3)).copy(),
        "volume": np.full(len(x), spacing**3, dtype=np.float32),
        "mass": density * spacing**3,
        "density": density,
        "E": E,
        "nu": nu,
        "object_id": object_id,
        "material_id": material_id,
        "render_particle_indices": np.flatnonzero(np.tile(surface, 2)).astype(np.int64),
    }


def self_check() -> None:
    scene = make_scene()
    n = len(scene["x"])
    render = scene["render_particle_indices"]
    assert n == 250 and 0 < len(render) < n
    assert np.any(np.diff(render) > 1)  # Interior particles make rendering non-prefix.
    assert set(scene["object_id"]) == {0, 1}
    for object_id in (0, 1):
        assert set(scene["material_id"][scene["object_id"] == object_id]) == {0, 1}
    assert np.allclose(scene["mass"], scene["density"] * scene["volume"])
    assert scene["v"][scene["object_id"] == 0, 0].mean() > 0
    assert scene["v"][scene["object_id"] == 1, 0].mean() < 0
    left = scene["x"][scene["object_id"] == 0, 0]
    right = scene["x"][scene["object_id"] == 1, 0]
    assert 0 < right.min() - left.max() < 0.1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Write the scene as an NPZ file")
    args = parser.parse_args()
    self_check()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output, **make_scene())
        print(args.output)
    else:
        print("synthetic two-object self-check passed")

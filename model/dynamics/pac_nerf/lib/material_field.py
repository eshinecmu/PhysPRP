"""PIXIE-to-PAC-NeRF data adapter; no material or simulation model lives here."""

import numpy as np
import torch


def particles_from_material(path, xyz_min, xyz_max):
    mask = torch.as_tensor(np.load(path)["mask"], device=xyz_min.device)
    indices = mask.nonzero().to(xyz_min.dtype)
    if not indices.shape[0]:
        raise ValueError("PIXIE material field contains no occupied voxels")
    scale = torch.tensor(mask.shape, dtype=xyz_min.dtype, device=xyz_min.device) - 1
    return xyz_min + indices / scale.clamp_min(1) * (xyz_max - xyz_min)


def particle_materials(path, particles, alphas):
    field = np.load(path)
    target_min, target_max = particles.amin(0), particles.amax(0)
    # ponytail: bbox registration is the existing common contract; replace it
    # with an explicit REST3D-to-PAC transform when calibration is available.
    normalized = (particles - target_min) / (target_max - target_min).clamp_min(1e-8)
    shape = torch.tensor(field["density"].shape, device=particles.device)
    indices = (normalized * (shape - 1)).round().long().clamp_min(0)
    indices = torch.minimum(indices, shape - 1)

    def values(name):
        grid = torch.as_tensor(field[name], dtype=particles.dtype, device=particles.device)
        return grid[indices[:, 0], indices[:, 1], indices[:, 2]]

    rho, E, nu = values("density"), values("E"), values("nu").clamp(-0.49, 0.49)
    mu = E / (2 * (1 + nu))
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    return alphas * rho, alphas.square() * alphas * mu, alphas.square() * alphas * lam

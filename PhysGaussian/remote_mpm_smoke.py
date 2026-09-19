"""Run one CUDA MPM substep on the synthetic relation-scene."""

import os
import sys

import torch

from relation_mpm import RelationMPM
from synthetic_two_object import make_scene
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "mpm_solver_warp"))
from mpm_solver_warp import MPM_Simulator_WARP


def make_solver(scene):
    device = "cuda:0"
    x = torch.from_numpy(scene["x"]).to(device)
    volume = torch.from_numpy(scene["volume"]).to(device)
    solver = MPM_Simulator_WARP(len(x), device=device)
    solver.load_initial_data_from_torch(x, volume, n_grid=32, grid_lim=2.0, device=device)
    solver.set_parameters_dict({"material": "jelly", "E": 2e4, "nu": .35, "density": 900., "g": [0., 0., 0.]}, device=device)
    solver.finalize_mu_lam(device=device)
    return solver


def main():
    scene = make_scene()
    step = 1e-4
    baseline = make_solver(scene)
    baseline.p2g2p(0, step, device="cuda:0")
    relation = make_solver(scene)
    model = RelationMPM(n_materials=2, local_radius=.095, contact_radius=.09).cuda()
    labels = (torch.from_numpy(scene["object_id"]), torch.from_numpy(scene["material_id"]))
    relation.relation_step = lambda solver, dt: model.step_solver(solver, *labels, dt)
    relation.p2g2p(0, step, device="cuda:0")
    x0 = baseline.export_particle_x_to_torch()
    x1 = relation.export_particle_x_to_torch()
    assert torch.equal(x0, x1), "zero-initialized relation model must reproduce baseline"
    with torch.no_grad():
        model.force_output.bias[0] = .1
    relation = make_solver(scene)
    relation.relation_step = lambda solver, dt: model.step_solver(solver, *labels, dt)
    relation.p2g2p(0, step, device="cuda:0")
    x2 = relation.export_particle_x_to_torch()
    v2 = relation.export_particle_v_to_torch()
    v0 = baseline.export_particle_v_to_torch()
    gap0 = x0[scene["object_id"] == 1, 0].min() - x0[scene["object_id"] == 0, 0].max()
    gap1 = x2[scene["object_id"] == 1, 0].min() - x2[scene["object_id"] == 0, 0].max()
    print(f"baseline_gap={gap0.item():.8f} relation_gap={gap1.item():.8f} delta={gap1.item()-gap0.item():.8f} max_dv={(v2-v0).abs().max().item():.8e}")


if __name__ == "__main__":
    main()

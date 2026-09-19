import sys
import os
import types

_gs_root = os.path.abspath("gaussian-splatting")
sys.path.insert(0, _gs_root)
# The runtime checkout has a namespace-style ``utils`` directory while some
# environments ship an unrelated regular ``utils`` package. Pin imports to GS.
_gs_utils = types.ModuleType("utils")
_gs_utils.__path__ = [_gs_root + "/utils", os.path.join(os.path.dirname(__file__), "utils")]
sys.modules["utils"] = _gs_utils

import argparse
import math
import cv2
import torch
import numpy as np
import json
from tqdm import tqdm

# Gaussian splatting dependencies
from utils.sh_utils import eval_sh
from scene.gaussian_model import GaussianModel
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from scene.cameras import Camera as GSCamera
from gaussian_renderer import render, GaussianModel
from utils.system_utils import searchForMaxIteration
from utils.graphics_utils import focal2fov

# MPM dependencies
from mpm_solver_warp.engine_utils import *
from mpm_solver_warp.mpm_solver_warp import MPM_Simulator_WARP
import warp as wp
from relation_mpm import RELATION_TYPES, RelationMPM

# Particle filling dependencies
from particle_filling.filling import *

# Utils
from utils.decode_param import *
from utils.transformation_utils import *
from utils.camera_view_utils import *
from utils.render_utils import *

wp.init()
wp.config.verify_cuda = True

ti.init(arch=ti.cuda, device_memory_GB=8.0)


class PipelineParamsNoparse:
    """Same as PipelineParams but without argument parser."""

    def __init__(self):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False


def load_checkpoint(model_path, sh_degree=3, iteration=-1):
    # Find checkpoint
    checkpt_dir = os.path.join(model_path, "point_cloud")
    if iteration == -1:
        iteration = searchForMaxIteration(checkpt_dir)
    checkpt_path = os.path.join(
        checkpt_dir, f"iteration_{iteration}", "point_cloud.ply"
    )

    # Load guassians
    gaussians = GaussianModel(sh_degree)
    gaussians.load_ply(checkpt_path)
    return gaussians


def load_object_relations(path):
    """Load the object-id relation manifest produced by the REST3D bridge."""
    if not path:
        return {}
    payload = json.load(open(path, encoding="utf-8"))
    edges = payload.get("edges", []) if isinstance(payload, dict) else payload
    relations = {}
    for edge in edges:
        source = edge.get("source", edge.get("child_id"))
        target = edge.get("target", edge.get("parent_id"))
        if source is None or target is None:
            continue
        relation = str(edge.get("relation", "other")).lower()
        relations[tuple(sorted((int(source), int(target))))] = RELATION_TYPES.get(relation, RELATION_TYPES["other"])
    return relations


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output_ply", action="store_true")
    parser.add_argument("--output_h5", action="store_true")
    parser.add_argument("--render_img", action="store_true")
    parser.add_argument("--compile_video", action="store_true")
    parser.add_argument("--white_bg", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--enable_relation", action="store_true")
    parser.add_argument("--object_id_path", type=str, default=None)
    parser.add_argument("--property_path", type=str, default=None)
    parser.add_argument("--scene_relation_path", type=str, default=None)
    parser.add_argument("--relation_checkpoint", type=str, default=None)
    parser.add_argument("--relation_full_checkpoint", type=str, default=None)
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        AssertionError("Model path does not exist!")
    if not os.path.exists(args.config):
        AssertionError("Scene config does not exist!")
    if args.output_path is not None and not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    # load scene config
    print("Loading scene config...")
    (
        material_params,
        bc_params,
        time_params,
        preprocessing_params,
        camera_params,
    ) = decode_param_json(args.config)

    # load gaussians
    print("Loading gaussians...")
    model_path = args.model_path
    gaussians = load_checkpoint(model_path)
    pipeline = PipelineParamsNoparse()
    pipeline.compute_cov3D_python = True
    background = (
        torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
        if args.white_bg
        else torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    )

    # init the scene
    print("Initializing scene and pre-processing...")
    params = load_params_from_gs(gaussians, pipeline)

    init_pos = params["pos"]
    init_cov = params["cov3D_precomp"]
    init_screen_points = params["screen_points"]
    init_opacity = params["opacity"]
    init_shs = params["shs"]
    source_particle_indices = torch.arange(init_pos.shape[0], device=init_pos.device)

    # throw away low opacity kernels
    mask = init_opacity[:, 0] > preprocessing_params["opacity_threshold"]
    init_pos = init_pos[mask, :]
    init_cov = init_cov[mask, :]
    init_opacity = init_opacity[mask, :]
    init_screen_points = init_screen_points[mask, :]
    init_shs = init_shs[mask, :]
    source_particle_indices = source_particle_indices[mask]

    # rorate and translate object
    if args.debug:
        if not os.path.exists("./log"):
            os.makedirs("./log")
        particle_position_tensor_to_ply(
            init_pos,
            "./log/init_particles.ply",
        )
    rotation_matrices = generate_rotation_matrices(
        torch.tensor(preprocessing_params["rotation_degree"]),
        preprocessing_params["rotation_axis"],
    )
    rotated_pos = apply_rotations(init_pos, rotation_matrices)

    if args.debug:
        particle_position_tensor_to_ply(rotated_pos, "./log/rotated_particles.ply")

    # select a sim area and save params of unslected particles
    unselected_pos, unselected_cov, unselected_opacity, unselected_shs = (
        None,
        None,
        None,
        None,
    )
    if preprocessing_params["sim_area"] is not None:
        boundary = preprocessing_params["sim_area"]
        assert len(boundary) == 6
        mask = torch.ones(rotated_pos.shape[0], dtype=torch.bool).to(device="cuda")
        for i in range(3):
            mask = torch.logical_and(mask, rotated_pos[:, i] > boundary[2 * i])
            mask = torch.logical_and(mask, rotated_pos[:, i] < boundary[2 * i + 1])

        unselected_pos = init_pos[~mask, :]
        unselected_cov = init_cov[~mask, :]
        unselected_opacity = init_opacity[~mask, :]
        unselected_shs = init_shs[~mask, :]

        rotated_pos = rotated_pos[mask, :]
        init_cov = init_cov[mask, :]
        init_opacity = init_opacity[mask, :]
        init_shs = init_shs[mask, :]
        source_particle_indices = source_particle_indices[mask]

    transformed_pos, scale_origin, original_mean_pos = transform2origin(rotated_pos, preprocessing_params["scale"])
    transformed_pos = shift2center111(transformed_pos)

    # modify covariance matrix accordingly
    init_cov = apply_cov_rotations(init_cov, rotation_matrices)
    init_cov = scale_origin * scale_origin * init_cov

    if args.debug:
        particle_position_tensor_to_ply(
            transformed_pos,
            "./log/transformed_particles.ply",
        )

    # fill particles if needed
    gs_num = transformed_pos.shape[0]
    surface_particle_count = gs_num
    device = "cuda:0"
    filling_params = preprocessing_params["particle_filling"]

    if filling_params is not None:
        print("Filling internal particles...")
        mpm_init_pos = fill_particles(
            pos=transformed_pos,
            opacity=init_opacity,
            cov=init_cov,
            grid_n=filling_params["n_grid"],
            max_samples=filling_params["max_particles_num"],
            grid_dx=material_params["grid_lim"] / filling_params["n_grid"],
            density_thres=filling_params["density_threshold"],
            search_thres=filling_params["search_threshold"],
            max_particles_per_cell=filling_params["max_partciels_per_cell"],
            search_exclude_dir=filling_params["search_exclude_direction"],
            ray_cast_dir=filling_params["ray_cast_direction"],
            boundary=filling_params["boundary"],
            smooth=filling_params["smooth"],
        ).to(device=device)

        if args.debug:
            particle_position_tensor_to_ply(mpm_init_pos, "./log/filled_particles.ply")
    else:
        mpm_init_pos = transformed_pos.to(device=device)

    # init the mpm solver
    print("Initializing MPM solver and setting up boundary conditions...")
    mpm_init_vol = get_particle_volume(
        mpm_init_pos,
        material_params["n_grid"],
        material_params["grid_lim"] / material_params["n_grid"],
        unifrom=material_params["material"] == "sand",
    ).to(device=device)

    if filling_params is not None and filling_params["visualize"] == True:
        shs, opacity, mpm_init_cov = init_filled_particles(
            mpm_init_pos[:gs_num],
            init_shs,
            init_cov,
            init_opacity,
            mpm_init_pos[gs_num:],
        )
        gs_num = mpm_init_pos.shape[0]
    else:
        mpm_init_cov = torch.zeros((mpm_init_pos.shape[0], 6), device=device)
        mpm_init_cov[:gs_num] = init_cov
        shs = init_shs
        opacity = init_opacity

    render_particle_indices = torch.arange(gs_num, device=device)

    if args.debug:
        print("check *.ply files to see if it's ready for simulation")

    # set up the mpm solver
    mpm_solver = MPM_Simulator_WARP(10)
    mpm_solver.load_initial_data_from_torch(
        mpm_init_pos,
        mpm_init_vol,
        mpm_init_cov,
        n_grid=material_params["n_grid"],
        grid_lim=material_params["grid_lim"],
    )
    mpm_solver.set_parameters_dict(material_params)
    particle_object_id = None
    particle_material_id = None
    if args.object_id_path:
        particle_object_id = torch.from_numpy(np.load(args.object_id_path)).to(device=device, dtype=torch.long).reshape(-1)
        if particle_object_id.numel() != surface_particle_count and source_particle_indices.numel() == surface_particle_count:
            if source_particle_indices.numel() and int(source_particle_indices.max()) >= particle_object_id.numel():
                raise ValueError("object_id labels do not cover the source Gaussian indices")
            particle_object_id = particle_object_id[source_particle_indices]
        if particle_object_id.numel() == surface_particle_count and mpm_init_pos.shape[0] > surface_particle_count:
            filled_labels = []
            surface_pos = mpm_init_pos[:surface_particle_count]
            for start in range(surface_particle_count, mpm_init_pos.shape[0], 8192):
                filled = mpm_init_pos[start:start + 8192]
                nearest = torch.cdist(filled, surface_pos).argmin(1)
                filled_labels.append(particle_object_id[nearest])
            particle_object_id = torch.cat((particle_object_id, *filled_labels))
        if particle_object_id.numel() != mpm_init_pos.shape[0]:
            raise ValueError(f"object_id count mismatch: {particle_object_id.numel()} labels for {mpm_init_pos.shape[0]} particles")
    if args.property_path:
        if particle_object_id is None:
            raise ValueError("--property_path requires --object_id_path")
        records = {int(r["object_id"]): r for r in json.load(open(args.property_path))}
        E = torch.full((mpm_init_pos.shape[0],), float(material_params["E"]), device=device)
        nu = torch.full_like(E, float(material_params["nu"]))
        density = torch.full_like(E, float(material_params["density"]))
        material_names = sorted({str(record.get("material", "unknown")) for record in records.values()})
        material_lookup = {name: idx for idx, name in enumerate(material_names)}
        particle_material_id = torch.zeros_like(particle_object_id)
        for oid in torch.unique(particle_object_id).tolist():
            r = records[int(oid)]; mask = particle_object_id == oid
            E[mask] = min(max(float(r["E"]), 1e4), 1e7)
            nu[mask] = min(max(float(r["nu"]), .05), .45)
            density[mask] = min(max(float(r["density"]), 50.), 3000.)
            particle_material_id[mask] = material_lookup[str(r.get("material", "unknown"))]
        mpm_solver.reset_material_fields(E, nu, density, device=device)
    elif particle_object_id is not None:
        particle_material_id = torch.zeros_like(particle_object_id)

    # Note: boundary conditions may depend on mass, so the order cannot be changed!
    set_boundary_conditions(mpm_solver, bc_params, time_params)

    mpm_solver.finalize_mu_lam()

    if args.enable_relation:
        object_relations = load_object_relations(args.scene_relation_path)
        # Ficus is a single-object checkpoint; real multi-object runs must
        # provide persistent per-particle object/material labels here.
        relation = RelationMPM(
            n_materials=max(16, int(particle_material_id.max().item()) + 1 if particle_material_id is not None else 1),
            local_radius=1.75 * material_params["grid_lim"] / material_params["n_grid"],
            contact_radius=1.75 * material_params["grid_lim"] / material_params["n_grid"],
        ).to(device)
        if args.relation_full_checkpoint:
            full = torch.load(args.relation_full_checkpoint, map_location=device)["model"]
            current = relation.state_dict()
            usable = {k:v for k,v in full.items() if k in current and current[k].shape == v.shape}
            relation.load_state_dict(usable, strict=False)
            print("loaded full relation Transformer", args.relation_full_checkpoint, "tensors", len(usable))
        if args.relation_checkpoint:
            raise ValueError("Legacy acceleration LoRA checkpoints are incompatible with cached relation parameters; retrain the relation model")
        if os.getenv("RELATION_ANALYTIC_PRIOR") == "1":
            with torch.no_grad():
                relation.force_output.bias[:] = torch.tensor([0.01, 0.01, 0.005, 0.0], device=device)
        object_id = particle_object_id if particle_object_id is not None else torch.zeros(mpm_init_pos.shape[0], dtype=torch.long, device=device)
        material_id = particle_material_id if particle_material_id is not None else torch.zeros_like(object_id)
        mpm_solver.relation_step = lambda solver, dt: relation.step_solver(
            solver, object_id, material_id, dt, object_relations
        )

    # camera setting
    mpm_space_viewpoint_center = (
        torch.tensor(camera_params["mpm_space_viewpoint_center"]).reshape((1, 3)).cuda()
    )
    mpm_space_vertical_upward_axis = (
        torch.tensor(camera_params["mpm_space_vertical_upward_axis"])
        .reshape((1, 3))
        .cuda()
    )
    (
        viewpoint_center_worldspace,
        observant_coordinates,
    ) = get_center_view_worldspace_and_observant_coordinate(
        mpm_space_viewpoint_center,
        mpm_space_vertical_upward_axis,
        rotation_matrices,
        scale_origin,
        original_mean_pos,
    )

    # run the simulation
    if args.output_ply or args.output_h5:
        directory_to_save = os.path.join(args.output_path, "simulation_ply")
        if not os.path.exists(directory_to_save):
            os.makedirs(directory_to_save)

        save_data_at_frame(
            mpm_solver,
            directory_to_save,
            0,
            save_to_ply=args.output_ply,
            save_to_h5=args.output_h5,
        )

    substep_dt = time_params["substep_dt"]
    frame_dt = time_params["frame_dt"]
    frame_num = time_params["frame_num"]
    step_per_frame = int(frame_dt / substep_dt)
    opacity_render = opacity
    shs_render = shs
    height = None
    width = None
    for frame in tqdm(range(frame_num)):
        current_camera = get_camera_view(
            model_path,
            default_camera_index=camera_params["default_camera_index"],
            center_view_world_space=viewpoint_center_worldspace,
            observant_coordinates=observant_coordinates,
            show_hint=camera_params["show_hint"],
            init_azimuthm=camera_params["init_azimuthm"],
            init_elevation=camera_params["init_elevation"],
            init_radius=camera_params["init_radius"],
            move_camera=camera_params["move_camera"],
            current_frame=frame,
            delta_a=camera_params["delta_a"],
            delta_e=camera_params["delta_e"],
            delta_r=camera_params["delta_r"],
        )
        rasterize = initialize_resterize(
            current_camera, gaussians, pipeline, background
        )

        for step in range(step_per_frame):
            mpm_solver.p2g2p(frame, substep_dt, device=device)

        if args.output_ply or args.output_h5:
            save_data_at_frame(
                mpm_solver,
                directory_to_save,
                frame + 1,
                save_to_ply=args.output_ply,
                save_to_h5=args.output_h5,
            )

        if args.render_img:
            pos = mpm_solver.export_particle_x_to_torch().index_select(0, render_particle_indices).to(device)
            cov3D = mpm_solver.export_particle_cov_to_torch()
            rot = mpm_solver.export_particle_R_to_torch()
            cov3D = cov3D.view(-1, 6).index_select(0, render_particle_indices).to(device)
            rot = rot.view(-1, 3, 3).index_select(0, render_particle_indices).to(device)

            pos = apply_inverse_rotations(
                undotransform2origin(
                    undoshift2center111(pos), scale_origin, original_mean_pos
                ),
                rotation_matrices,
            )
            cov3D = cov3D / (scale_origin * scale_origin)
            cov3D = apply_inverse_cov_rotations(cov3D, rotation_matrices)
            opacity = opacity_render
            shs = shs_render
            if preprocessing_params["sim_area"] is not None:
                pos = torch.cat([pos, unselected_pos], dim=0)
                cov3D = torch.cat([cov3D, unselected_cov], dim=0)
                opacity = torch.cat([opacity_render, unselected_opacity], dim=0)
                shs = torch.cat([shs_render, unselected_shs], dim=0)

            colors_precomp = convert_SH(shs, current_camera, gaussians, pos, rot)
            rendering, raddi = rasterize(
                means3D=pos,
                means2D=init_screen_points,
                shs=None,
                colors_precomp=colors_precomp,
                opacities=opacity,
                scales=None,
                rotations=None,
                cov3D_precomp=cov3D,
            )
            cv2_img = rendering.permute(1, 2, 0).detach().cpu().numpy()
            cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
            if height is None or width is None:
                height = cv2_img.shape[0] // 2 * 2
                width = cv2_img.shape[1] // 2 * 2
            assert args.output_path is not None
            cv2.imwrite(
                os.path.join(args.output_path, f"{frame}.png".rjust(8, "0")),
                255 * cv2_img,
            )

    if args.render_img and args.compile_video:
        fps = int(1.0 / time_params["frame_dt"])
        os.system(
            f"ffmpeg -framerate {fps} -i {args.output_path}/%04d.png -c:v libx264 -s {width}x{height} -y -pix_fmt yuv420p {args.output_path}/output.mp4"
        )

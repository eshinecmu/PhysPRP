import sys
import os
import types
from pathlib import Path
import subprocess

_gs_root = str(Path(os.environ.get("PHYS_GS_ROOT", Path(__file__).resolve().parents[1] / "gaussian-splatting")).resolve())
if not (Path(_gs_root) / "scene" / "gaussian_model.py").is_file():
    raise FileNotFoundError(f"Gaussian splatting source missing: {_gs_root}; set PHYS_GS_ROOT")
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
from relation_mpm import RELATION_TYPES, PostStepRelationMPM, load_relation_checkpoint
from gaussian_property_adapter import MATERIAL_VOCAB, particle_material_fields

# Particle filling dependencies
from particle_filling.filling import *

# Utils
from utils.decode_param import *
from utils.transformation_utils import *
from utils.camera_view_utils import *
from utils.render_utils import *

wp.config.kernel_cache_dir = os.environ.get("WARP_CACHE_PATH", str(Path(__file__).resolve().parents[1] / ".cache" / "warp"))
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
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output_ply", action="store_true")
    parser.add_argument("--output_h5", action="store_true")
    parser.add_argument("--render_img", action="store_true")
    parser.add_argument("--compile_video", action="store_true")
    parser.add_argument("--white_bg", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--enable_relation", action="store_true")
    parser.add_argument("--object_manifest", type=str)
    parser.add_argument("--object_id_path", type=str, default=None)
    parser.add_argument("--property_path", type=str, default=None)
    parser.add_argument("--scene_relation_path", type=str, default=None)
    parser.add_argument("--relation_checkpoint", type=str, default=None)
    parser.add_argument("--relation_full_checkpoint", type=str, default=None)
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        raise FileNotFoundError("Model path does not exist!")
    if not os.path.exists(args.config):
        raise FileNotFoundError("Scene config does not exist!")
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
    scene_config = json.loads(Path(args.config).read_text(encoding="utf-8"))

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
    source_ids = None
    object_records = None
    if args.object_manifest:
        if not args.object_id_path:
            raise ValueError("Object manifest requires Gaussian object labels")
        object_records = json.loads(Path(args.object_manifest).read_text())
        source_ids = torch.as_tensor(np.load(args.object_id_path), device=init_pos.device, dtype=torch.long)
        if source_ids.shape != (len(init_pos),):
            raise ValueError("Source Gaussian object labels have wrong length")
        if set(source_ids.tolist()) != {r["object_id"] for r in object_records}:
            raise ValueError("Manifest and Gaussian identities differ")

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
    has_background = False
    if preprocessing_params["sim_area"] is not None or object_records is not None:
        boundary = preprocessing_params["sim_area"]
        if boundary is not None and len(boundary) != 6:
            raise ValueError("sim_area must have six bounds")
        mask = torch.ones(rotated_pos.shape[0], dtype=torch.bool).to(device="cuda")
        if boundary is not None:
            for i in range(3):
                mask &= (rotated_pos[:, i] > boundary[2*i]) & (rotated_pos[:, i] < boundary[2*i+1])
        if object_records is not None:
            dynamic = torch.zeros_like(mask)
            for record in object_records:
                if record["role"] == "dynamic":
                    dynamic |= source_ids[source_particle_indices] == record["object_id"]
                elif record["role"] in ("floor", "static"):
                    settings = scene_config["objects"][record["name"]]
                    if "collider" not in settings:
                        raise ValueError(f"Explicit collider required for {record['name']}")
            mask &= dynamic
        if not mask.any():
            raise ValueError("No dynamic Gaussians remain after role/area filtering")
        has_background = bool((~mask).any())

        unselected_pos = init_pos[~mask, :]
        unselected_cov = init_cov[~mask, :]
        unselected_opacity = init_opacity[~mask, :]
        unselected_shs = init_shs[~mask, :]

        rotated_pos = rotated_pos[mask, :]
        init_cov = init_cov[mask, :]
        init_opacity = init_opacity[mask, :]
        init_shs = init_shs[mask, :]
        source_particle_indices = source_particle_indices[mask]

    if object_records is not None:
        if preprocessing_params["rotation_degree"] or "mpm_origin" not in scene_config:
            raise ValueError("Native scene requires explicit mpm_origin and no implicit preprocessing rotation")
        offset = torch.as_tensor(scene_config["mpm_origin"], device=rotated_pos.device, dtype=rotated_pos.dtype)
        if offset.shape != (3,) or not torch.isfinite(offset).all():
            raise ValueError("mpm_origin must be a finite translation in meters")
        scale_origin = torch.tensor(1., device=rotated_pos.device)
        original_mean_pos = 1. - offset
        transformed_pos = rotated_pos + offset
    else:
        transformed_pos, scale_origin, original_mean_pos = transform2origin(rotated_pos, preprocessing_params["scale"])
        transformed_pos = shift2center111(transformed_pos)
    margin = material_params["grid_lim"] / material_params["n_grid"] * 3
    if not torch.isfinite(transformed_pos).all() or (transformed_pos < margin).any() or (transformed_pos > material_params["grid_lim"] - margin).any():
        raise ValueError("Dynamic scene outside MPM grid margin; adjust explicit mpm_origin/grid_lim")

    # modify covariance matrix accordingly
    init_cov = apply_cov_rotations(init_cov, rotation_matrices)
    init_cov = scale_origin * scale_origin * init_cov

    if args.debug:
        particle_position_tensor_to_ply(
            transformed_pos,
            "./log/transformed_particles.ply",
        )

    if object_records is not None:
        expected_dynamic = {r["object_id"] for r in object_records if r["role"] == "dynamic"}
        retained_dynamic = set(source_ids[source_particle_indices].tolist())
        if expected_dynamic != retained_dynamic:
            raise ValueError(f"Dynamic objects removed by filtering: {expected_dynamic-retained_dynamic}")
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
    mpm_solver.enable_contact_diagnostics(True)
    particle_object_id = None
    particle_material_id = None
    if args.object_id_path:
        particle_object_id = torch.from_numpy(np.load(args.object_id_path)).to(device=device, dtype=torch.long).reshape(-1)
        if source_particle_indices.numel() and int(source_particle_indices.max()) >= particle_object_id.numel():
            raise ValueError("object_id labels do not cover source Gaussian indices")
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
        fields, material_audit = particle_material_fields(
            records, particle_object_id
        )

        particle_material_id = fields["material_id"]

        mpm_solver.reset_material_fields(
            fields["E"], fields["nu"], fields["density"], device=device
        )

        for audit in material_audit:
            audit["constitutive_model"] = material_params["material"]
        Path(args.output_path, "material_actual.json").write_text(json.dumps(material_audit, indent=2))
    elif particle_object_id is not None:
        particle_material_id = torch.zeros_like(particle_object_id)

    if object_records is not None:
        velocity = torch.zeros_like(mpm_init_pos)
        for record in object_records:
            settings = scene_config.get("objects", {}).get(record["name"], {})
            if record["role"] == "dynamic":
                value = torch.as_tensor(settings.get("velocity", [0., 0., 0.]), device=device, dtype=velocity.dtype)
                if value.shape != (3,) or not torch.isfinite(value).all():
                    raise ValueError("Initial velocity must be finite meters/second")
                velocity[particle_object_id == record["object_id"]] = value
            elif record["role"] in ("floor", "static"):
                collider = dict(settings["collider"])
                if collider.get("type") != "surface_collider":
                    raise ValueError("Native static roles currently require an explicit plane collider")
                collider["point"] = (np.asarray(collider["point"]) + np.asarray(scene_config["mpm_origin"])).tolist()
                bc_params.append(collider)
        mpm_solver.import_particle_v_from_torch(velocity.contiguous(), device=device)
    # Note: boundary conditions may depend on mass, so the order cannot be changed!
    set_boundary_conditions(mpm_solver, bc_params, time_params)

    mpm_solver.finalize_mu_lam()

    if args.enable_relation:
        object_relations = load_object_relations(args.scene_relation_path)
        # Ficus is a single-object checkpoint; real multi-object runs must
        # provide persistent per-particle object/material labels here.
        if not args.relation_full_checkpoint or args.relation_checkpoint:
            raise ValueError("Verified relation inference requires a compatible full checkpoint")
        relation_config = scene_config.get("relation", {})
        checkpoint_settings = torch.load(args.relation_full_checkpoint, map_location="cpu", weights_only=True).get("metadata", {})
        relation = PostStepRelationMPM(
            n_materials=len(MATERIAL_VOCAB),
            hidden_dim=relation_config.get("hidden_dim", checkpoint_settings.get("hidden_dim", 32)),
            local_radius=relation_config.get("local_radius", checkpoint_settings.get("local_radius", 1.75 * material_params["grid_lim"] / material_params["n_grid"])),
            contact_radius=relation_config.get("contact_radius", checkpoint_settings.get("contact_radius", 1.75 * material_params["grid_lim"] / material_params["n_grid"])),
            max_tokens=relation_config.get("max_tokens", checkpoint_settings.get("max_tokens", 4096)),
            event_check_interval=relation_config.get("event_check_interval", 1),
            max_force=relation_config.get("max_force", 1.),
            max_delta_v=relation_config.get("max_delta_v", checkpoint_settings.get("max_delta_v")),
            use_state_features=relation_config.get("use_state_features", checkpoint_settings.get("use_state_features", False)),
            residual_scale=relation_config.get("residual_scale", checkpoint_settings.get("residual_scale", 1.)),
            contact_hysteresis=relation_config.get("contact_hysteresis", .2),
            max_refresh_steps=relation_config.get("max_refresh_steps", 100),
            diagnostics_path=str(Path(args.output_path) / "relation_events.jsonl"),
        ).to(device)
        metadata = load_relation_checkpoint(relation, args.relation_full_checkpoint, MATERIAL_VOCAB)
        relation.eval()
        Path(args.output_path, "relation_model.json").write_text(json.dumps(metadata, indent=2))
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
    if not math.isfinite(frame_dt) or not math.isfinite(substep_dt) or frame_dt <= 0 or substep_dt <= 0 or frame_num <= 0:
        raise ValueError("Positive finite timesteps and frame_num required")
    step_per_frame = math.ceil(frame_dt / substep_dt)
    substep_dt = frame_dt / step_per_frame
    opacity_render = opacity
    shs_render = shs
    height = None
    width = None
    state_checks = []
    frame_times = {"initial_render": "../render/first_frame.png", "simulation_png_start_time": frame_dt, "simulation_png_interval": frame_dt}
    Path(args.output_path, "frame_times.json").write_text(json.dumps(frame_times, indent=2))
    mass = wp.to_torch(mpm_solver.mpm_state.particle_mass)
    if not torch.isfinite(mpm_init_vol).all() or not (mpm_init_vol > 0).all() or not torch.isfinite(mass).all() or not (mass > 0).all():
        raise ValueError("Particle mass and volume must be finite and positive")
    np.save(Path(args.output_path) / "particle_object_id.npy", particle_object_id.cpu().numpy() if particle_object_id is not None else np.zeros(len(mpm_init_pos), dtype=np.int32))
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

        x_check = mpm_solver.export_particle_x_to_torch()
        v_check = mpm_solver.export_particle_v_to_torch()
        f_check = wp.to_torch(mpm_solver.mpm_state.particle_F_trial).reshape(-1,3,3)
        det = torch.linalg.det(f_check)
        cfl = float(v_check.abs().max()) * substep_dt / float(mpm_solver.mpm_model.dx)
        if not all(torch.isfinite(t).all() for t in (x_check,v_check,f_check)) or (det <= 0).any() or cfl > 1:
            raise ValueError(f"Invalid/CFL-unsafe MPM state at frame {frame+1}")
        contacts = mpm_solver.contact_diagnostics_snapshot()
        with Path(args.output_path, "surface_contacts.jsonl").open("a") as contact_log:
            contact_log.write(json.dumps(dict(frame=frame+1, **contacts)) + "\n")
        state_checks.append(dict(frame=frame+1,time=(frame+1)*frame_dt,min_det=float(det.min()),frame_end_advective_cfl=cfl))
        Path(args.output_path, "validation.json").write_text(json.dumps(dict(status="running", relation_enabled=args.enable_relation, dt=substep_dt, steps_per_frame=step_per_frame, frames=state_checks),indent=2))
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
            if has_background:
                rot = torch.cat([rot, torch.eye(3, device=device).expand(len(unselected_pos), -1, -1)], dim=0)
                pos = torch.cat([pos, unselected_pos], dim=0)
                cov3D = torch.cat([cov3D, unselected_cov], dim=0)
                opacity = torch.cat([opacity_render, unselected_opacity], dim=0)
                shs = torch.cat([shs_render, unselected_shs], dim=0)

            colors_precomp = convert_SH(shs, current_camera, gaussians, pos, rot)
            rendering, raddi = rasterize(
                means3D=pos,
                means2D=torch.zeros_like(pos, requires_grad=True),
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

    Path(args.output_path, "validation.json").write_text(json.dumps(dict(status="forward_complete", relation_enabled=args.enable_relation, learned_relation_verified=False, dt=substep_dt, steps_per_frame=step_per_frame, frames=state_checks, contact_events="see_solver_contact_log_if_enabled"),indent=2))
    if args.render_img and args.compile_video:
        subprocess.run(["ffmpeg", "-framerate", str(1.0 / frame_dt), "-i", str(Path(args.output_path) / "%04d.png"), "-c:v", "libx264", "-s", f"{width}x{height}", "-y", "-pix_fmt", "yuv420p", str(Path(args.output_path) / "output.mp4")], check=True)

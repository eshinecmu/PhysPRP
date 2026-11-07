"""Train official PhysCoRe MfM/RfD stages or a known-material numerical rollout.

The explicit --pipeline post-step option preserves the older particle-residual experiment:

Trajectory: x/v [T,N,3], F_trial/C [T,N,3,3], time [T], particle_id [N]
or [T,N] in persistent order, selection/volume/mass/E/nu/density/object_id/
material_id [N], material_vocab, mpm_config. Config requires material='jelly',
n_grid, grid_lim, g, rpic_damping, grid_v_damping_scale, substep_dt, and
boundary_conditions (static surfaces/bounding box). Precomputed --gt requires
x_base/v_base/target [N,3], static fields except volume, dt, and explicit metadata.
"""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import torch

from PhysGaussian.gaussian_property_adapter import MATERIAL_VOCAB
from PhysGaussian.relation_mpm import PostStepRelationMPM, RELATION_TYPES, load_relation_checkpoint


SAMPLE_METADATA = {
    "input_schema": "post-mpm-velocity-residual-v1",
    "output_schema": "delta-v-after-mpm-v1",
    "position_update": "x_base+dt*delta_v",
    "input_state": "mpm_baseline_after_one_substep",
    "target_definition": "v_true_next-v_base",
    "baseline_reset": "true_current_state",
}
STATIC_FIELDS = ("selection", "mass", "E", "nu", "density", "object_id", "material_id")
MPM_FIELDS = ("material", "n_grid", "grid_lim", "g", "rpic_damping",
              "grid_v_damping_scale", "substep_dt", "boundary_conditions")
MPM_OPTIONAL_FIELDS = {"transfer_scheme", "grid_mass_epsilon"}


def _tensor(data, key, shape, *, integer=False):
    value = data.get(key)
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(shape):
        raise ValueError(f"{key} must be a tensor of shape {tuple(shape)}")
    if integer:
        if value.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"{key} must be int32 or int64")
    elif value.dtype != torch.float32:
        raise ValueError(f"{key} must be float32")
    if not torch.isfinite(value).all():
        raise ValueError(f"{key} contains non-finite values")
    return value


def _static_fields(data, n):
    for key in STATIC_FIELDS:
        _tensor(data, key, (n,), integer=key in ("selection", "object_id", "material_id"))
    if data.get("material_vocab") != list(MATERIAL_VOCAB):
        raise ValueError("material_vocab must exactly match the ordered inference MATERIAL_VOCAB")
    if (data["object_id"] < 0).any() or (data["material_id"] < 0).any() or (data["material_id"] >= len(MATERIAL_VOCAB)).any():
        raise ValueError("Object IDs must be nonnegative; material IDs must index MATERIAL_VOCAB")
    if not ((data["selection"] == 0) | (data["selection"] == 1)).all():
        raise ValueError("selection must be 0 (simulated) or 1 (static)")
    if not (data["selection"] == 0).any():
        raise ValueError("At least one simulated particle is required")
    if any((data[key] <= 0).any() for key in ("mass", "E", "density")):
        raise ValueError("mass, E and density must be positive")
    if ((data["nu"] < 0) | (data["nu"] >= .5)).any():
        raise ValueError("Require 0 <= nu < 0.5")
    relations = data.get("object_relations", {})
    if not isinstance(relations, dict):
        raise ValueError("object_relations must map sorted object-ID pairs to relation type IDs")
    objects = set(data["object_id"].tolist())
    for pair, kind in relations.items():
        if not isinstance(pair, tuple) or len(pair) != 2 or any(type(oid) is not int for oid in pair) or pair[0] >= pair[1] or not set(pair) <= objects or type(kind) is not int or kind not in RELATION_TYPES.values():
            raise ValueError("object_relations requires distinct sorted known integer object pairs and valid relation IDs")


def load_gt(path, device):
    """Reject legacy force labels and samples whose baseline semantics are absent."""
    data = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(data, dict) or not isinstance(data.get("metadata"), dict):
        raise ValueError("GT sample requires explicit post-MPM metadata")
    for key, expected in SAMPLE_METADATA.items():
        if data["metadata"].get(key) != expected:
            raise ValueError(f"GT metadata {key!r} must be {expected!r}")
    x = data.get("x_base")
    if not isinstance(x, torch.Tensor) or x.ndim != 2 or x.shape[1] != 3 or not len(x):
        raise ValueError("GT x_base must have nonempty shape [N,3]")
    n = len(x)
    for key in ("x_base", "v_base", "target"):
        _tensor(data, key, (n, 3))
    if "F_trial" in data or "C" in data:
        for key in ("F_trial", "C"):
            _tensor(data, key, (n, 3, 3))
    _static_fields(data, n)
    dt = float(data.get("dt", float("nan")))
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("GT dt must be finite and positive")
    if torch.count_nonzero(data["target"][data["selection"] != 0]):
        raise ValueError("Static particles cannot have a velocity correction target")
    return {**data, "dt": dt, "object_id": data["object_id"].long(), "material_id": data["material_id"].long()}


def _validate_trajectory(data):
    if not isinstance(data, dict):
        raise ValueError("Trajectory must be a dictionary")
    x = data.get("x")
    if not isinstance(x, torch.Tensor) or x.ndim != 3 or x.shape[2] != 3 or x.shape[0] < 2 or not x.shape[1]:
        raise ValueError("Trajectory x must have shape [T>=2,N>0,3]")
    frames, n, _ = x.shape
    for key in ("x", "v"):
        _tensor(data, key, (frames, n, 3))
    for key in ("F_trial", "C"):
        _tensor(data, key, (frames, n, 3, 3))
    if (torch.linalg.det(data["F_trial"]) <= 0).any():
        raise ValueError("F_trial must have positive determinant")
    time = data.get("time")
    if not isinstance(time, torch.Tensor) or time.shape != (frames,) or time.dtype not in (torch.float32, torch.float64) or not torch.isfinite(time).all():
        raise ValueError("time must be a finite floating tensor [T]")
    if (time < 0).any() or (time[1:] <= time[:-1]).any():
        raise ValueError("time must be nonnegative and strictly increasing")
    _static_fields(data, n)
    volume = _tensor(data, "volume", (n,))
    if (volume <= 0).any() or not torch.allclose(data["mass"], data["density"] * volume, rtol=1e-5, atol=1e-10):
        raise ValueError("volume must be positive and mass must equal density * volume")
    ids = data.get("particle_id")
    if not isinstance(ids, torch.Tensor) or ids.dtype not in (torch.int32, torch.int64) or ids.shape not in ((n,), (frames, n)):
        raise ValueError("particle_id must be integer [N] or [T,N]")
    ordered = ids if ids.ndim == 1 else ids[0]
    if (ordered < 0).any() or torch.unique(ordered).numel() != n:
        raise ValueError("particle_id must contain distinct nonnegative IDs")
    if ids.ndim == 2 and not torch.equal(ids, ordered.expand_as(ids)):
        raise ValueError("Particle order changes between frames; explicitly align all tensors by ID first")
    config = data.get("mpm_config")
    if not isinstance(config, dict) or any(key not in config for key in MPM_FIELDS):
        raise ValueError(f"mpm_config requires {', '.join(MPM_FIELDS)}")
    if set(config) - set(MPM_FIELDS) - MPM_OPTIONAL_FIELDS:
        raise ValueError("Unknown mpm_config fields would be silently ignored; supply only documented fields")
    if config.get("transfer_scheme", "gradient") not in ("gradient", "mls"):
        raise ValueError("transfer_scheme must be gradient or mls")
    epsilon = float(config.get("grid_mass_epsilon", 0.))
    if not math.isfinite(epsilon) or epsilon < 0:
        raise ValueError("grid_mass_epsilon must be finite and nonnegative")
    if config["material"] != "jelly":
        raise ValueError("Only elastic jelly solids are supported; plastic history cannot be restored")
    if type(config["n_grid"]) is not int or config["n_grid"] < 6:
        raise ValueError("n_grid must be an integer >= 6")
    for key in ("grid_lim", "substep_dt"):
        if not math.isfinite(float(config[key])) or float(config[key]) <= 0:
            raise ValueError(f"mpm_config {key} must be finite and positive")
    for key in ("rpic_damping", "grid_v_damping_scale"):
        if not math.isfinite(float(config[key])):
            raise ValueError(f"mpm_config {key} must be finite")
    if not -1 <= float(config["rpic_damping"]) <= 1 or not 0 <= float(config["grid_v_damping_scale"]) <= 1.1:
        raise ValueError("Unsupported RPIC/grid damping range")
    gravity = torch.as_tensor(config["g"])
    if gravity.shape != (3,) or not torch.isfinite(gravity).all():
        raise ValueError("g must contain three finite acceleration components")
    intervals = time[1:].double() - time[:-1].double()
    dt = float(config["substep_dt"])
    if not torch.allclose(intervals, torch.full_like(intervals, dt), rtol=1e-4, atol=dt * 1e-4):
        raise ValueError("Every saved interval must equal one declared substep_dt; sparse frames need substep states")
    dx = float(config["grid_lim"]) / config["n_grid"]
    active_x = x[:, data["selection"] == 0]
    if (active_x < 2 * dx).any() or (active_x > float(config["grid_lim"]) - 2 * dx).any():
        raise ValueError("Active particles must stay at least two grid cells inside the MPM domain")
    if not isinstance(config["boundary_conditions"], list):
        raise ValueError("boundary_conditions must be an explicit list, including [] for no boundaries")
    for bc in config["boundary_conditions"]:
        if not isinstance(bc, dict) or bc.get("type") not in ("surface_collider", "bounding_box"):
            raise ValueError("Only static surface_collider/bounding_box boundaries can be restored without BC history")
        if bc["type"] == "bounding_box":
            if set(bc) != {"type"}:
                raise ValueError("bounding_box accepts only type in the existing boundary adapter")
            continue
        expected = {"type", "point", "normal", "surface", "friction", "start_time", "end_time"}
        if not expected <= set(bc) or not set(bc) <= expected | {"radius"}:
            raise ValueError("surface_collider needs point, normal, surface, friction, start_time and end_time")
        for key in ("point", "normal"):
            vector = torch.as_tensor(bc[key])
            if vector.shape != (3,) or not torch.isfinite(vector).all():
                raise ValueError(f"Collider {key} must be finite [3]")
        if torch.as_tensor(bc["normal"]).norm() == 0 or bc["surface"] not in ("sticky", "slip", "separate"):
            raise ValueError("Collider requires a nonzero normal and sticky/slip/separate surface")
        if any(not math.isfinite(float(bc[k])) for k in ("friction", "start_time", "end_time")) or float(bc["friction"]) < 0 or float(bc["start_time"]) >= float(bc["end_time"]):
            raise ValueError("Collider friction/time range is invalid")
        if not math.isfinite(float(bc.get("radius", 0))) or float(bc.get("radius", 0)) < 0:
            raise ValueError("Collider radius must be finite and nonnegative")
    static = data["selection"] != 0
    if not torch.equal(x[:, static], x[0, static].expand_as(x[:, static])) or not torch.equal(data["v"][:, static], data["v"][0, static].expand_as(data["v"][:, static])):
        raise ValueError("Static selected particles must preserve positions and velocities")
    return frames


def make_baseline_solver(trajectory, frame_index, device):
    """Restore one complete elastic GT state into the inference solver."""
    # Precede the incomplete, same-named solver at repository root.
    physics_root = str(Path(__file__).resolve().parent / "PhysGaussian")
    if not sys.path or sys.path[0] != physics_root:
        sys.path.insert(0, physics_root)
    from mpm_solver_warp.mpm_solver_warp import MPM_Simulator_WARP
    from PhysGaussian.utils.decode_param import set_boundary_conditions
    import warp as wp

    if wp.config.kernel_cache_dir is None:
        wp.config.kernel_cache_dir = os.environ.get("PHYSPRP_WARP_CACHE_DIR", str(Path(__file__).resolve().parent / ".cache" / "warp"))
    wp.init()
    config = trajectory["mpm_config"]
    device = str(torch.device(device))
    solver = MPM_Simulator_WARP(len(trajectory["mass"]), n_grid=config["n_grid"], grid_lim=config["grid_lim"], device=device)
    solver.load_initial_data_from_torch(trajectory["x"][frame_index].to(device), trajectory["volume"].to(device),
                                        n_grid=config["n_grid"], grid_lim=config["grid_lim"], device=device)
    solver.set_parameters_dict({key: config[key] for key in
                               ("material", "g", "rpic_damping", "grid_v_damping_scale", "transfer_scheme", "grid_mass_epsilon")
                               if key in config}, device=device)
    solver.reset_material_fields(*(trajectory[key].to(device) for key in ("E", "nu", "density")), device=device)
    solver.import_particle_v_from_torch(trajectory["v"][frame_index].to(device), device=device)
    solver.import_particle_C_from_torch(trajectory["C"][frame_index].to(device), device=device)
    # import_particle_F_from_torch writes F; compute_stress overwrites it from F_trial.
    solver.mpm_state.particle_F_trial = wp.from_torch(trajectory["F_trial"][frame_index].to(device).contiguous().clone(), dtype=wp.mat33)
    solver.mpm_state.particle_selection = wp.from_torch(trajectory["selection"].to(device=device, dtype=torch.int32).contiguous().clone(), dtype=wp.int32)
    solver.time = float(trajectory["time"][frame_index])
    set_boundary_conditions(solver, config["boundary_conditions"], {"substep_dt": config["substep_dt"]})
    solver.relation_step = None
    if hasattr(solver, "grid_correction_step"):
        solver.grid_correction_step = None
    return solver


def build_trajectory_samples(trajectory, device, frame_indices=None, solver_factory=make_baseline_solver):
    """Reset each baseline from GT S_t; never subtract independent rollouts.

    ponytail: retain transitions for small datasets; stream independent samples
    when keeping the whole trajectory's baseline tensors exceeds device memory.
    """
    frames = _validate_trajectory(trajectory)
    indices = list(range(frames - 1)) if frame_indices is None else list(frame_indices)
    if not indices or any(type(index) is not int or not 0 <= index < frames - 1 for index in indices):
        raise ValueError("frame_indices must index valid current frames with a successor")
    samples = []
    with torch.no_grad():
        for index in indices:
            solver = solver_factory(trajectory, index, device)
            dt = float(trajectory["mpm_config"]["substep_dt"])
            solver.p2g2p(index, dt, device=str(torch.device(device)))
            x_base = solver.export_particle_x_to_torch().detach().clone().to(device)
            v_base = solver.export_particle_v_to_torch().detach().clone().to(device)
            import warp as wp
            F_trial = wp.to_torch(solver.mpm_state.particle_F_trial).detach().clone().to(device)
            C = wp.to_torch(solver.mpm_state.particle_C).detach().clone().to(device)
            if not torch.isfinite(x_base).all() or not torch.isfinite(v_base).all():
                raise ValueError(f"Non-finite baseline at GT frame {index}")
            sample = {key: trajectory[key].to(device) for key in STATIC_FIELDS}
            for key in ("object_id", "material_id"):
                sample[key] = sample[key].long()
            target = trajectory["v"][index + 1].to(device) - v_base
            if torch.count_nonzero(target[sample["selection"] != 0]):
                raise ValueError("Baseline/GT static velocities differ")
            samples.append({**sample, "x_base": x_base, "v_base": v_base, "target": target,
                            "F_trial": F_trial, "C": C,
                            "dt": dt, "frame_index": index, "source_time": float(trajectory["time"][index]),
                            "metadata": dict(SAMPLE_METADATA), "material_vocab": list(MATERIAL_VOCAB),
                            "object_relations": trajectory.get("object_relations", {})})
    return samples


def predict_sample(model, sample):
    return model(sample["x_base"], sample["v_base"], sample["object_id"], sample["material_id"],
                 sample["mass"], sample["dt"], sample["E"], sample["nu"], sample["density"],
                 object_relations=sample.get("object_relations"), selection=sample["selection"],
                 **({"F_trial": sample["F_trial"], "C": sample["C"]} if model.use_state_features else {}))


def load_trajectory(args):
    trajectory = torch.load(args.trajectory, map_location="cpu", weights_only=True)
    trajectory["mpm_config"] = dict(trajectory["mpm_config"])
    for key in ("transfer_scheme", "grid_mass_epsilon"):
        if getattr(args, key) is not None:
            trajectory["mpm_config"][key] = getattr(args, key)
    return trajectory


def rollout_model(trajectory, model, initial, horizons, device):
    """Restore GT once, then measure a real closed-loop rollout including strain."""
    import warp as wp
    solver = make_baseline_solver(trajectory, initial, device)
    if model is not None:
        model.eval()
        solver.relation_step = lambda actual, dt: model.step_solver(actual, trajectory["object_id"], trajectory["material_id"], dt,
                                                                   trajectory.get("object_relations"))
    rows, failure = [], None
    try:
        for offset in range(1, max(horizons) + 1):
            solver.p2g2p(initial + offset - 1, trajectory["mpm_config"]["substep_dt"], device=str(device))
            values = {"position": solver.export_particle_x_to_torch(), "velocity": solver.export_particle_v_to_torch(),
                      "F": wp.to_torch(solver.mpm_state.particle_F_trial), "C": wp.to_torch(solver.mpm_state.particle_C)}
            if any(not torch.isfinite(value).all() for value in values.values()):
                raise ValueError("Non-finite rollout state")
            target_keys = {"position": "x", "velocity": "v", "F": "F_trial", "C": "C"}
            rows.append(dict(step=offset, **{key + "_mse": (value - trajectory[target_keys[key]][initial + offset].to(device)).square().mean().item()
                                             for key, value in values.items()}))
    except (ValueError, RuntimeError) as error:
        failure = str(error)
    return rows, failure


def train_temporal_split(args):
    """Single-trajectory experiment, with model selection before the held-out test."""
    destination = Path(args.output)
    artifacts = [destination] + [destination.parent / name for name in
                                 ("metrics.json", "history.csv", "per_frame.csv", "rollout.csv", "validation_rollout.csv")]
    if any(path.exists() for path in artifacts):
        raise FileExistsError("Use a fresh experiment directory; refusing to overwrite results")
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    trajectory = load_trajectory(args)
    total = _validate_trajectory(trajectory) - 1
    if total < 50:
        raise ValueError("Temporal experiment needs at least 50 transitions")
    # ponytail: fixed chronological blocks suit this one-trajectory experiment;
    # independent trajectories are required for a cross-scene generalization claim.
    splits = {"train": list(range(int(.60 * total))),
              "validation": list(range(int(.62 * total), int(.78 * total))),
              "test": list(range(int(.80 * total), total))}
    endpoints = [{j for i in indices for j in (i, i + 1)} for indices in splits.values()]
    assert all(not endpoints[i] & endpoints[j] for i in range(3) for j in range(i + 1, 3))
    started = time.monotonic()
    print("experiment", json.dumps(vars(args)), flush=True)
    print("split", {name: [indices[0], indices[-1], len(indices)] for name, indices in splits.items()}, flush=True)
    samples = {s["frame_index"]: s for s in build_trajectory_samples(
        trajectory, device, frame_indices=sum(splits.values(), []))}
    baseline_seconds = time.monotonic() - started
    print(f"independent baseline samples ready: {len(samples)} in {baseline_seconds:.1f}s", flush=True)
    train_mean = torch.stack([samples[i]["target"].mean(0) for i in splits["train"]]).mean(0)
    train_target_rms = torch.stack([samples[i]["target"].square().mean() for i in splits["train"]]).mean().sqrt().item()
    residual_scale = max(train_target_rms, torch.finfo(torch.float32).eps) if args.normalize_residual else 1.
    model = PostStepRelationMPM(n_materials=len(MATERIAL_VOCAB), hidden_dim=args.hidden_dim,
                               local_radius=args.local_radius, contact_radius=args.contact_radius,
                               max_tokens=args.max_tokens, max_delta_v=args.max_delta_v,
                               use_state_features=args.use_state_features, residual_scale=residual_scale).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps, eta_min=args.lr * .01)

    @torch.no_grad()
    def evaluate(indices, constant=None):
        model.eval()
        rows = []
        for index in indices:
            sample = samples[index]
            prediction = predict_sample(model, sample) if constant is None else constant.expand_as(sample["target"])
            velocity_error = (prediction - sample["target"]).square().mean().item()
            position_error = (sample["x_base"] + sample["dt"] * prediction
                              - trajectory["x"][index + 1].to(device)).square().mean().item()
            rows.append({"frame": index, "velocity_mse": velocity_error, "position_mse": position_error})
        return rows

    train_monitor = splits["train"][::max(1, len(splits["train"]) // 64)]
    history, best_state, best_step, best_validation = [], None, None, float("inf")
    zero_validation = sum(row["velocity_mse"] for row in evaluate(splits["validation"], train_mean * 0)) / len(splits["validation"])
    zero_train = sum(row["velocity_mse"] for row in evaluate(train_monitor, train_mean * 0)) / len(train_monitor)
    validation_horizon = min(args.validation_rollout_steps, len(splits["validation"]))
    validation_rollouts = []
    baseline_rollout, failure = rollout_model(trajectory, None, splits["validation"][0], [validation_horizon], device)
    if failure:
        raise ValueError(f"Baseline validation rollout failed: {failure}")
    baseline_rollout_mse = baseline_rollout[-1]["position_mse"]
    validation_rollouts.extend(dict(checkpoint_step=0, model="mpm", **row) for row in baseline_rollout)
    best_score = float("inf")
    if args.allow_zero_correction:
        best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        best_step, best_validation, best_score = 0, zero_validation, 1.
    history.append(dict(step=0, train_monitor_mse=zero_train, validation_mse=zero_validation,
                        validation_rollout_position_mse=baseline_rollout_mse, selection_score=1.,
                        learning_rate=args.lr, elapsed_seconds=0.))
    order, stale = [], 0
    generator = torch.Generator().manual_seed(args.seed)
    optimization_started = time.monotonic()
    for step in range(1, args.steps + 1):
        if not order:
            order = [splits["train"][i] for i in torch.randperm(len(splits["train"]), generator=generator).tolist()]
        sample = samples[order.pop()]
        model.train()
        loss = ((predict_sample(model, sample) - sample["target"]) / residual_scale).square().mean()
        if not torch.isfinite(loss):
            raise ValueError(f"Non-finite training loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        if step % args.eval_every == 0 or step == args.steps:
            validation = sum(row["velocity_mse"] for row in evaluate(splits["validation"])) / len(splits["validation"])
            train_loss = sum(row["velocity_mse"] for row in evaluate(train_monitor)) / len(train_monitor)
            rollout_mse = float("nan")
            score = validation / max(zero_validation, 1e-30)
            eligible = True
            if args.select_closed_loop:
                candidate_rollout, failure = rollout_model(trajectory, model, splits["validation"][0], [validation_horizon], device)
                validation_rollouts.extend(dict(checkpoint_step=step, model="network", **row) for row in candidate_rollout)
                rollout_mse = candidate_rollout[-1]["position_mse"] if not failure else float("inf")
                score = .5 * score + .5 * rollout_mse / max(baseline_rollout_mse, 1e-30)
                # Both physical criteria must improve; do not trade closed-loop damage for one-step gains.
                eligible = validation <= zero_validation and rollout_mse <= baseline_rollout_mse
            history.append(dict(step=step, train_monitor_mse=train_loss, validation_mse=validation,
                                validation_rollout_position_mse=rollout_mse, selection_score=score,
                                learning_rate=scheduler.get_last_lr()[0],
                                elapsed_seconds=time.monotonic() - optimization_started))
            if eligible and score < best_score:
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                best_validation, best_step, best_score, stale = validation, step, score, 0
            else:
                stale += 1
            print(f"step={step} train_monitor={train_loss:.8g} validation={validation:.8g} val_rollout={rollout_mse:.8g} best_step={best_step}", flush=True)
            if args.patience and stale >= args.patience:
                break
    optimization_seconds = time.monotonic() - optimization_started
    if best_state is None:
        raise ValueError("No eligible checkpoint; enable --allow-zero-correction for a safe baseline candidate")
    metadata = dict(SAMPLE_METADATA, input_schema=model.input_schema, material_vocab=list(MATERIAL_VOCAB), relation_types=RELATION_TYPES,
                    hidden_dim=model.hidden_dim, n_materials=model.n_materials, local_radius=model.local_radius,
                    contact_radius=model.contact_radius, max_tokens=model.max_tokens, max_delta_v=model.max_delta_v,
                    use_state_features=model.use_state_features, residual_scale=model.residual_scale,
                    zero_correction=best_step == 0,
                    baseline_discretization={"transfer_scheme": trajectory["mpm_config"].get("transfer_scheme", "gradient"),
                                             "grid_mass_epsilon": trajectory["mpm_config"].get("grid_mass_epsilon", 0.)})
    with destination.open("xb") as output:
        torch.save({"model_state_dict": best_state, "metadata": metadata,
                    "training": vars(args) | {"samples": len(splits["train"]), "actual_steps": step,
                                               "best_step": best_step, "selection": "validation one-step and rollout" if args.select_closed_loop else "minimum validation velocity MSE"}}, output)
    load_relation_checkpoint(model, destination, MATERIAL_VOCAB)
    results, per_frame = {}, []
    for name, indices in splits.items():
        corrected = evaluate(indices)
        baseline = evaluate(indices, train_mean * 0)
        constant = evaluate(indices, train_mean)
        result = {}
        for label, rows in (("mpm", baseline), ("train_mean_correction", constant), ("network", corrected)):
            result[label] = {key: sum(row[key] for row in rows) / len(rows) for key in ("velocity_mse", "position_mse")}
            result[label].update(velocity_rmse=math.sqrt(result[label]["velocity_mse"]),
                                 position_rmse=math.sqrt(result[label]["position_mse"]))
        result["velocity_mse_reduction_percent"] = 100 * (1 - result["network"]["velocity_mse"] / result["mpm"]["velocity_mse"])
        result["frames_improved_percent"] = 100 * sum(c["velocity_mse"] < b["velocity_mse"] for c, b in zip(corrected, baseline)) / len(indices)
        results[name] = result
        for b, c, k in zip(baseline, corrected, constant):
            per_frame.append(dict(split=name, frame=b["frame"], mpm_velocity_mse=b["velocity_mse"],
                                  network_velocity_mse=c["velocity_mse"], constant_velocity_mse=k["velocity_mse"],
                                  mpm_position_mse=b["position_mse"], network_position_mse=c["position_mse"]))
    rollout, rollout_failure = [], {}
    initial = splits["test"][0]
    horizons = sorted({min(h, total - initial) for h in (20, 50, 100)})
    for label in ("mpm", "network"):
        rows, failure = rollout_model(trajectory, model if label == "network" else None, initial, horizons, device)
        rollout.extend(dict(model=label, **row) for row in rows)
        if failure:
            rollout_failure[label] = failure
    for filename, rows in (("history.csv", history), ("per_frame.csv", per_frame), ("rollout.csv", rollout),
                           ("validation_rollout.csv", validation_rollouts)):
        with (destination.parent / filename).open("x", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=list(rows[0]) if rows else ["model", "step", "velocity_mse", "position_mse"])
            writer.writeheader()
            writer.writerows(rows)
    source_paths = [Path(__file__), Path("PhysGaussian/relation_mpm.py"),
                    Path("PhysGaussian/mpm_solver_warp/mpm_solver_warp.py"), Path("PhysGaussian/mpm_solver_warp/mpm_utils.py")]
    import warp as wp
    report = dict(configuration=vars(args), baseline_mpm_config=trajectory["mpm_config"],
                  selected_zero_correction=best_step == 0, residual_scale=residual_scale, train_target_rms=train_target_rms,
                  validation_selection=dict(score=best_score, zero_velocity_mse=zero_validation,
                                            zero_rollout_position_mse=baseline_rollout_mse, horizon=validation_horizon),
                  data=dict(particles=len(trajectory["mass"]), frames=total + 1,
                  dt=trajectory["mpm_config"]["substep_dt"], source=trajectory.get("source"),
                  split_current_frames=splits, train_mean_delta_v=train_mean.cpu().tolist()),
                  best_step=best_step, actual_steps=step, baseline_seconds=baseline_seconds,
                  optimization_seconds=optimization_seconds, total_seconds=time.monotonic() - started,
                  parameters=sum(p.numel() for p in model.parameters()), metrics=results,
                  rollout_initial_frame=initial, rollout_horizons=horizons, rollout_failure=rollout_failure,
                  rollout_at_horizons=[row for row in rollout if row["step"] in horizons],
                  environment=dict(torch=str(torch.__version__), warp=getattr(wp.config, "version", None),
                                   python=sys.executable, warp_module=wp.__file__),
                  source_sha256={str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
                  limitations=["One trajectory, one object and one material; chronological extrapolation only",
                               "One-step targets reset the complete GT state for each sample",
                               "Closed-loop evaluation restores GT only at its initial state",
                               "C/F are observed only when use_state_features; the correction keeps the baseline strain state",
                               "Previously viewed test segment is a diagnostic comparison, not a fresh holdout",
                               "No independent-scene generalization or repeated-seed uncertainty estimate"])
    with (destination.parent / "metrics.json").open("x") as output:
        json.dump(report, output, indent=2)
    print("RESULT", json.dumps({"best_step": best_step, "metrics": results,
                               "rollout_at_horizons": report["rollout_at_horizons"]}), flush=True)


def physcore_trajectory_cases(dataset, trajectory=None):
    """Validate independent episode splits; retain the original one-trajectory split."""
    sys.path.insert(0, str(Path(__file__).resolve().parent / "tools"))
    from prepare_uniphy_jelly import to_physcore_episode

    names = ("train", "validation", "test")
    multiple = any(name + "_trajectories" in dataset for name in names)
    if multiple and (trajectory or dataset.get("trajectory")):
        raise ValueError("Explicit episode splits cannot be combined with --trajectory/dataset.trajectory")
    paths = {name: dataset.get(name + "_trajectories") for name in names} if multiple else {
        "all": [trajectory or dataset.get("trajectory")]}
    cases, seen_paths, seen_files, seen_content, seen_ids = {}, set(), set(), set(), set()
    for split, values in paths.items():
        if not isinstance(values, (list, tuple)) or not values:
            raise ValueError(f"dataset.{split}_trajectories must be a nonempty list of complete trajectory files")
        cases[split] = []
        for value in values:
            if not isinstance(value, (str, Path)) or not str(value):
                raise ValueError("Every trajectory path must be a nonempty string")
            path = Path(value).resolve(strict=True)
            if not path.is_file():
                raise ValueError(f"Trajectory is not a file: {path}")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if path in seen_paths or digest in seen_files:
                raise ValueError(f"Duplicate trajectory path/content in episode splits: {path}")
            data = torch.load(path, map_location="cpu", weights_only=True)
            episode = to_physcore_episode(data)
            source = data.get("source", {})
            ids = [item for item in (data.get("episode_id"), source.get("episode_id")) if item is not None]
            if any(isinstance(item, bool) or not isinstance(item, (str, int)) or not str(item).strip() for item in ids):
                raise ValueError(f"episode_id must be a nonempty string/integer: {path}")
            ids = {str(item).strip() for item in ids}
            if len(ids) > 1 or (multiple and not ids):
                raise ValueError(f"Independent trajectories require one unambiguous episode_id or source.episode_id: {path}")
            episode_id = next(iter(ids), path.stem)
            # Ignore source labels when detecting a re-saved copy of the same numerical trajectory.
            content = hashlib.sha256()
            for key in ("x", "v", "F_trial", "C", "time", "particle_id", "volume", *STATIC_FIELDS):
                tensor = data[key].contiguous()
                content.update(f"{key}:{tensor.dtype}:{tuple(tensor.shape)}".encode())
                content.update(memoryview(tensor.numpy()).cast("B"))
            content.update(json.dumps(data["mpm_config"], sort_keys=True, allow_nan=False).encode())
            content_digest = content.hexdigest()
            if content_digest in seen_content or (multiple and episode_id in seen_ids):
                raise ValueError(f"Duplicate numerical trajectory/source episode_id in episode splits: {path}")
            physics = episode["physcore_adapter"]
            json.dumps(physics, allow_nan=False)  # Check provenance/scene metadata before writing a checkpoint.
            total = len(episode["particle_coords"]) - 1
            cases[split].append(dict(path=str(path), episode_id=episode_id, sha256=digest,
                                     content_sha256=content_digest, physics=physics,
                                     particles=int(episode["particle_coords"].shape[1]), first=0, stop=total))
            seen_paths.add(path); seen_files.add(digest); seen_content.add(content_digest); seen_ids.add(episode_id)
    if not multiple:
        case = cases["all"][0]
        total = case["stop"]
        if total < 50:
            raise ValueError("The numeric chronological experiment requires at least 50 transitions")
        cases = {name: [dict(case, first=int(total * first), stop=int(total * stop))]
                 for name, first, stop in (("train", 0., .60), ("validation", .62, .78), ("test", .80, 1.))}
    return cases, multiple


def make_physcore_corrector(cfg, device):
    from omegaconf import OmegaConf
    correction = cfg.get("correction", cfg.get("gvc"))
    kind = cfg.get("model", {}).get("kind", "gvc")
    if kind == "worldparticle":
        if device.type == "cuda":
            torch.cuda.set_device(device)  # Open3D CUDA neighbor search follows the current device.
        from PhysGaussian.worldparticle_grid_correction import WorldParticlePhysCoreCorrector
        options = OmegaConf.to_container(cfg.model, resolve=True)
        options.pop("kind")
        return WorldParticlePhysCoreCorrector(n_materials=len(MATERIAL_VOCAB),
            max_delta_v=float(correction.max_delta_v), **options).to(device)
    if kind != "gvc":
        raise ValueError("model.kind must be worldparticle or gvc")
    from physcore.model_RfD import GridVelocityCorrector
    return GridVelocityCorrector(in_channels=19, cond_dim=16, base_channels=int(correction.base_channels),
        levels=int(correction.levels), max_delta_v=float(correction.max_delta_v)).to(device)


def make_physcore_engine(episode, rollout_config, device, corrector=None, cadence=1, material_only=False):
    """Reuse the pinned official solver, overriding only particle/scene inputs."""
    from omegaconf import OmegaConf
    from physcore.sim.mpm import BatchedMPMSolver, add_batched_surface_collider

    physics = episode["physcore_adapter"]
    worldparticle = material_only or bool(getattr(corrector, "worldparticle_core", False))
    if worldparticle and cadence != 1:
        raise ValueError("WorldParticle correction currently requires h=1")
    material_id = episode["material_id"].to(device) if worldparticle and not material_only else None
    if worldparticle:
        from physcore.particle_flow.rollout import BatchedDifferentiableRolloutEngine
        class CorrectionEngine(BatchedDifferentiableRolloutEngine):
            def attach_corrector(self, model, h=1):
                self._gvc_active = True
            def enable_corrector(self, enabled):
                self._gvc_active = bool(enabled)
            def forward(self, positions, velocities, F, deltas, log_E, nu, **kwargs):
                self._gvc_log_E, self._gvc_nu = log_E, nu
                try:
                    return super().forward(positions, velocities, F, deltas, log_E, nu, **kwargs)
                finally:
                    self._gvc_log_E = self._gvc_nu = None
    else:
        from physcore.particle_flow.rfd_engine import GVCRolloutEngine as CorrectionEngine
    config = OmegaConf.to_container(rollout_config, resolve=True) if OmegaConf.is_config(rollout_config) else dict(rollout_config)
    config.update(default_dt=physics["dt"], default_steps_per_frame=1,
                  num_grids=physics["n_grid"], gravity=physics["gravity"],
                  rho=physics["density"], damping=physics["damping"])
    floors = [bc["point"][2] for bc in physics["boundary_conditions"] if bc["normal"] == [0., 0., 1.]]
    if not floors:
        raise ValueError("The numeric adapter needs an explicit z-up floor for RfD features")
    config["ground_height"] = max(floors)

    class TrajectoryEngine(CorrectionEngine):
        def _install_gvc_hooks(self, model):
            if material_only:
                return  # Material gradients use the original MPM without a dynamics residual.
            if not worldparticle:
                return super()._install_gvc_hooks(model)
            original = model.p2g2p
            def with_particle_state(x, v, C, F, stress, *args, **kwargs):
                model._worldparticle_state = (x, v, F, C)
                try:
                    return original(x, v, C, F, stress, *args, **kwargs)
                finally:
                    del model._worldparticle_state
            model.p2g2p = with_particle_state
            def correct_grid(model):
                if self._gvc_active:
                    delta = corrector(*model._worldparticle_state, self._gvc_log_E, self._gvc_nu,
                                      model.grid_m, physics, material_id)
                    model.grid_mv = model.grid_mv + delta
            model.post_grid_process.append(correct_grid)

        def _make_batched_solver(self, positions):
            solver = BatchedMPMSolver(num_grids=self.num_grids, dt=self.dt,
                gravity=self.gravity, damping=self.damping, clip_bound_factor=self.clip_bound, device=str(self.device))
            # The upstream factory infers volume from the current AABB. Preserve the supplied rest volume instead.
            state = solver.init_particles(positions, rho=self.rho,
                vol_override=positions.new_full((len(positions),), physics["volume"]))
            for bc in physics["boundary_conditions"]:
                before = len(solver.post_step_process)
                add_batched_surface_collider(solver, point=bc["point"], normal=bc["normal"],
                    surface="coulomb" if bc["surface"] == "separate" else bc["surface"], friction=bc["friction"], elasticity=0.)
                if bc["surface"] == "separate":
                    # Zero-friction Coulomb is separation; keep the source's grid-only boundary, not particle projection.
                    del solver.post_step_process[before:]
            self._install_gvc_hooks(solver.model)
            return solver, state

    engine = TrajectoryEngine(OmegaConf.create(config), device=device)
    if physics.get("elasticity_model") == "NeoHookeanElasticity":
        class NeoHookeanElasticity(torch.nn.Module):
            """Original UniPhy/PAC Kirchhoff stress; the upstream MPM and RfD remain unchanged."""
            def forward(self, F, log_E, nu):
                E = log_E.exp()
                mu = (E / (2 * (1 + nu))).reshape(-1, 1, 1)
                lam = (E * nu / ((1 + nu) * (1 - 2 * nu))).reshape(-1, 1, 1)
                eye = torch.eye(3, device=F.device, dtype=F.dtype)
                J = torch.linalg.det(F).reshape(-1, 1, 1)
                stress = mu * (F @ F.transpose(-1, -2) - eye) + lam * J.log() * eye
                if not torch.isfinite(stress).all() or not (J > 0).all():
                    raise ValueError(f"Invalid Neo-Hookean state: min(det F)={J.detach().min().item()}")
                return stress
        # The pinned registry lacks Neo-Hookean. Adapt this formula, rather than asking RfD to learn the wrong law.
        engine.elasticity_model = NeoHookeanElasticity().to(device).eval()
    if corrector is not None:
        engine.attach_corrector(corrector, h=cadence)
    return engine


def physcore_state(episode, index, device):
    return {key: episode[field][index:index + 1].to(device) for key, field in
            (("x", "particle_coords"), ("v", "particle_velocities"),
             ("F", "particle_deformation_gradients"), ("C", "particle_affine_velocity_matrices"))}


def physcore_advance(engine, state, material, steps, **controls):
    """No GT reset inside a segment; the official grid hook updates x/v/F/C together."""
    out = engine(state["x"], state["v"], state["F"],
                 state["x"].new_zeros((steps,) + tuple(state["x"].shape)),
                 material["log_E"], material["nu"], C=state["C"], **controls)
    state = {"x": out["predicted_positions"][-1], "v": out["final_velocity"],
             "F": out["final_deformation_gradient"], "C": out["final_C"]}
    if any(not torch.isfinite(value).all() for value in state.values()):
        raise ValueError("Non-finite official PhysCoRe rollout state")
    return out, state


def train_material_step(model, engine, optimizer, batch,
                        w_track=1., w_depth=1., w_prior=1e-4):
    from physcore.particle_flow.rfd_runtime import tracked_l2, chamfer_gt_to_pred

    observations = batch["observations"]
    if not observations:
        raise ValueError("Material training requires observed video frames")

    model.train()
    engine.enable_corrector(False)
    optimizer.zero_grad(set_to_none=True)
    prior, state = batch["prior"], batch["state"]
    material = model(
        state["x"][0], state["v"][0], batch["motion_features"],
        prior, batch["boundary"], batch["normals"]
    )
    out, _ = physcore_advance(
        engine, state, material, batch["steps"], **batch.get("controls", {})
    )
    predictions = out["predicted_positions"][batch["frame_indices"], 0]
    tracking = predictions.new_zeros(())
    depth = predictions.new_zeros(())
    for prediction, observation in zip(predictions, observations, strict=True):
        if observation.get("tracked_idx") is not None:
            tracking = tracking + tracked_l2(
                observation["coords"], prediction, observation["tracked_idx"]
            )
        if observation.get("depth_points") is not None:
            depth = depth + chamfer_gt_to_pred(
                observation["depth_points"], observation["depth_valid"], prediction
            )
    motion_loss = (w_track * tracking + w_depth * depth) / len(observations)
    if not motion_loss.requires_grad:
        raise ValueError("No differentiable motion supervision; check observations and MPM graph")

    prior_loss = ((material["log_E"][0] - prior["E"].log()) / math.log(10.)).square().mean()
    prior_loss = prior_loss + ((material["nu"][0] - prior["nu"]) / .05).square().mean()
    loss = motion_loss + w_prior * prior_loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), .5, error_if_nonfinite=True)
    optimizer.step()
    return float(loss.detach())


def physcore_gt_step(engine, episode, index, material, device):
    """One independent observed-state pair; never accept a prior predicted state."""
    return physcore_advance(engine, physcore_state(episode, index, device), material, 1)


def train_physcore(args, overrides):
    """Delegate full MfM/RfD to upstream; use its actual core for known-material numeric episodes."""
    import importlib
    import runpy
    from omegaconf import OmegaConf

    vendor = Path(__file__).resolve().parent / "third_party" / "PhysCoRe"
    if not (vendor / "physcore/particle_flow/rfd_engine.py").is_file():
        raise FileNotFoundError("Pinned official PhysCoRe source is required at third_party/PhysCoRe")
    sys.path.insert(0, str(vendor))
    import warp as wp
    wp.config.kernel_cache_dir = os.environ.get("PHYSPRP_WARP_CACHE_DIR", str(vendor.parents[1] / ".cache/warp"))
    wp.init()
    legacy_options = {"--hidden-dim", "--local-radius", "--contact-radius", "--max-tokens", "--max-delta-v",
        "--temporal-split", "--eval-every", "--patience", "--use-state-features", "--normalize-residual",
        "--allow-zero-correction", "--select-closed-loop", "--validation-rollout-steps"}
    if any(token.split("=", 1)[0] in legacy_options for token in sys.argv[1:]):
        raise ValueError("PhysCoRe architecture, rollout and selection are configured in YAML; post-step options require --pipeline post-step")
    if args.stage != "known-material-rfd":
        if not args.config:
            raise ValueError("Official MfM/RfD stages require --config pointing to real/augmented episode roots")
        if args.trajectory or args.gt or args.output or args.steps is not None:
            raise ValueError("Official stages use their YAML dataset/output/budget settings; use key=value overrides")
        name = ("validate_" if args.evaluate_only else "train_") + {"mfm": "MfM", "rfd": "RfD"}[args.stage] + ".py"
        forwarded = list(overrides)
        if args.checkpoint:
            if args.stage != "mfm" or not args.evaluate_only:
                raise ValueError("Use the official stage's YAML checkpoint settings; --checkpoint is for known-material/MfM evaluation")
            forwarded += ["--checkpoint", args.checkpoint]
        if args.device:
            forwarded += ["--device", args.device] if args.evaluate_only and args.stage == "mfm" else ["train.device=" + args.device]
        if args.lr is not None:
            if args.evaluate_only:
                raise ValueError("--lr is a training option")
            forwarded += ["train.lr=" + str(args.lr)]
        if args.seed is not None:
            forwarded += ["train.seed=" + str(args.seed)]
        if args.stage == "mfm" and not args.evaluate_only and forwarded and forwarded != ["--help"]:
            raise ValueError("The original MfM trainer accepts --config only; put its training overrides in that YAML")
        previous = sys.argv
        try:
            sys.argv = [str(vendor / name), "--config", args.config] + forwarded
            runpy.run_path(str(vendor / name), run_name="__main__")
        finally:
            sys.argv = previous
        return
    if args.gt or args.transfer_scheme is not None or args.grid_mass_epsilon is not None:
        raise ValueError("PhysCoRe uses its official grid solver, not precomputed particle targets or PhysGaussian overrides")
    cfg = OmegaConf.load(args.config or Path(__file__).resolve().parent / "config/physcore_jelly.yaml")
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    if args.seed is not None:
        cfg.train.seed = args.seed
    sys.path.insert(0, str(Path(__file__).resolve().parent / "tools"))
    from prepare_uniphy_jelly import to_physcore_episode
    cases, multiple = physcore_trajectory_cases(OmegaConf.to_container(cfg.dataset, resolve=True), args.trajectory)
    split_ranges = {name: [dict(episode_id=case["episode_id"], frames=[case["first"], case["stop"]]) for case in items]
                    if multiple else [items[0]["first"], items[0]["stop"]] for name, items in cases.items()}
    device = torch.device(args.device or cfg.train.device)
    torch.manual_seed(int(cfg.train.seed))
    correction = cfg.get("correction", cfg.get("gvc"))
    if correction is None:
        raise ValueError("A correction configuration is required")
    chunk = int(correction.bptt_frames)
    cadence = int(correction.h)
    epochs = int(cfg.train.epochs)
    mode = str(cfg.train.get("mode", "rollout"))
    if mode not in ("rollout", "single-step"):
        raise ValueError("train.mode must be rollout or single-step")
    single_step = mode == "single-step"
    batch_size = int(cfg.train.get("batch_size", 1))
    if single_step and (chunk != 1 or cadence != 1 or batch_size < 1):
        raise ValueError("Independent single-step training requires bptt_frames=h=1 and positive batch_size")
    max_updates = args.steps if args.steps is not None else int(cfg.train.get("max_updates", 0))
    lr = args.lr if args.lr is not None else float(cfg.train.lr)
    if min(chunk, cadence, epochs) < 1 or chunk % cadence or max_updates < 0 or not math.isfinite(lr) or lr <= 0:
        raise ValueError("Require positive epochs/TBPTT/cadence/lr, TBPTT divisible by cadence, nonnegative max_updates")
    if any((case["stop"] - case["first"]) % cadence for case in cases["train"]):
        raise ValueError("Training episode lengths must be divisible by gvc.h; use h=1 to consume every saved substep")
    model = make_physcore_corrector(cfg, device)
    if getattr(model, "worldparticle_core", False):
        # Full numerical tracks: same mean particle-distance objective, without importing the unused U-Net runtime.
        def tracked_l2(gt, pred, indices):
            return (pred[indices] - gt[indices]).norm(dim=-1).mean()
    else:
        from physcore.particle_flow.rfd_runtime import tracked_l2
    if getattr(model, "worldparticle_core", False) and not single_step:
        raise ValueError("WorldParticle is verified for independent single-step training; set train.mode=single-step")
    destination = Path(args.output or Path(cfg.train.output_dir) / "checkpoint.pt")
    cfg.train.lr, cfg.train.device, cfg.train.max_updates = lr, str(device), max_updates
    cfg.train.output_dir = str(destination.parent)
    if multiple:
        for name, items in cases.items():
            cfg.dataset[name + "_trajectories"] = [case["path"] for case in items]
    else:
        cfg.dataset.trajectory = cases["train"][0]["path"]
    if args.evaluate_only:
        if not args.checkpoint:
            raise ValueError("--evaluate-only requires --checkpoint from the same PhysCoRe pipeline")
    elif destination.exists() or (destination.parent / "metrics.json").exists():
        raise FileExistsError(f"Refusing to overwrite the completed experiment: {destination.parent}")
    physics = cases["train"][0]["physics"]
    contract = dict(schema="physprp-physcore-known-material-rfd-v1", upstream_commit="ba7ddf744c7468369529d3baf996fb829fec30dd",
        physics=physics, gvc=OmegaConf.to_container(correction, resolve=True),
        rollout=OmegaConf.to_container(cfg.rollout, resolve=True),
        trajectory_sha256=cases["train"][0]["sha256"],
        official_discretization="BatchedMPMModel: B-spline gradient stress/F; grid mass clamp 1e-15; no epsilon addition")
    if multiple:
        contract.pop("physics"); contract.pop("trajectory_sha256")
        contract.update(schema="physprp-physcore-multi-episode-rfd-v1", dataset=cases,
                        metric_aggregation="unweighted episode means", phase="FiLM phase restarts per TBPTT engine call; physical state restarts only at episode start")
    if single_step:
        contract.update(schema="physprp-physcore-single-step-v1", batch_size=batch_size,
                        phase="one-step engine calls in training and evaluation; training resets complete GT state for every pair",
                        selection="mean validation one-step position MSE, including zero correction")
    if getattr(model, "worldparticle_core", False):
        contract.pop("gvc")
        contract.update(schema="physprp-worldparticle-single-step-v1", correction=OmegaConf.to_container(correction, resolve=True),
                        model=OmegaConf.to_container(cfg.model, resolve=True), worldparticle=model.provenance())

    context = None

    def load_context(case):
        nonlocal context
        if context is not None and context["path"] == case["path"]:
            return context
        path = Path(case["path"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != case["sha256"]:
            raise ValueError(f"Trajectory changed after split validation: {path}")
        episode = to_physcore_episode(torch.load(path, map_location="cpu", weights_only=True))
        # ponytail: keep only the most recent episode; a large corpus must not retain every full F/C trajectory in RAM.
        context = dict(path=case["path"], episode=episode,
                       engine=make_physcore_engine(episode, cfg.rollout, device, model, cadence),
                       material={key: value.unsqueeze(0).to(device) for key, value in episode["particle_material_params"].items()},
                       indices=torch.arange(episode["particle_coords"].shape[1], device=device))
        return context

    @torch.no_grad()
    def evaluate(case, enable, save_positions=False, teacher_forcing=False):
        ctx = load_context(case)
        episode, engine, material, indices = (ctx[key] for key in ("episode", "engine", "material", "indices"))
        engine.enable_corrector(enable)
        model.eval()
        first, stop = case["first"], case["stop"]
        state = physcore_state(episode, first, device)
        rows, predictions = [], []
        for start in range(first, stop, chunk):
            if teacher_forcing:
                if chunk != 1:
                    raise ValueError("Teacher-forced evaluation requires one-step engine calls")
            count = min(chunk, stop - start)
            out, state = (physcore_gt_step(engine, episode, start, material, device) if teacher_forcing else
                          physcore_advance(engine, state, material, count))
            for offset, pred in enumerate(out["predicted_positions"][:, 0], 1):
                gt = episode["particle_coords"][start + offset].to(device)
                rows.append(dict(episode_id=case["episode_id"], frame=start + offset, position_mse=(pred - gt).square().mean().item(),
                                 tracked_l2=tracked_l2(gt, pred, indices).item()))
                if teacher_forcing:
                    rows[-1]["velocity_mse"] = (state["v"][0] - episode["particle_velocities"][start + 1].to(device)).square().mean().item()
                if save_positions:
                    predictions.append(pred.cpu())
        result = dict(position_mse=sum(r["position_mse"] for r in rows) / len(rows),
                    endpoint_position_mse=rows[-1]["position_mse"],
                    tracked_l2=sum(r["tracked_l2"] for r in rows) / len(rows),
                    endpoint_velocity_mse=(state["v"][0] - episode["particle_velocities"][stop].to(device)).square().mean().item())
        if teacher_forcing:
            result["velocity_mse"] = sum(r["velocity_mse"] for r in rows) / len(rows)
        return result, rows, torch.stack(predictions) if save_positions else None

    def evaluate_split(name, enable, save_positions=False, teacher_forcing=False):
        details, rows, trajectories = [], [], {}
        for case in cases[name]:
            result, episode_rows, positions = evaluate(case, enable, save_positions, teacher_forcing)
            details.append(dict(episode_id=case["episode_id"], trajectory=case["path"],
                                first=case["first"], stop=case["stop"], **result))
            rows.extend(episode_rows)
            print("physcore evaluation", json.dumps(dict(split=name, episode_id=case["episode_id"],
                  corrector=enable, teacher_forcing=teacher_forcing, **result)), flush=True)
            if save_positions:
                rotation = positions.new_tensor(case["physics"]["rotation"])
                translation = positions.new_tensor(case["physics"]["translation"])
                trajectories[case["episode_id"]] = dict(positions_z_up=positions,
                    positions_original=(positions - translation) @ rotation,
                    source_frames=torch.arange(case["first"] + 1, case["stop"] + 1), trajectory=case["path"])
        aggregate = {key: sum(item[key] for item in details) / len(details) for key in
                     ("position_mse", "endpoint_position_mse", "tracked_l2", "endpoint_velocity_mse")}
        if teacher_forcing:
            aggregate["velocity_mse"] = sum(item["velocity_mse"] for item in details) / len(details)
        return aggregate, details, rows, trajectories

    def restore(checkpoint):
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved.get("contract") != contract:
            raise ValueError("Checkpoint architecture, physical scene or adapter contract does not match")
        model.load_state_dict(saved["model_state_dict"], strict=True)
        return saved

    if args.evaluate_only:
        saved = restore(args.checkpoint)
        if single_step:
            target = destination.parent / "evaluation.json"
            if target.exists():
                raise FileExistsError(f"Refusing to overwrite {target}")
            evaluations = {}
            for label, teacher in (("single_step", True), ("closed_loop", False)):
                evaluations[label] = {}
                all_rows = []
                for name, enable in (("baseline", False), ("selected", True), ("last_candidate", True)):
                    # Selection is already frozen; retain the rejected final candidate as a labelled diagnostic.
                    model.load_state_dict(saved["last_model_state_dict"] if name == "last_candidate" else saved["model_state_dict"], strict=True)
                    result, details, rows, _ = evaluate_split("test", enable, teacher_forcing=teacher)
                    evaluations[label][name] = dict(aggregate=result, per_episode=details)
                    all_rows.extend(dict(model=name, **row) for row in rows)
                with (destination.parent / (label + ".csv")).open("x", newline="") as output:
                    writer = csv.DictWriter(output, fieldnames=list(all_rows[0])); writer.writeheader(); writer.writerows(all_rows)
            with target.open("x") as output:
                json.dump(dict(contract=contract, selected_step=saved["selected_step"], evaluations=evaluations), output, indent=2)
            print("SINGLE STEP EVALUATION", json.dumps(evaluations), flush=True)
            return
        result, details, _, _ = evaluate_split("test", True)
        print("EVALUATION", json.dumps(dict(checkpoint=args.checkpoint, result=result,
              per_episode=details, selected_step=saved["selected_step"])))
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, destination.parent / "config.yaml")
    baseline_val, baseline_val_episodes, _, _ = evaluate_split("validation", False, teacher_forcing=single_step)
    best_metric, best_endpoint, best_step = baseline_val["position_mse"], baseline_val["endpoint_position_mse"], 0
    best_state = {key: value.cpu().clone() for key, value in model.state_dict().items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=float(cfg.train.weight_decay))
    history, validations, updates, started = [], [], 0, time.monotonic()
    for epoch in range(1, epochs + 1):
        decay_start, decay_end = int(cfg.train.get("lr_decay_start", epochs)), int(cfg.train.get("lr_decay_end", epochs))
        alpha = min(max((epoch - decay_start) / max(decay_end - decay_start, 1), 0.), 1.)
        for group in optimizer.param_groups:
            group["lr"] = lr + alpha * (float(cfg.train.get("final_lr", lr)) - lr)
        generator = torch.Generator().manual_seed(int(cfg.train.seed) + epoch)
        for index in torch.randperm(len(cases["train"]), generator=generator).tolist():
            case = cases["train"][index]
            ctx = load_context(case)
            episode, engine, material, indices = (ctx[key] for key in ("episode", "engine", "material", "indices"))
            engine.enable_corrector(True)
            model.train()
            state = physcore_state(episode, case["first"], device)
            frame_order = torch.randperm(case["stop"] - case["first"], generator=generator) + case["first"] if single_step else None
            stride = batch_size if single_step else chunk
            for start in range(case["first"], case["stop"], stride):
                count = min(stride, case["stop"] - start)
                optimizer.zero_grad(set_to_none=True)
                if single_step:
                    frames = frame_order[start - case["first"]:start - case["first"] + count]
                    losses = []
                    # Independent GT pairs; accumulate gradients without retaining 50 separate simulation graphs.
                    for frame in frames.tolist():
                        out, _ = physcore_gt_step(engine, episode, frame, material, device)
                        pair_loss = tracked_l2(episode["particle_coords"][frame + 1].to(device), out["predicted_positions"][0, 0], indices)
                        if not torch.isfinite(pair_loss):
                            raise ValueError(f"Non-finite single-step loss: episode={case['episode_id']}, frame={frame}")
                        (pair_loss / count).backward()
                        losses.append(pair_loss.detach())
                    loss = torch.stack(losses).mean()
                else:
                    out, state = physcore_advance(engine, state, material, count)
                    loss = sum(tracked_l2(episode["particle_coords"][start + offset].to(device), pred, indices)
                               for offset, pred in enumerate(out["predicted_positions"][:, 0], 1)) / count
                if not torch.isfinite(loss):
                    raise ValueError(f"Non-finite PhysCoRe reconstruction loss at {updates + 1}, episode={case['episode_id']}")
                if not single_step:
                    loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.train.grad_clip), error_if_nonfinite=True)
                optimizer.step()
                updates += 1
                # TBPTT truncates the graph, not the physical trajectory or its F/C history.
                state = {key: value.detach() for key, value in state.items()}
                history.append(dict(update=updates, epoch=epoch, episode_id=case["episode_id"],
                                    source_frame=start, frames=count, tracked_l2=float(loss.detach()), grad_norm=float(norm)))
                if single_step:
                    history[-1]["source_frames"] = " ".join(map(str, frames.tolist()))
                print(f"physcore update={updates} epoch={epoch} episode={case['episode_id']} frame={start} "
                      f"tracked_l2={float(loss):.8g} grad_norm={float(norm):.6g}", flush=True)
                if max_updates and updates >= max_updates:
                    break
            if max_updates and updates >= max_updates:
                break
        validation, validation_episodes, _, _ = evaluate_split("validation", True, teacher_forcing=single_step)
        validations.append(dict(epoch=epoch, update=updates, **validation, per_episode=validation_episodes))
        if validation["position_mse"] < best_metric and (single_step or validation["endpoint_position_mse"] <= best_endpoint):
            best_metric, best_endpoint, best_step = validation["position_mse"], validation["endpoint_position_mse"], updates
            best_state = {key: value.cpu().clone() for key, value in model.state_dict().items()}
        print("physcore validation", json.dumps(validation | {"selected_step": best_step}), flush=True)
        if max_updates and updates >= max_updates:
            break
    with destination.open("xb") as output:
        torch.save(dict(model_state_dict=best_state, contract=contract, selected_step=best_step,
                        actual_updates=updates, zero_correction=best_step == 0,
                        **({"last_model_state_dict": {key: value.cpu().clone() for key, value in model.state_dict().items()}} if single_step else {})), output)
    with (destination.parent / "history.csv").open("x", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(history[0])); writer.writeheader(); writer.writerows(history)
    if single_step:
        baseline_train, baseline_train_episodes, _, _ = evaluate_split("train", False, teacher_forcing=True)
        final_train, final_train_episodes, _, _ = evaluate_split("train", True, teacher_forcing=True)
        metrics = dict(contract=contract, actual_updates=updates, selected_step=best_step,
                       selected_zero_correction=best_step == 0, baseline_validation=baseline_val,
                       baseline_validation_episodes=baseline_val_episodes, validation_history=validations,
                       baseline_train=baseline_train, baseline_train_episodes=baseline_train_episodes,
                       final_train=final_train, final_train_episodes=final_train_episodes,
                       optimization_seconds=time.monotonic() - started,
                       source_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in
                                      (Path(__file__), Path(to_physcore_episode.__code__.co_filename))})
        with (destination.parent / "metrics.json").open("x") as output:
            json.dump(metrics, output, indent=2)
        print("SINGLE STEP TRAINING COMPLETE", json.dumps(metrics), flush=True)
        return
    restore(destination)
    # No test rollout is run until the validation-selected weights have been restored.
    baseline_test, baseline_test_episodes, baseline_rows, _ = evaluate_split("test", False)
    test, test_episodes, rows, trajectories = evaluate_split("test", True, save_positions=True)
    with (destination.parent / "test_rollout.csv").open("x", newline="") as output:
        if multiple:
            rows = [dict(model="mpm", **row) for row in baseline_rows] + [dict(model="network", **row) for row in rows]
        writer = csv.DictWriter(output, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    with (destination.parent / "test_rollout.pt").open("xb") as output:
        torch.save(dict(episodes=trajectories) if multiple else next(iter(trajectories.values())), output)
    source_paths = [Path(__file__), Path(to_physcore_episode.__code__.co_filename),
                    Path(importlib.import_module("physcore.model_RfD").__file__),
                    Path(importlib.import_module("physcore.particle_flow.rfd_engine").__file__),
                    Path(importlib.import_module("physcore.mpm.mpm_model").__file__),
                    Path(importlib.import_module("physcore.mpm.warp.svd").__file__)]
    metrics = dict(pipeline="physcore", stage="known-material-rfd", contract=contract,
        trajectory=None if multiple else cases["train"][0]["path"], splits=split_ranges, actual_updates=updates, selected_step=best_step,
        selected_zero_correction=best_step == 0, baseline_validation=baseline_val,
        baseline_validation_episodes=baseline_val_episodes,
        baseline_test=baseline_test, test=test, validation_history=validations, optimization_seconds=time.monotonic() - started,
        baseline_test_episodes=baseline_test_episodes, test_episodes=test_episodes,
        source_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths},
        limitations=["GT material is provided; MfM is not trained or claimed in this numeric probe",
                     "Training loss is upstream tracked_l2, through the actual grid-corrected MPM",
                     "Neo-Hookean source scenes use their original stress law; upstream GVC stress features remain a corotated proxy",
                     "No RGB-D/Chamfer/PID adaptation is fabricated; evaluation starts from a complete observed state",
                     "Independent episode split; no repeated-seed uncertainty estimate" if multiple else
                     "One previously inspected trajectory is a diagnostic test, not independent-scene validation"])
    with (destination.parent / "metrics.json").open("x") as output:
        json.dump(metrics, output, indent=2)
    print("PHYSCORE RESULT", json.dumps(metrics), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline", choices=("physcore", "post-step"), default="physcore")
    parser.add_argument("--stage", choices=("mfm", "rfd", "known-material-rfd"), default="known-material-rfd")
    parser.add_argument("--config", help="PhysCoRe YAML; official stages use their original episode format")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--checkpoint", help="Known-material PhysCoRe checkpoint for --evaluate-only")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--gt", help="Explicit post-MPM baseline sample (.pt); legacy force labels are rejected")
    source.add_argument("--trajectory", help="Complete elastic GT states at every MPM substep (.pt)")
    parser.add_argument("--output")
    parser.add_argument("--steps", type=int, help="Optimizer steps; optional cap for the PhysCoRe numeric probe")
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--local-radius", type=float)
    parser.add_argument("--contact-radius", type=float)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--max-delta-v", type=float)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--device")
    parser.add_argument("--temporal-split", action="store_true", help="Chronological train/validation/test blocks with gaps and a report")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--eval-every", type=int, default=600)
    parser.add_argument("--patience", type=int, default=0, help="Stop after this many evaluations without improvement; 0 disables")
    parser.add_argument("--use-state-features", action="store_true", help="Observe baseline F-I and dt*C, using the solid-state schema")
    parser.add_argument("--normalize-residual", action="store_true", help="Scale output and loss by training-target RMS")
    parser.add_argument("--allow-zero-correction", action="store_true", help="Include the unchanged MPM as a validation candidate")
    parser.add_argument("--select-closed-loop", action="store_true", help="Require validation one-step and rollout improvements")
    parser.add_argument("--validation-rollout-steps", type=int, default=100)
    parser.add_argument("--transfer-scheme", choices=("gradient", "mls"), help="Explicit MPM discretization override; preserve dataset metadata if omitted")
    parser.add_argument("--grid-mass-epsilon", type=float, help="Explicit source grid-mass regularization override")
    args, overrides = parser.parse_known_args()
    if args.pipeline == "physcore":
        train_physcore(args, overrides)
        return
    if overrides or not (args.gt or args.trajectory) or not args.output or args.local_radius is None or args.contact_radius is None:
        parser.error("post-step requires --gt/--trajectory, --output, --local-radius and --contact-radius; unknown options are rejected")
    args.steps = args.steps if args.steps is not None else 300
    args.lr = args.lr if args.lr is not None else 2e-3
    args.device = args.device or "cuda:0"
    args.seed = args.seed if args.seed is not None else 7
    if args.steps < 1 or not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("Require positive --steps and finite positive --lr")
    if args.eval_every < 1 or args.patience < 0 or args.validation_rollout_steps < 1 or (args.temporal_split and not args.trajectory):
        parser.error("Require positive --eval-every, nonnegative --patience and --trajectory for --temporal-split")
    if args.grid_mass_epsilon is not None and (not math.isfinite(args.grid_mass_epsilon) or args.grid_mass_epsilon < 0):
        parser.error("grid-mass-epsilon must be finite and nonnegative")
    if args.gt and (args.transfer_scheme is not None or args.grid_mass_epsilon is not None):
        parser.error("A precomputed --gt baseline cannot be changed by MPM overrides; use --trajectory")
    if args.temporal_split:
        train_temporal_split(args)
        return
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    device = torch.device(args.device)
    samples = ([load_gt(args.gt, device)] if args.gt else
               build_trajectory_samples(load_trajectory(args), device))
    residual_scale = max(torch.stack([s["target"].square().mean() for s in samples]).mean().sqrt().item(), torch.finfo(torch.float32).eps) if args.normalize_residual else 1.
    model = PostStepRelationMPM(n_materials=len(MATERIAL_VOCAB), hidden_dim=args.hidden_dim,
                                local_radius=args.local_radius, contact_radius=args.contact_radius,
                                max_tokens=args.max_tokens, max_delta_v=args.max_delta_v,
                                use_state_features=args.use_state_features, residual_scale=residual_scale).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    for step in range(args.steps):
        sample = samples[step % len(samples)]
        prediction = predict_sample(model, sample)
        loss = ((prediction - sample["target"]) / residual_scale).square().mean()
        if not torch.isfinite(loss):
            raise ValueError(f"Non-finite training loss at step {step + 1}")
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if step == 0 or (step + 1) % 25 == 0:
            print(f"step {step + 1} sample {step % len(samples)} loss {float(loss):.6g}")
    metadata = dict(SAMPLE_METADATA, input_schema=model.input_schema, material_vocab=list(MATERIAL_VOCAB), relation_types=RELATION_TYPES,
                    hidden_dim=model.hidden_dim, n_materials=model.n_materials,
                    local_radius=model.local_radius, contact_radius=model.contact_radius,
                    max_tokens=model.max_tokens, max_delta_v=model.max_delta_v,
                    use_state_features=model.use_state_features, residual_scale=model.residual_scale)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as destination:
        torch.save({"model_state_dict": model.state_dict(), "metadata": metadata,
                    "training": vars(args) | {"samples": len(samples)}}, destination)
    print(f"saved {output}; samples={len(samples)}; target=v_true_next-v_base")


if __name__ == "__main__":
    main()

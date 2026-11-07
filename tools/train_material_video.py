"""Observed video motion -> original WorldParticle -> PhysCoRe MPM.

Supports the bird 2D pilot and converted real or augmented PhysCoRe episodes.
Neither route uses physical-property labels.
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
from torch.utils.checkpoint import checkpoint

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party/PhysCoRe"))
from PhysGaussian.gaussian_property_adapter import particle_material_fields, MATERIAL_VOCAB


def motion_losses(prediction, observation):
    """PhysCoRe RfD's mean tracked L2 and observed-depth -> particle Chamfer.

    Keep these small formulas here to avoid importing rfd_runtime's unused
    spconv-based Refiner. The original MPM and WorldParticle are called below.
    """
    ids = observation["tracked_idx"]
    track = (prediction[ids] - observation["coords"][ids]).norm(dim=-1).mean()
    points = observation["depth_points"]
    valid = observation["depth_valid"] & torch.isfinite(points).all(-1)
    points = points[valid]
    if not len(ids) or not len(points):
        raise ValueError("Each target frame needs valid tracks and observed point-cloud samples")
    depth = torch.cdist(points, prediction).min(-1).values.mean()
    return track, depth


def training_window(record, frame_count, epoch, length):
    """Advance one frame per epoch, without wrapping to an already used window."""
    # ponytail: stride-one windows overlap; convert more source frames when
    # more unique windows are needed instead of recycling these windows.
    start = int(record.get("start_frame", 0)) + epoch - 1
    return dict(record, start_frame=start) if start + length <= frame_count else None


def visible_frames(indices):
    """Union all camera views at each frame; real recordings already have T x M."""
    if indices.ndim == 3:
        indices = indices.permute(1, 0, 2).reshape(indices.shape[1], -1)
    if indices.ndim != 2:
        raise ValueError("Visible particle indices must be T x M or V x T x M")
    return indices


def load_observed_episode(record, args, device):
    """Read observed coordinates and known controls, never material/state labels."""
    from omegaconf import OmegaConf
    path = Path(record["episode"])
    raw = torch.load(path / "episode_data.pt", map_location="cpu", weights_only=False)
    config = OmegaConf.load(path / "config.yaml")
    source_dataset = str(raw.get("source_dataset", config.get("source_dataset", "unknown")))
    synthetic = source_dataset == "phystwin_augmented"
    if not raw.get("is_real_world", False):
        raise ValueError(f"{path}: requires a PhysCoRe RGB-D-layout episode")
    if raw["shape_completion"]["particle_mode"] != "completed":
        raise ValueError("Use completed mode: depth_only trajectories are interpolated, not measured tracks")
    if config.observation.get("coordinate_system") != "physcore_z_up" or config.observation.get("depth_units") != "meters":
        raise ValueError("Convert the source observations to metric PhysCoRe z-up coordinates")
    start = int(record.get("start_frame", 0))
    end = start + args.prefix_frames + args.target_frames
    coords = raw["particle_coords"][:end].float().to(device)
    if start < 0 or len(coords) != end or args.prefix_frames < 2 or args.target_frames < 1:
        raise ValueError(f"{path}: need >=2 prefix frames followed by target frames")
    x0 = coords[0]
    if not torch.isfinite(x0).all():
        raise ValueError(f"{path}: initial reconstruction has invalid particles")
    n, tracked = len(x0), int(raw["tracked_particle_count"])
    if tracked <= 0 or tracked > n:
        raise ValueError(f"{path}: missing real persistent-track block")
    frame_dt = float(raw["frame_dt"])
    ground = float(raw["ground_height"])
    # Only translate the measured scene; retain metres and the original scale.
    shift = x0.new_tensor([.5, .5, .1])
    shift[:2] -= x0[:, :2].mean(0)
    shift[2] -= ground
    coords = coords + shift
    x0 = coords[0]
    if (x0 < .05).any() or (x0 > .95).any():
        raise ValueError(f"{path}: metric reconstruction does not fit the 1 m MPM domain")
    motion_valid = raw["particle_motion_valid"][:end, :tracked].to(device).bool()
    model_x = coords[start]
    if not torch.isfinite(model_x).all():
        raise ValueError(f"{path}: window {start} has invalid observed geometry")
    prefix = coords[start:start+args.prefix_frames, :tracked]
    prefix_valid = motion_valid[start:start+args.prefix_frames] & torch.isfinite(prefix).all(-1)
    anchors = prefix_valid[0] & prefix_valid[1]
    anchor_ids = anchors.nonzero().flatten()
    if len(anchor_ids) < 8:
        raise ValueError(f"{path}: fewer than 8 valid prefix tracks")
    nearest = torch.cdist(model_x, prefix[0, anchor_ids]).argmin(-1)
    nearest_ids = anchor_ids[nearest]
    model_v = (prefix[1, nearest_ids] - prefix[0, nearest_ids]) / frame_dt
    velocity = model_v
    if start:
        # Retain the frame-0 physical initial state; later observations only
        # condition the material predictor, never reset MPM's x/v/F/C.
        initial_valid = motion_valid[:2] & torch.isfinite(coords[:2, :tracked]).all(-1)
        initial_ids = initial_valid.all(0).nonzero().flatten()
        if len(initial_ids) < 8:
            raise ValueError(f"{path}: fewer than 8 valid initial velocity tracks")
        initial_nearest = initial_ids[torch.cdist(x0, coords[0, initial_ids]).argmin(-1)]
        velocity = (coords[1, initial_nearest] - coords[0, initial_nearest]) / frame_dt
    controller_key = "controller_grid_points" if "controller_grid_points" in raw else "controller_points"
    controller = raw[controller_key][:end].float().to(device) + shift
    if controller.ndim != 3 or not controller.shape[1] or not torch.isfinite(controller).all():
        raise ValueError(f"{path}: missing finite measured controller trajectories")
    features = []
    for frame in range(1, args.prefix_frames):
        mask = prefix_valid[frame, nearest_ids]
        displacement = torch.where(mask[:, None], prefix[frame, nearest_ids] - prefix[0, nearest_ids], 0.)
        # Reuse PhysCoRe's distance-weighted known-controller features.
        distance = torch.cdist(model_x, controller[start+frame]).clamp_min(1e-6)
        weight = torch.exp(-distance / .04)
        weight_sum = weight.sum(-1, keepdim=True).clamp_min(1e-6)
        control_vec = (weight[..., None] * (controller[start+frame][None] - model_x[:, None])).sum(1) / weight_sum
        control_disp = (weight[..., None] * (controller[start+frame] - controller[start])[None]).sum(1) / weight_sum
        features.append(torch.cat((displacement / .03, mask[:, None].float(),
                                   control_vec, control_disp / .03,
                                   weight_sum / controller.shape[1]), -1))
    properties_path = Path(record["prior"])
    records = {int(r["object_id"]): r for r in json.loads(properties_path.read_text())}
    object_id = int(record.get("object_id", 0))
    prior, audit = particle_material_fields(records, torch.full((n,), object_id, device=device))
    density = float(prior["density"][0])
    # Conservative explicit elastic-wave CFL bound; units remain metres/seconds.
    E, nu = float(prior["E"].max()), float(prior["nu"].max())
    wave_speed = math.sqrt(E * (1 - nu) / ((1 + nu) * (1 - 2 * nu) * density))
    cfl_limit = .35 / (32 * wave_speed)
    steps = math.ceil(frame_dt / min(args.dt, cfl_limit))
    dt = frame_dt / steps
    pitch = float(raw["shape_completion"]["particle_voxel_size"])
    if pitch <= 0:
        raise ValueError(f"{path}: missing frame-0 particle rest-volume scale")
    physics = dict(dt=dt, n_grid=32, volume=pitch**3, density=density,
        damping=1., gravity=[0., 0., -9.81], boundary_conditions=[
            dict(point=[0., 0., .1], normal=[0., 0., 1.], surface="coulomb", friction=.5)])
    constants = record.get("physics", {})
    if set(constants) - {"volume", "density", "damping", "gravity"}:
        raise ValueError(f"{path}: unsupported physics constants")
    physics.update(constants)
    if (not math.isclose(float(physics["density"]), density, rel_tol=1e-6)
            or not math.isfinite(float(physics["volume"])) or float(physics["volume"]) <= 0
            or not 0 < float(physics["damping"]) <= 1
            or len(physics["gravity"]) != 3 or not all(math.isfinite(float(v)) for v in physics["gravity"])):
        raise ValueError(f"{path}: require positive rest volume, GP density, valid damping/gravity")
    rollout = dict(clip_bound=.5, ground_surface="coulomb", ground_friction=.5,
        ground_elasticity=0., use_explicit_rigid_contact=False,
        elasticity_model="CorotatedElasticity", plasticity_model="IdentityPlasticity",
        manipulation_controller_grid_contact_radius=.04,
        manipulation_controller_grid_velocity_blend=1.,
        corotated_rotation_backward="exact", corotated_volume_j_backward="svd")
    official = OmegaConf.load(ROOT / "third_party/PhysCoRe/configs/train_RfD.yaml")
    official_case = record.get("official_case", record["case"])
    override = official.get("per_sample_rollout_config", {}).get(official_case, {})
    rollout.update(OmegaConf.to_container(override, resolve=True) if OmegaConf.is_config(override) else override)
    rollout.update(record.get("rollout", {}))
    visible = visible_frames(raw["tracked_visible_particle_indices"])[:end].to(device)
    obs = raw["observation_data"]
    observations = {}
    for frame in range(start+args.prefix_frames, end):
        ids = visible[frame]
        ids = ids[(ids >= 0) & (ids < tracked)].unique()
        ids = ids[motion_valid[frame, ids] & torch.isfinite(coords[frame, ids]).all(-1)]
        points = obs["object_points_clean"][:, frame].float().to(device) + shift
        valid = obs["object_valid_mask"][:, frame].to(device).bool()
        valid &= torch.isfinite(points).all(-1)
        if len(ids) < 8 or valid.sum() < 8:
            raise ValueError(f"{record['case']} frame {frame}: insufficient observed tracks/depth")
        observations[frame] = dict(coords=coords[frame], tracked_idx=ids,
                                    depth_points=points, depth_valid=valid)
    state = dict(x=x0[None], v=velocity[None],
                 F=torch.eye(3, device=device).expand(1, n, 3, 3).clone(),
                 C=torch.zeros((1, n, 3, 3), device=device))
    return dict(case=record["case"], split=record["split"], state=state,
        record=dict(record), frame_count=len(raw["particle_coords"]), window_start=start,
        model_x=model_x, model_v=model_v,
        motion=torch.cat(features, -1), prior=prior, observations=observations,
        physics=physics, rollout=rollout, controller=controller,
        frame_dt=frame_dt, end=end,
        metadata=dict(case=record["case"], official_case=official_case, split=record["split"], episode=str(path),
            episode_sha256=hashlib.sha256((path / "episode_data.pt").read_bytes()).hexdigest(),
            source_dataset=source_dataset, is_synthetic=synthetic, controller_source=controller_key,
            observation_source=raw.get("depth_render_source", "measured_rgbd"),
            prior_generation=record.get("prior_generation", {}),
            prior_path=str(properties_path), prior_sha256=hashlib.sha256(properties_path.read_bytes()).hexdigest(),
            prior=audit, particles=n, real_track_count=tracked, start_frame=start,
            source_frames=len(raw["particle_coords"]), warmup_frames=start,
            prefix_frames=list(range(start, start+args.prefix_frames)),
            target_frames=list(observations), valid_target_tracks={f: len(o["tracked_idx"]) for f, o in observations.items()},
            valid_target_depth_points={f: int(o["depth_valid"].sum()) for f, o in observations.items()},
            units="metres, seconds, pascals, kg/m^3", shift_metres=shift.tolist(),
            frame_dt=frame_dt, dt=dt, frame_steps=steps, wave_speed=wave_speed,
            prior_cfl_dt_limit=cfl_limit, simulated_seconds=(end - 1)*frame_dt,
            particle_rest_volume=physics["volume"], rollout=rollout,
            shape_completion=dict(raw["shape_completion"])))


STATE_KEYS = ("x", "v", "F", "C")


def material_rollout_block(engine, x, v, F, C, log_E, nu, controller, indicator):
    """Original MPM, returning owned final states instead of trajectory views."""
    from train_relation_full import physcore_advance
    _, state = physcore_advance(engine, dict(x=x, v=v, F=F, C=C),
        dict(log_E=log_E, nu=nu), controller.shape[1]-1,
        controller_grid_points=controller, manipulation_indicator=indicator)
    return tuple(state[key].clone() for key in STATE_KEYS)


def advance_material_frame(advance, state, material, controller, indicator, args, training):
    """Checkpoint blocks retain the entire frame graph; tail mode is the baseline."""
    full = args.grad_mode == "full" and training
    steps_per_frame = controller.shape[1]-1
    if full:
        block = args.checkpoint_steps or steps_per_frame
        schedule = [(min(block, steps_per_frame-start), training)
                    for start in range(0, steps_per_frame, block)]
    else:
        tail = min(args.bptt, steps_per_frame) if training else 0
        schedule = [(steps_per_frame-tail, False), (tail, True)]
    done = 0
    for steps, active in schedule:
        if not steps:
            continue
        tensors = (*state, material["log_E"], material["nu"],
                   controller[:, done:done+steps+1], indicator)
        with torch.set_grad_enabled(active):
            if full and active and args.checkpoint == "reentrant" and args.checkpoint_steps:
                state = checkpoint(advance, *tensors, use_reentrant=True)
            else:
                state = advance(*tensors)
        done += steps
    return state


def episode_forward(model, case, args, training=False, *, material_override=None,
                    fixed_frame_steps=None):
    """Observed window: full cross-frame gradients or the explicit tail baseline.

    Diagnostics can supply material tensors and freeze the integer time grid;
    no target, constitutive or controller settings change between perturbations.
    """
    from omegaconf import OmegaConf
    from train_relation_full import make_physcore_engine
    model.train(training)
    device = next(model.parameters()).device
    # ponytail: cache small observation windows on CPU and stage one scene at a
    # time; use a dataset/DataLoader only when these windows no longer fit RAM.
    def on_device(value):
        if torch.is_tensor(value):
            return value.to(device)
        if isinstance(value, dict):
            return {key: on_device(item) for key, item in value.items()}
        return value
    state = on_device(case["state"])
    prior = on_device(case["prior"])
    x, v = state["x"][0], state["v"][0]
    model.backbone.apply_radius_override(radius_scale=1.5,
        particle_radius=case["physics"]["volume"]**(1/3)/2)
    with torch.set_grad_enabled(training):
        material = material_override if material_override is not None else model(
            case["model_x"].to(device), case["model_v"].to(device), case["motion"].to(device), prior,
            case["boundary"].to(device), case["normals"].to(device))
    E, nu = material["log_E"].detach().exp(), material["nu"].detach()
    speed = (E * (1 - nu) / ((1 + nu) * (1 - 2*nu) * prior["density"])).sqrt().max()
    # E/nu change during training: choose a fresh wave-CFL-safe integer number
    # of steps, while preserving the exact observed frame interval.
    safe_steps = math.ceil(case["frame_dt"] / min(args.dt, .35 / (32 * float(speed))))
    frame_steps = safe_steps if fixed_frame_steps is None else int(fixed_frame_steps)
    if frame_steps < safe_steps:
        raise ValueError(f"{case['case']}: fixed time grid violates the current material CFL bound")
    physics = dict(case["physics"], dt=case["frame_dt"] / frame_steps)
    engine = make_physcore_engine(dict(physcore_adapter=physics), OmegaConf.create(case["rollout"]),
                                  device, material_only=True)
    engine.enable_corrector(False)
    # Only known measured controller trajectories enter MPM; object targets
    # stay exclusively in the motion loss below.
    from physcore.particle_flow.episode_runtime import _piecewise_catmull_rom_window
    controller = _piecewise_catmull_rom_window(case["controller"].to(device), 0, case["end"] - 1, frame_steps)
    tracks, depths = [], []
    # Engine is fixed for the window. All changing controls and material tensors
    # are explicit checkpoint inputs, so backward never captures a later frame.
    def advance(*tensors):
        return material_rollout_block(engine, *tensors)
    state = tuple(state[key] for key in STATE_KEYS)
    for frame in range(1, case["end"]):
        start = (frame-1)*frame_steps
        state = advance_material_frame(advance, state, material,
            controller[start:start+frame_steps+1][None], x.new_ones(1), args,
            training and (args.grad_mode == "full" or frame in case["observations"]))
        with torch.set_grad_enabled(training):
            if frame in case["observations"]:
                track, depth = motion_losses(state[0][0], on_device(case["observations"][frame]))
                tracks.append(track)
                depths.append(depth)
        if args.grad_mode == "tail":
            state = tuple(value.detach() for value in state)
        print(f"case={case['case']} frame={frame}/{case['end']-1} train={int(training)}", flush=True)
    with torch.set_grad_enabled(training):
        track, depth = torch.stack(tracks).mean(), torch.stack(depths).mean()
        prior_loss = ((material["log_E"][0]-prior["E"].log())/math.log(10.)).square().mean()
        prior_loss = prior_loss + ((material["nu"][0]-prior["nu"])/.05).square().mean()
        motion_loss = args.w_track*track + args.w_depth*depth
        loss = motion_loss + args.w_prior*prior_loss
    if not torch.isfinite(loss):
        raise RuntimeError(f"{case['case']}: non-finite motion loss")
    return loss, dict(loss=float(loss.detach()), motion_loss=float(motion_loss.detach()), tracked_m=float(track.detach()),
        chamfer_m=float(depth.detach()), prior_loss=float(prior_loss.detach()),
        E_mean=float(E.mean()), nu_mean=float(nu.mean()), dt=physics["dt"], frame_steps=frame_steps,
        window_start=case["window_start"], window_end=case["end"]-1,
        gradient_horizon_steps=((case["end"]-1)*frame_steps if args.grad_mode == "full" else min(args.bptt, frame_steps)))


def sample_fingerprint(case):
    """Hash the actual training inputs/targets, ignoring names and file metadata."""
    def encoded(value):
        if torch.is_tensor(value):
            value = value.detach().cpu().contiguous()
            return dict(dtype=str(value.dtype), shape=list(value.shape),
                        tensor_sha256=hashlib.sha256(value.numpy().tobytes()).hexdigest())
        if isinstance(value, dict):
            return {key: encoded(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [encoded(item) for item in value]
        return value
    inputs = {key: case[key] for key in ("state", "motion", "rollout", "controller", "frame_dt")}
    inputs.update({key: case[key] for key in ("model_x", "model_v") if key in case})
    inputs["prior"] = {key: case["prior"][key] for key in ("E", "nu", "density")}
    inputs["physics"] = {key: value for key, value in case["physics"].items() if key != "dt"}
    targets = {frame: dict(tracked_idx=obs["tracked_idx"], coords=obs["coords"][obs["tracked_idx"]],
        depth_points=obs["depth_points"][obs["depth_valid"] & torch.isfinite(obs["depth_points"]).all(-1)])
        for frame, obs in case["observations"].items()}
    return hashlib.sha256(json.dumps(encoded(dict(inputs=inputs, targets=targets)),
                                    sort_keys=True, allow_nan=False).encode()).hexdigest()


def episode_handle(case):
    """Retain only tiny static data between streamed training updates."""
    return {key: case[key] for key in ("case", "split", "record", "frame_count", "window_start",
            "metadata", "boundary", "normals", "motion_dim")}


def train_episodes(args, device):
    from omegaconf import OmegaConf
    from PhysGaussian.worldparticle_grid_correction import MaterialTransformer
    from train_relation_full import make_physcore_corrector
    records = json.loads(args.episode_manifest.read_text())
    if not records or any(r["split"] not in ("train", "val") for r in records):
        raise ValueError("Require nonempty observed episodes with train/val source splits")
    if not args.train_only and set(r["split"] for r in records) != {"train", "val"}:
        raise ValueError("Require independent train and val cases")
    if args.w_track < 0 or args.w_depth < 0 or args.w_prior < 0 or args.w_track+args.w_depth <= 0:
        raise ValueError("Require nonnegative loss weights and positive motion supervision")
    cfg = OmegaConf.create(dict(model=dict(kind="worldparticle",
        source_root=str(ROOT / "third_party/WorldParticle"), width=32,
        merge_iterations=3, decoder_layers=2), correction=dict(max_delta_v=.002)))
    wrapper = make_physcore_corrector(cfg, device)
    provenance = wrapper.provenance()
    cases, duplicates, unique, boundaries = [], [], {}, {}
    for record in records:
        case = load_observed_episode(record, args, torch.device("cpu"))
        if args.train_only:
            digest = sample_fingerprint(case)
            if digest in unique:
                duplicates.append(dict(case=case["case"], duplicate_of=unique[digest], sample_sha256=digest))
                continue
            unique[digest] = case["case"]
            case["metadata"].update(source_split=case["split"], sample_sha256=digest)
            case["split"] = case["metadata"]["split"] = "train"
        key = json.dumps((case["physics"]["n_grid"], case["physics"]["boundary_conditions"]), sort_keys=True)
        if key not in boundaries:
            boundary, normals = wrapper._boundaries(case["physics"], case["state"]["x"][0].to(device))
            boundaries[key] = boundary.cpu(), normals.cpu()
        case["boundary"], case["normals"] = boundaries[key]
        case["motion_dim"] = case["motion"].shape[-1]
        cases.append(episode_handle(case) if args.stream_episodes else case)
    names = [case["case"] for case in cases]
    if len(set(names)) != len(names):
        raise ValueError("Distinct samples must have distinct case names")
    model = MaterialTransformer(wrapper.backbone, cases[0]["motion_dim"], args.material_range).to(device)
    provenance.update(
        pipeline="Original CConv -> k-partite merging -> RoPE -> SuperParticleDecoder -> adapted 2-channel material head",
        physical_features=f"x0/v0 geometry plus {cases[0]['motion_dim']} observed-prefix motion/controller features and log(E/2000), nu, log(rho/1000), linearly projected to 32 channels; no F/C labels or material embedding",
        correction="Two raw channels refine log(E) and nu around GaussianProperty priors; density fixed; output_scale=1; no dynamics or P2G residual",
        predictor="MaterialTransformer.compute_correction feeds original PhysCoRe MPM; " +
            ("motion losses differentiate through every historical and target frame using reentrant activation checkpoint blocks"
             if args.grad_mode == "full" and args.checkpoint == "reentrant" and args.checkpoint_steps else
             "motion losses differentiate through every historical and target frame without activation checkpointing"
             if args.grad_mode == "full" else "motion losses differentiate through each target frame's final BPTT tail"))
    del wrapper
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.)
    checkpoint_path = args.resume_checkpoint or args.evaluate_checkpoint
    start_epoch = updates = 0
    resume = dict(mode="fresh", optimizer_reset=False, rng_restored=False)
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        saved_range = checkpoint.get("metadata", {}).get("args", {}).get("material_range", "prior")
        if saved_range != args.material_range:
            raise ValueError("Checkpoint material range must match --material-range")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        previous = checkpoint.get("metadata", {})
        start_epoch = int(checkpoint.get("epoch", previous.get("completed_epochs", 0)))
        updates = int(checkpoint.get("updates", previous.get("actual_updates", 0)))
        resume = dict(path=str(checkpoint_path), sha256=hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
            mode="evaluation", start_epoch=start_epoch, previous_updates=updates,
            optimizer_reset=False, rng_restored=False)
        if args.resume_checkpoint:
            if start_epoch < 1 or updates < 1:
                raise ValueError("Resume checkpoint must record completed epoch/update counts")
            if start_epoch >= args.epochs:
                raise ValueError("--epochs is the total epoch count and must exceed the resumed epoch")
            resume["optimizer_reset"] = args.reset_optimizer or "optimizer_state_dict" not in checkpoint
            resume["mode"] = "warm_start" if resume["optimizer_reset"] else "resume"
            resume["optimizer_reset_reason"] = ("explicit --reset-optimizer" if args.reset_optimizer else
                "checkpoint has no optimizer state" if resume["optimizer_reset"] else None)
            if not resume["optimizer_reset"]:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                for group in optimizer.param_groups:
                    group["lr"] = args.lr
            if "torch_rng_state" in checkpoint:
                torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
                if "cuda_rng_state_all" in checkpoint:
                    torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["cuda_rng_state_all"]])
                resume["rng_restored"] = True
            print(json.dumps(dict(checkpoint_loaded=resume)), flush=True)
    original_head = model.backbone.output_network[-1].weight.detach().clone()
    started = time.monotonic()
    initial_updates = updates
    decay_updates = max(1, (args.epochs-start_epoch) * sum(c["split"] == "train" for c in cases))
    core_gradient_seen = False
    rows = []
    metadata = dict(args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        cases=[case["metadata"] for case in cases], provenance=provenance, resume=resume,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        physics_source_sha256={name: hashlib.sha256((ROOT / "third_party/PhysCoRe" / name).read_bytes()).hexdigest()
            for name in ("physcore/particle_flow/rollout.py", "physcore/mpm/mpm_model.py", "physcore/mpm/warp/svd.py")},
        loss_source_sha256=hashlib.sha256((ROOT / "third_party/PhysCoRe/physcore/particle_flow/rfd_runtime.py").read_bytes()).hexdigest(),
        runtime=dict(python=sys.executable, torch=torch.__version__, pythonpath=os.environ.get("PYTHONPATH"),
            cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES")),
        supervision="Observed 3D persistent tracks + one-way observation-to-particle Chamfer; no material labels or material-coded colors; synthetic provenance recorded per case",
        limitations=[("All source splits used for training, no evaluation"
                      if args.train_only else "One short observed window per case, held out by independent case"),
            "Training advances the window by one frame each epoch; exhausted windows are never recycled",
            "Initial interior geometry and rest volume come from frame-0 shape completion",
            "F=I/C=0 at frame 0; prefix tracks estimate initial velocity",
            "Nearest observed surface history is propagated to interior particles",
            "Known hand/controller paths drive grid contact; contact radius is configured, not learned",
            ("Continuous x/v/F/C gradients from frame 0 through every target; checkpoint blocks only control memory"
             if args.grad_mode == "full" else "Only the last BPTT chunk of each target frame carries gradients"),
            ("Density fixed to GP prior; augmented E range [100, GP prior], nu range [0.05,0.45]"
             if args.material_range == "augmented" else
             "Density fixed to GP prior; E limited to 0.1x..10x prior and [1e4,1e7], nu delta limited to 0.05"),
            "Dynamics residual is disabled; no material or internal state labels are accessed"],
        metrics=("Training forwards only; optimization.csv and TensorBoard contain pre-update losses. No baseline, validation or train-set evaluation. sample_sha256 identifies actual inputs and targets."
                 if args.train_only else
                 "Equal-case mean of target-frame mean Euclidean metre errors; baseline and epoch evaluations use identical windows. motion_loss excludes the prior penalty; loss includes it. history.csv contains evaluations at fixed epoch weights; optimization.csv contains pre-update training passes. gradient_horizon_steps records the configured maximum training-state dependency span (full: frame 0 to final target; tail: each target's independent final tail)."),
        best_metric=None if args.train_only else "motion_loss",
        checkpoint_files=(dict(last="material_last.pt") if args.train_only else
                          dict(last="material_last.pt", best_validation="material_best.pt")),
        duplicates_removed=duplicates, unique_samples=len(cases),
        data_loading="one episode at a time" if args.stream_episodes else "CPU window cache",
        training_window_policy=dict(length=args.prefix_frames+args.target_frames,
            stride=1, order="start_frame + epoch - 1", exhaustion="stop without wrapping"),
        sample_policy="one training pass per content fingerprint" if args.no_repeat else "one new consecutive window per case per epoch",
        initialization="random backbone with zero material correction head" if not checkpoint_path else "checkpoint",
        baseline_epoch=None if args.train_only else start_epoch,
        baseline_note="No baseline/evaluation forward passes" if args.train_only else
                      "Re-evaluated loaded weights on this manifest; not comparable to old smaller-split losses")
    (args.output / "run.json").write_text(json.dumps(metadata, indent=2))
    fields = ["epoch", "split", "case", "loss", "motion_loss", "tracked_m", "chamfer_m", "prior_loss",
              "E_mean", "nu_mean", "dt", "frame_steps", "gradient_horizon_steps", "updates", "head_grad_norm", "core_grad_norm", "elapsed_seconds", "sample_sha256"]
    fields += ["window_start", "window_end", "learning_rate"]
    used_samples = set()
    best_val = None if args.train_only else math.inf
    best_epoch = None if args.train_only else start_epoch
    tensorboard = None
    if args.tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        tensorboard = SummaryWriter(str(args.tensorboard), max_queue=1, flush_secs=1)
        tensorboard.add_text("configuration", json.dumps(metadata["args"], indent=2), 0)
        tensorboard.flush()
    def save_checkpoint(filename, epoch):
        # Atomic replacement keeps the previous usable checkpoint if writing fails.
        target = args.output / filename
        temporary = target.with_suffix(".tmp")
        torch.save(dict(model_state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(),
            torch_rng_state=torch.get_rng_state(), cuda_rng_state_all=torch.cuda.get_rng_state_all(),
            epoch=epoch, updates=updates, best_val_loss=best_val, best_epoch=best_epoch,
            metadata=metadata), temporary)
        os.replace(temporary, target)
    with (args.output / "history.csv").open("x", newline="") as handle, \
         (args.output / "optimization.csv").open("x", newline="") as training_handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        training_writer = csv.DictWriter(training_handle, fieldnames=fields)
        writer.writeheader()
        training_writer.writeheader()
        epochs = [start_epoch] if args.evaluate_checkpoint else range(start_epoch + int(args.train_only), args.epochs + 1)
        for epoch in epochs:
            head_grad = core_grad = 0.
            if epoch > start_epoch:
                epoch_updates = updates
                for case in cases:
                    if case["split"] != "train":
                        continue
                    window = training_window(case["record"], case["frame_count"], epoch,
                                             args.prefix_frames+args.target_frames)
                    if window is None:
                        continue
                    if args.stream_episodes or window["start_frame"] != case["window_start"]:
                        sampled = load_observed_episode(window, args, torch.device("cpu"))
                        sampled.update(boundary=case["boundary"], normals=case["normals"], split=case["split"])
                        sampled["metadata"].update(split=case["split"],
                            source_split=case["metadata"].get("source_split", case["split"]))
                        case = sampled
                    digest = sample_fingerprint(case)
                    if args.no_repeat and digest in used_samples:
                        continue
                    used_samples.add(digest)
                    case["metadata"]["sample_sha256"] = digest
                    optimizer.zero_grad(set_to_none=True)
                    loss, metrics = episode_forward(model, case, args, training=True)
                    loss.backward()
                    head = model.backbone.output_network[-1]
                    if any(p.grad is None for p in head.parameters()):
                        raise RuntimeError(f"{case['case']}: material head is disconnected from the loss")
                    head_grad = math.sqrt(sum(float(p.grad.detach().double().square().sum())
                                              for p in head.parameters()))
                    core_grad = math.sqrt(sum(float(p.grad.detach().double().square().sum())
                        for name, p in model.backbone.named_parameters()
                        if not name.startswith("output_network.") and p.grad is not None))
                    # ponytail: a fully controlled scene can have zero material
                    # sensitivity; verify nonzero gradient flow across the run.
                    if not math.isfinite(head_grad):
                        raise RuntimeError(f"{case['case']}: non-finite material-head gradient")
                    core_gradient_seen |= core_grad > 0 and math.isfinite(core_grad)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), .5, error_if_nonfinite=True)
                    # PhysCoRe MfM uses linear decay; use updates for single-epoch runs.
                    if args.final_lr is not None:
                        progress = min(1., (updates-initial_updates)/max(1, decay_updates-1))
                        for group in optimizer.param_groups:
                            group["lr"] = args.lr + (args.final_lr-args.lr)*progress
                    optimizer.step()
                    updates += 1
                    training_row = dict(epoch=epoch, split="train", case=case["case"], updates=updates,
                        learning_rate=optimizer.param_groups[0]["lr"],
                        head_grad_norm=head_grad, core_grad_norm=core_grad,
                        sample_sha256=case["metadata"].get("sample_sha256", ""),
                        elapsed_seconds=time.monotonic()-started, **metrics)
                    training_writer.writerow(training_row)
                    training_handle.flush()
                    if tensorboard:
                        for key in ("loss", "motion_loss", "tracked_m", "chamfer_m", "prior_loss"):
                            tensorboard.add_scalar("train/"+key, metrics[key], updates)
                        tensorboard.add_scalar("grad/head_norm", head_grad, updates)
                        tensorboard.add_scalar("grad/core_norm", core_grad, updates)
                        tensorboard.add_scalar("optimizer/learning_rate", optimizer.param_groups[0]["lr"], updates)
                        tensorboard.flush()
                    print(json.dumps(dict(phase="optimization", **training_row)), flush=True)
                if updates == epoch_updates:
                    if updates == initial_updates:
                        raise ValueError("No unused video windows remain at this epoch")
                    print("Unique video windows exhausted; stopping without reuse", flush=True)
                    break
            validation = []
            for case in ([] if args.train_only else cases):
                if case["split"] == "train" and epoch not in (start_epoch, args.epochs) and \
                        (epoch - start_epoch) % args.train_eval_every:
                    continue
                _, metrics = episode_forward(model, case, args)
                row = dict(epoch=epoch, split=case["split"], case=case["case"], updates=updates,
                    head_grad_norm=head_grad, core_grad_norm=core_grad,
                    elapsed_seconds=time.monotonic()-started, **metrics)
                rows.append(row)
                writer.writerow(row)
                handle.flush()
                if case["split"] == "val":
                    validation.append(metrics["motion_loss"])
                print(json.dumps(dict(phase="evaluation", **row)), flush=True)
            val_mean = sum(validation) / len(validation) if validation else None
            improved = val_mean is not None and val_mean < best_val
            if improved:
                best_val, best_epoch = val_mean, epoch
            metadata.update(actual_updates=updates, new_updates=updates-initial_updates,
                new_unique_training_samples=len(used_samples),
                original_core_gradient_seen=core_gradient_seen, completed_epochs=epoch,
                best_val_loss=best_val, best_epoch=best_epoch)
            (args.output / "run.json").write_text(json.dumps(metadata, indent=2))
            if not args.evaluate_checkpoint:
                save_checkpoint("material_last.pt", epoch)
                if improved:
                    save_checkpoint("material_best.pt", epoch)
    if tensorboard:
        tensorboard.close()
    if not args.evaluate_checkpoint:
        if torch.equal(original_head, model.backbone.output_network[-1].weight) or not core_gradient_seen:
            raise RuntimeError("Original WorldParticle core did not receive nonzero gradients/update")
    print("COMPLETE", flush=True)


def self_check():
    """CPU check of the shared frame loop; real MPM checks are separate."""
    record = dict(case="example", start_frame=0)
    windows = [training_window(record, 12, epoch, 6) for epoch in range(1, 9)]
    assert [window["start_frame"] for window in windows[:-1]] == list(range(7))
    assert len({tuple(range(window["start_frame"], window["start_frame"]+6))
                for window in windows[:-1]}) == 7
    assert windows[-1] is None, "Exhausted windows must never wrap to frame 0"
    assert training_window(record, 12, 4, 6) == windows[3], "Resume must keep the epoch/window mapping"
    assert training_window(dict(start_frame=2), 12, 1, 6)["start_frame"] == 2
    assert record["start_frame"] == 0, "Selection must not mutate the base record"

    def advance(x, v, F, C, log_E, nu, controls, indicator):
        dt = .01
        for step in range(controls.shape[1]-1):
            # All four state channels affect the future; each must retain graph.
            C = C + dt*(log_E + controls[:, step])
            F = F + dt*C
            v = v + dt*(log_E + nu*F)
            x = x + dt*v
        return x.clone(), v.clone(), F.clone(), C.clone()

    def run(mode, implementation, training=True):
        args = argparse.Namespace(grad_mode=mode, checkpoint=implementation,
                                  checkpoint_steps=2, bptt=2)
        E = torch.tensor(.8, dtype=torch.float64, requires_grad=True)
        nu = torch.tensor(.3, dtype=torch.float64, requires_grad=True)
        state = tuple(torch.zeros(1, dtype=torch.float64) for _ in STATE_KEYS)
        targets = []
        for frame in range(3):
            controls = torch.arange(frame*5, frame*5+6, dtype=torch.float64)[None]/100
            state = advance_material_frame(advance, state, dict(log_E=E, nu=nu),
                controls, E.new_ones(1), args, training and (mode == "full" or frame > 0))
            if frame > 0:
                targets.append(state[0].square().sum())
            if mode == "tail":
                state = tuple(value.detach() for value in state)
        loss = torch.stack(targets).mean()
        gradients = None
        if training:
            loss.backward()
            gradients = torch.stack((E.grad, nu.grad))
        return torch.cat(state), loss.detach(), gradients

    direct, direct_loss, direct_grad = run("full", "off")
    saved, saved_loss, saved_grad = run("full", "reentrant")
    tail, tail_loss, tail_grad = run("tail", "off")
    evaluated, evaluation_loss, _ = run("full", "reentrant", training=False)
    assert torch.allclose(direct, saved, atol=1e-12, rtol=1e-12)
    assert torch.allclose(direct_loss, saved_loss, atol=1e-12, rtol=1e-12)
    assert torch.allclose(direct_grad, saved_grad, atol=1e-12, rtol=1e-12)
    assert torch.allclose(direct, tail) and torch.allclose(direct_loss, tail_loss)
    assert torch.allclose(direct, evaluated) and torch.allclose(direct_loss, evaluation_loss)
    assert not evaluated.requires_grad
    assert torch.isfinite(saved_grad).all() and (saved_grad.abs() > 0).all()
    assert saved_grad[0] > tail_grad[0]*2, "Cross-frame graph was truncated"
    observation = dict(tracked_idx=torch.tensor([0]), coords=torch.zeros(1, 3),
                       depth_points=torch.ones(1, 3), depth_valid=torch.tensor([True]))
    sample = dict(state=dict(x=torch.zeros(1, 1, 3)), motion=torch.ones(1, 2),
        prior={key: torch.ones(1) for key in ("E", "nu", "density")}, physics=dict(dt=.01),
        rollout={}, controller=torch.zeros(2, 1, 3), frame_dt=.1, observations={1: observation})
    fingerprint = sample_fingerprint(sample)
    assert fingerprint == sample_fingerprint(dict(sample, case="renamed", split="val", metadata={})), "Names must not evade deduplication"
    changed = dict(observation, coords=torch.ones(1, 3))
    assert fingerprint != sample_fingerprint(dict(sample, observations={1: changed})), "Different targets must stay distinct"
    assert sample_fingerprint(dict(sample, model_x=torch.zeros(1, 3))) != sample_fingerprint(
        dict(sample, model_x=torch.ones(1, 3))), "Observed window geometry must participate in deduplication"
    view_ids = torch.arange(3*6*2).reshape(3, 6, 2)
    merged = visible_frames(view_ids)
    assert merged.shape == (6, 6) and torch.equal(merged[4], view_ids[:, 4].reshape(-1))
    assert visible_frames(merged) is merged, "Real T x M visibility must stay unchanged"
    static = dict(case="sample", split="train", record={}, frame_count=12, window_start=0,
                  metadata={}, boundary=torch.zeros(1, 3), normals=torch.ones(1, 3), motion_dim=33)
    handle = episode_handle(dict(static, state=sample["state"], observations=sample["observations"]))
    assert set(handle) == set(static) and "state" not in handle and "observations" not in handle
    assert handle["metadata"] is static["metadata"], "Streamed provenance must stay attached"
    from PhysGaussian.worldparticle_grid_correction import refine_material
    prior = dict(E=torch.tensor([2e6, 5e5], dtype=torch.float64),
                 nu=torch.tensor([.45, .4], dtype=torch.float64))
    raw = torch.zeros(2, 2, dtype=torch.float64, requires_grad=True)
    original = refine_material(raw, prior)
    wide = refine_material(raw, prior, "augmented")
    assert torch.equal(wide["log_E"][0], prior["E"].log()) and torch.equal(wide["nu"][0], prior["nu"])
    assert torch.equal(original["log_E"], wide["log_E"]) and torch.equal(original["nu"], wide["nu"])
    (wide["log_E"].sum()+wide["nu"].sum()).backward()
    assert torch.isfinite(raw.grad).all() and (raw.grad.abs() > 0).all(), "Zero head must retain material gradients"
    target_E = torch.tensor([200., 25000.], dtype=torch.float64)
    target_nu = torch.tensor([.1, .2], dtype=torch.float64)
    encoded = torch.stack((torch.atanh((target_E.log()-prior["E"].log())/(prior["E"].log()-math.log(100.))),
                           torch.atanh((target_nu-prior["nu"])/.4)), -1)
    decoded = refine_material(encoded, prior, "augmented")
    assert torch.allclose(decoded["log_E"][0].exp(), target_E) and torch.allclose(decoded["nu"][0], target_nu)
    assert (refine_material(torch.ones_like(raw)*10, prior, "augmented")["log_E"][0] <= prior["E"].log()).all()
    print(json.dumps(dict(self_check="PASS", full_gradient=direct_grad.tolist(),
        checkpoint_gradient=saved_grad.tolist(), tail_gradient=tail_grad.tolist(),
        deduplication="PASS", unique_windows="PASS", streaming="PASS", material_range="PASS", multi_view_visibility="PASS",
        scope="Pure Torch frame-loop only; does not replace actual MPM gradient/memory checks")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--asset", type=Path)
    source.add_argument("--episode-manifest", type=Path,
                        help="JSON list of {case, episode, prior, split}; split is train or val")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-check", action="store_true", help="Pure Torch CPU check of cross-frame checkpoint gradients")
    parser.add_argument("--epochs", type=int, default=5, help="Total completed epochs, including resumed epochs")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--final-lr", type=float, help="Linearly decay LR per update to this value over new epochs")
    parser.add_argument("--dt", type=float, default=5e-5)
    parser.add_argument("--bptt", type=int, default=40)
    parser.add_argument("--grad-mode", choices=("full", "tail"),
                        help="Observed episodes: full cross-frame gradient or historical final-tail baseline")
    parser.add_argument("--checkpoint-steps", type=int, default=64,
                        help="Full-mode recomputation block length; 0 keeps each whole frame graph directly")
    parser.add_argument("--checkpoint", choices=("reentrant", "off"), default="reentrant",
                        help="Full-mode activation checkpoint implementation; off is for direct-gradient checks")
    parser.add_argument("--prefix-frames", type=int, default=4)
    parser.add_argument("--target-frames", type=int, default=2)
    parser.add_argument("--w-track", type=float, default=1.)
    parser.add_argument("--w-depth", type=float, default=1.)
    parser.add_argument("--w-prior", type=float, default=1e-4)
    parser.add_argument("--material-range", choices=("prior", "augmented"), default="prior",
                        help="augmented allows E down to 100 Pa up to the GP prior, and full nu [0.05,0.45]")
    parser.add_argument("--stream-episodes", action="store_true",
                        help="Train-only: load one episode per update instead of caching all CPU windows")
    parser.add_argument("--train-only", action="store_true", help="Train on all source splits without baseline or evaluation passes")
    parser.add_argument("--no-repeat", action="store_true", help="Use each content-unique video window once; requires fresh train-only mode")
    parser.add_argument("--tensorboard", type=Path, help="Write live training scalars to this TensorBoard log directory")
    checkpoints = parser.add_mutually_exclusive_group()
    checkpoints.add_argument("--evaluate-checkpoint", type=Path, help="Evaluate saved weights without optimizer updates")
    checkpoints.add_argument("--resume-checkpoint", type=Path, help="Resume weights/counters; legacy checkpoints without optimizer state explicitly warm-start AdamW")
    parser.add_argument("--reset-optimizer", action="store_true", help="Warm-start resumed weights with a fresh AdamW for controlled comparisons")
    parser.add_argument("--train-eval-every", type=int, default=1,
                        help="Evaluate train cases every N new epochs; validation is always evaluated each epoch")
    args = parser.parse_args()
    if args.final_lr is not None and not 0 <= args.final_lr <= args.lr:
        parser.error("--final-lr must lie between zero and --lr")
    if args.self_check:
        return self_check()
    args.grad_mode = args.grad_mode or ("full" if args.episode_manifest else "tail")
    if not args.output or not (args.asset or args.episode_manifest):
        parser.error("Require --output and either --asset or --episode-manifest")
    if args.epochs < 1 or args.dt <= 0 or args.bptt < 1 or args.lr <= 0 or args.train_eval_every < 1:
        parser.error("Require positive training settings")
    if args.checkpoint_steps < 0:
        parser.error("--checkpoint-steps must be nonnegative")
    if args.resume_checkpoint and not args.episode_manifest:
        parser.error("--resume-checkpoint requires the observed-episode route")
    if args.reset_optimizer and not args.resume_checkpoint:
        parser.error("--reset-optimizer requires --resume-checkpoint")
    if args.stream_episodes and (not args.train_only or not args.episode_manifest):
        parser.error("--stream-episodes requires observed --train-only")
    if args.material_range != "prior" and not args.episode_manifest:
        parser.error("--material-range augmented requires observed episodes")
    if args.no_repeat and (not args.train_only or args.resume_checkpoint or args.evaluate_checkpoint):
        parser.error("--no-repeat requires fresh --train-only")
    if (args.train_only or args.tensorboard) and not args.episode_manifest:
        parser.error("Train-only mode and TensorBoard require the observed-episode route")
    if args.train_only and args.evaluate_checkpoint:
        parser.error("--train-only cannot evaluate a checkpoint")
    if args.asset and args.grad_mode == "full":
        parser.error("The bird pilot retains its historical tail workflow; pass --grad-mode tail")
    args.output.mkdir(parents=True, exist_ok=False)
    import warp as wp
    wp.config.kernel_cache_dir = str(args.output / "cache/warp")
    wp.init()
    torch.manual_seed(7)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    if args.episode_manifest:
        return train_episodes(args, device)
    import cv2
    import numpy as np
    from omegaconf import OmegaConf
    from PhysGaussian.worldparticle_grid_correction import MaterialTransformer
    from train_relation_full import make_physcore_corrector, make_physcore_engine, physcore_advance
    work = args.asset
    data = np.load(work / "bird_physical_particles.npz")
    camera = json.loads((work / "source_camera_check.json").read_text())
    fx, cx = camera["camera"]["signed_fx_cx"]
    fy, cy = camera["camera"]["signed_fy_cy"]
    scale = float(data["source_units_to_meters"])
    xyz = torch.as_tensor(data["positions"], device=device, dtype=torch.float32)
    n = len(xyz)
    # The raw camera coordinates and metric coordinates differ by a uniform scale.
    def camera_project(x):
        return torch.stack((fx * x[:, 0] / x[:, 2] + cx,
                            fy * x[:, 1] / x[:, 2] + cy), -1)
    uv0 = camera_project(xyz).cpu().numpy()
    cap = cv2.VideoCapture(str(work / "source_video.mkv"))
    gray = []
    for frame in range(7):
        ok, image = cap.read()
        if not ok:
            raise ValueError(f"Missing real video frame {frame}")
        gray.append(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    cap.release()
    mask = cv2.imread(str(work / "first_frame_fitted_alpha.png"), cv2.IMREAD_GRAYSCALE)
    if mask is None or mask.shape != gray[0].shape:
        raise ValueError("Initial reconstruction mask does not match the video")
    mask = cv2.erode((mask > 128).astype(np.uint8) * 255, np.ones((5, 5), np.uint8))
    points = cv2.goodFeaturesToTrack(gray[0], 200, .02, 5, mask=mask)
    if points is None:
        raise ValueError("No visible object features")
    tracks = [points[:, 0].copy()]
    valid = [np.ones(len(points), dtype=bool)]
    for frame in range(1, 7):
        nxt, status, _ = cv2.calcOpticalFlowPyrLK(gray[frame-1], gray[frame], points, None)
        back, backward_status, _ = cv2.calcOpticalFlowPyrLK(gray[frame], gray[frame-1], nxt, None)
        good = (status[:, 0] != 0) & (backward_status[:, 0] != 0)
        good &= np.linalg.norm(back[:, 0] - points[:, 0], axis=1) < 1.
        good &= np.isfinite(nxt[:, 0]).all(1)
        valid.append(valid[-1] & good)
        tracks.append(nxt[:, 0].copy())
        points = nxt
    tracks, valid = np.stack(tracks), np.stack(valid)
    distances = np.linalg.norm(tracks[0, :, None] - uv0[None], axis=-1)
    # ponytail: nearest projected particle is an approximate surface binding;
    # occlusion-aware rasterized bindings are the next upgrade.
    binding = distances.argmin(1)
    keep = distances.min(1) < 3.
    keep &= valid[4]
    if keep.sum() < 8 or any((keep & valid[f]).sum() < 8 for f in (5, 6)):
        raise ValueError("Too few consistent real-video tracks")
    all_ids = np.flatnonzero(keep)
    binding = binding[keep]
    if len(np.unique(binding)) != len(binding):
        _, unique = np.unique(binding, return_index=True)
        all_ids, binding = all_ids[unique], binding[unique]
    ids = torch.tensor(binding, device=device)
    target_uv = torch.tensor(uv0[binding][None] + tracks[:, all_ids] - tracks[0, all_ids],
                             device=device, dtype=torch.float32)
    up = torch.tensor(data["floor_normal"], device=device, dtype=torch.float32)
    up = up / up.norm()
    axis_x = torch.linalg.cross(torch.tensor([0., 0., 1.], device=device), up)
    axis_x = axis_x / axis_x.norm()
    rotation = torch.stack((axis_x, torch.linalg.cross(up, axis_x), up))
    floor = torch.tensor(data["floor_point"], device=device, dtype=torch.float32)
    origin = (xyz @ rotation.T).mean(0)
    shift = torch.tensor([.5, .5, .12], device=device) - origin
    shift[2] = .12 - (floor @ rotation.T)[2]
    def project(x):
        return camera_project((x - shift) @ rotation)
    gravity_camera = -9.81 * up
    velocity0 = torch.tensor(data["velocity"], device=device, dtype=torch.float32)
    time4 = .2
    current = xyz + time4 * velocity0 + .5 * time4**2 * gravity_camera
    velocity = velocity0 + time4 * gravity_camera
    # Fit release position/velocity from the observed prefix only; depth is held
    # to the existing calibrated ballistic estimate, not read from future frames.
    desired = target_uv[4] - camera_project(current)[ids]
    shift_uv = desired.median(0).values
    depth = current[ids, 2].median()
    current[:, 0] += shift_uv[0] * depth / fx
    current[:, 1] += shift_uv[1] * depth / fy
    screen_v = ((target_uv[4] - target_uv[3]) / .05).median(0).values
    u = camera_project(current)[ids].median(0).values
    vz = velocity[:, 2].median()
    velocity[:, 0] = (screen_v[0] * depth + (u[0] - cx) * vz) / fx
    velocity[:, 1] = (screen_v[1] * depth + (u[1] - cy) * vz) / fy
    x = current @ rotation.T + shift
    state0 = dict(x=x[None], v=(velocity @ rotation.T)[None],
                  F=torch.eye(3, device=device).expand(1, n, 3, 3).clone(),
                  C=torch.zeros((1, n, 3, 3), device=device))
    records = {int(r["object_id"]): r for r in json.loads((work / "object_properties.json").read_text())}
    asset = json.loads((work / "physical_asset_check.json").read_text())
    prior, audits = particle_material_fields(records, torch.full((n,), asset["object_id"], device=device))
    frame_steps = math.ceil(.05 / args.dt)
    dt = .05 / frame_steps
    volume = float(data["particle_pitch"]) ** 3
    physics = dict(dt=dt, n_grid=32, volume=volume, density=float(prior["density"][0]),
                   damping=1., gravity=[0., 0., -9.81],
                   boundary_conditions=[dict(point=[0., 0., .12], normal=[0., 0., 1.],
                                             surface="coulomb", friction=.3)])
    episode = dict(physcore_adapter=physics)
    rollout = OmegaConf.create(dict(clip_bound=.5, ground_surface="coulomb",
        ground_friction=.3, ground_elasticity=0., use_explicit_rigid_contact=False,
        elasticity_model="CorotatedElasticity", plasticity_model="IdentityPlasticity"))
    engine = make_physcore_engine(episode, rollout, device, material_only=True)
    engine.enable_corrector(False)
    cfg = OmegaConf.create(dict(model=dict(kind="worldparticle",
        source_root=str(ROOT / "third_party/WorldParticle"), width=32,
        merge_iterations=3, decoder_layers=2), correction=dict(max_delta_v=.002)))
    wrapper = make_physcore_corrector(cfg, device)
    boundary, normals = wrapper._boundaries(physics, x)
    provenance = wrapper.provenance()
    wrapper.backbone.apply_radius_override(radius_scale=1.5, particle_radius=volume**(1/3)/2)
    # Every particle receives nearby visible history plus its visibility mask.
    nearest = torch.cdist(camera_project(xyz), target_uv[0]).argmin(1)
    motion = torch.cat([torch.cat(((target_uv[f] - target_uv[0])[nearest] / 64.,
                        torch.tensor(valid[f, all_ids], device=device)[nearest, None].float()), -1)
                        for f in range(1, 5)], -1)
    model = MaterialTransformer(wrapper.backbone, motion.shape[1]).to(device)
    del wrapper
    if args.evaluate_checkpoint:
        model.load_state_dict(torch.load(args.evaluate_checkpoint, map_location=device, weights_only=False)["model_state_dict"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.)
    original_head = model.backbone.output_network[-1].weight.detach().clone()
    start = time.monotonic()
    rows = []
    output_csv = args.output / "history.csv"
    with output_csv.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "loss", "motion_px", "prior_loss",
            "head_grad_norm", "E_mean", "nu_mean", "elapsed_seconds"])
        writer.writeheader()
        for epoch in range(1 if args.evaluate_checkpoint else args.epochs + 1):
            model.train(epoch > 0)
            optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(epoch > 0):
                material = model(x, state0["v"][0], motion, prior, boundary, normals)
            state = {k: v.clone() for k, v in state0.items()}
            losses = []
            for frame in (5, 6):
                done = 0
                while done < frame_steps:
                    steps = min(args.bptt, frame_steps - done)
                    tail = done + steps == frame_steps
                    active = epoch > 0 and tail
                    with torch.set_grad_enabled(active):
                        mats = material if active else {k: v.detach() for k, v in material.items()}
                        _, state = physcore_advance(engine, state, mats, steps)
                        if tail:
                            prediction = project(state["x"][0])
                            coords = prediction.new_zeros(prediction.shape)
                            coords[ids] = target_uv[frame]
                            visible = torch.tensor(valid[frame, all_ids], device=device)
                            # Same mean Euclidean distance as PhysCoRe tracked_l2,
                            # measured here in pixels after the camera projection.
                            losses.append((prediction[ids[visible]] - coords[ids[visible]]).norm(dim=-1).mean())
                    state = {k: v.detach() for k, v in state.items()}
                    done += steps
                print(f"epoch={epoch} observed_frame={frame} loss_px={float(losses[-1].detach()):.6g}", flush=True)
            motion_loss = torch.stack(losses).mean()
            prior_loss = ((material["log_E"][0] - prior["E"].log()) / math.log(10.)).square().mean()
            prior_loss += ((material["nu"][0] - prior["nu"]) / .05).square().mean()
            loss = motion_loss + 1e-4 * prior_loss
            head_grad = 0.
            if epoch:
                loss.backward()
                head_grad = float(model.backbone.output_network[-1].weight.grad.norm())
                if not math.isfinite(head_grad) or head_grad <= 0:
                    raise RuntimeError("Material head has no finite nonzero MPM/video gradient")
                torch.nn.utils.clip_grad_norm_(model.parameters(), .5, error_if_nonfinite=True)
                optimizer.step()
            row = dict(epoch=epoch, loss=float(loss.detach()), motion_px=float(motion_loss.detach()),
                prior_loss=float(prior_loss.detach()), head_grad_norm=head_grad,
                E_mean=float(material["log_E"].detach().exp().mean()),
                nu_mean=float(material["nu"].detach().mean()), elapsed_seconds=time.monotonic()-start)
            rows.append(row); writer.writerow(row); handle.flush()
            print(json.dumps(row), flush=True)
    if not args.evaluate_checkpoint and torch.equal(original_head, model.backbone.output_network[-1].weight):
        raise RuntimeError("No material weight update")
    metadata = dict(args=vars(args) | {"asset": str(work), "output": str(args.output),
        "evaluate_checkpoint": str(args.evaluate_checkpoint) if args.evaluate_checkpoint else None},
        runtime=dict(python=sys.executable, torch=torch.__version__,
            pythonpath=os.environ.get("PYTHONPATH"), cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES")),
        prior=audits, tracks=len(ids), frames=[5, 6], calibration_frames=[0, 1, 2, 3, 4],
        particles=n, frame_dt=.05, dt=dt, steps_per_frame=frame_steps, bptt=args.bptt,
        supervision="real-video 2D LK tracks, mean Euclidean pixel error; no material GT",
        limitations=["Single-sequence calibration, no held-out evaluation",
            "Reconstruction, camera, ballistic depth and F=I at frame 4 are approximate",
            "Nearest projected particle bindings are not occlusion-aware",
            "Only last BPTT chunk per observed frame receives gradient",
            "Density fixed to GaussianProperty prior; dynamics residual disabled",
            "Epoch loss measured before that epoch's update"],
        provenance=provenance, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if not args.evaluate_checkpoint:
        torch.save(dict(model_state_dict=model.state_dict(), metadata=metadata), args.output / "material.pt")
    (args.output / "run.json").write_text(json.dumps(metadata, indent=2))
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    main()

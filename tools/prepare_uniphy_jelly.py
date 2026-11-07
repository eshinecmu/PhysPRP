"""Convert original UniPhy elastic states to post-MPM or PhysCoRe episodes.

Run with the existing project environment; checks the actual saved substep states.
Dense demo exports retain their original physical parameters and substep phase.
"""

import argparse
import copy
import io
import math
from pathlib import Path
import sys
import zipfile

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_relation_full import MATERIAL_VOCAB, _validate_trajectory


def to_physcore_episode(data):
    """Map complete numerical GT into PhysCoRe's dataset contract without cameras.

    The backend remains PhysCoRe's own MPM; source discretization is provenance,
    not a request to replace the official solver's transfer or mass division.
    """
    frames = _validate_trajectory(data)
    config = data["mpm_config"]
    n = data["x"].shape[1]
    if float(config["grid_lim"]) != 1. or float(config["rpic_damping"]) != 0.:
        raise ValueError("PhysCoRe adaptation requires a unit grid and APIC (rpic_damping=0)")
    if not 0. <= float(config["grid_v_damping_scale"]) <= 1.:
        raise ValueError("PhysCoRe grid damping must be in [0,1]")
    if data["selection"].any() or torch.unique(data["object_id"]).numel() != 1 or data.get("object_relations"):
        raise ValueError("PhysCoRe adaptation requires one fully simulated object without object relations")
    for key in ("volume", "density", "mass"):
        if not torch.equal(data[key], data[key][0].expand_as(data[key])):
            raise ValueError(f"PhysCoRe's scalar particle {key} requires uniform {key}")
    source_boundaries = config["boundary_conditions"]
    for bc in source_boundaries:
        if (bc["type"] != "surface_collider" or float(bc.get("radius", 0.)) != 0.
                or float(bc["start_time"]) > float(data["time"][0])
                or float(bc["end_time"]) <= float(data["time"][-1])):
            raise ValueError("PhysCoRe adaptation requires explicit infinite surfaces active for the entire trajectory")
        if bc["surface"] == "separate" and float(bc["friction"]) != 0.:
            raise ValueError("Only frictionless separation maps exactly to the upstream grid-only Coulomb operation")
    rotation = data["x"].new_tensor([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
    translation = data["x"].new_tensor([0., 1., 0.])
    coords = (data["x"] @ rotation.T + translation).contiguous()
    boundaries = copy.deepcopy(source_boundaries)
    for bc in boundaries:
        bc["point"] = (rotation @ rotation.new_tensor(bc["point"]) + translation).tolist()
        bc["normal"] = (rotation @ rotation.new_tensor(bc["normal"])).tolist()
    dt = float(config["substep_dt"])
    source = copy.deepcopy(data.get("source", {}))
    if not isinstance(source, dict):
        raise ValueError("Trajectory source metadata must be a dictionary")
    particle_ids = data["particle_id"] if data["particle_id"].ndim == 1 else data["particle_id"][0]
    # ponytail: these are complete numerical tracks, not fabricated camera/surface observations.
    indices = torch.arange(n, device=coords.device).expand(frames, -1).clone()
    episode = dict(
        particle_coords=coords,
        particle_velocities=(data["v"] @ rotation.T).contiguous(),
        particle_deformation_gradients=(rotation @ data["F_trial"] @ rotation.T).contiguous(),
        particle_affine_velocity_matrices=(rotation @ data["C"] @ rotation.T).contiguous(),
        particle_material_params={"log_E": data["E"].log(), "nu": data["nu"].clone()},
        tracked_visible_particle_indices=indices,
        visible_particle_indices=indices.clone(),
        rigid_body_coords=coords.new_empty((frames, 0, 3)),
        frame_dt=dt, sim_dt=dt, steps_per_frame=1,
        source_dataset="uniphy", is_real_world=False,
        manipulation_flag=0., is_manipulation=False, interaction_start_frame=0,
        source_time=data["time"].clone(), particle_id=particle_ids.clone(),
        object_id=data["object_id"].clone(), material_id=data["material_id"].clone(),
        physcore_adapter=dict(
            schema="physprp-uniphy-physcore-v1", n_grid=config["n_grid"], grid_lim=1., dt=dt,
            volume=float(data["volume"][0]), density=float(data["density"][0]),
            mass=float(data["mass"][0]), damping=float(config["grid_v_damping_scale"]),
            gravity=(rotation @ rotation.new_tensor(config["g"])).tolist(),
            boundary_conditions=boundaries, rotation=rotation.tolist(), translation=translation.tolist(),
            source_mpm_config=copy.deepcopy(config), source=source,
            source_transfer_scheme=config.get("transfer_scheme", "unspecified"),
            source_grid_mass_epsilon=config.get("grid_mass_epsilon", source.get("grid_mass_epsilon")),
            observation_kind="complete_numerical_particle_tracks",
        ),
    )
    if source.get("elasticity") == "NeoHookeanElasticity":
        episode["physcore_adapter"]["elasticity_model"] = "NeoHookeanElasticity"
    return episode


def convert(archive):
    with zipfile.ZipFile(archive) as source:
        config = yaml.safe_load(source.read("jelly/hydra.yaml"))
        blob, sim = config["env"]["blob"], config["sim"]
        shape, material = blob["shape"], blob["material"]
        if (set(config["env"]) != {"blob"} or shape["type"] != "cube"
                or shape["mode"] != "uniform" or material["elasticity"]["cls"] != "CorotatedElasticity"
                or material["plasticity"]["cls"] != "IdentityPlasticity"
                or sim["skip_frame"] != 1 or sim["bc"] != "freeslip"):
            raise ValueError("Expected the original single elastic cube with every substep saved")
        names = [f"jelly/state/{i:04d}.pt" for i in range(sim["num_steps"] + 1)]
        saved = {name for name in source.namelist() if name.startswith("jelly/state/") and name.endswith(".pt")}
        if saved != set(names):
            raise ValueError("Missing or unexpected state files")
        states = {key: [] for key in ("x", "v", "C", "F_trial")}
        n = None
        for name in names:
            frame = torch.load(io.BytesIO(source.read(name)), map_location="cpu", weights_only=True)
            n = len(frame["x"]) if n is None else n
            if list(frame["sections"]) != [n]:
                raise ValueError(f"Particle groups changed in {name}")
            for key in states:
                value = frame["F" if key == "F_trial" else key]
                expected = (n, 3, 3) if key in ("C", "F_trial") else (n, 3)
                if value.dtype != torch.float32 or value.shape != expected or not torch.isfinite(value).all():
                    raise ValueError(f"Invalid {key} in {name}")
                states[key].append(value.contiguous())
    data = {key: torch.stack(values) for key, values in states.items()}
    dt, grid = float(sim["dt"]), int(sim["num_grids"])
    size = torch.tensor(shape["size"], dtype=torch.float32)
    resolution = torch.round(size * int(shape["resolution"]) / size.max()).long()
    if int(resolution.prod()) != n or blob["span"] != [0, sim["num_steps"]]:
        raise ValueError("Unexpected sampler count or changing particle activation")
    torch.testing.assert_close(data["x"][0].amin(0), torch.tensor(shape["center"]) - size / 2)
    torch.testing.assert_close(data["x"][0].amax(0), torch.tensor(shape["center"]) + size / 2)
    # NCLaw updates F with the same new C as its x/v endpoint; identity plasticity
    # makes this F exactly the next step's trial deformation, not a force label.
    x_error = (data["x"][1:] - data["x"][:-1] - dt * data["v"][1:]).abs().max().item()
    f_error = (data["F_trial"][1:] - (torch.eye(3) + dt * data["C"][1:]) @ data["F_trial"][:-1]).abs().max().item()
    if x_error > 2e-6 or f_error > 2e-5:
        raise ValueError(f"Substep recurrence mismatch: x={x_error}, F={f_error}")
    volume = math.prod(shape["size"]) / n
    data.update(volume=torch.full((n,), volume), mass=torch.full((n,), volume * float(blob["rho"])),
                density=torch.full((n,), float(blob["rho"])),
                E=torch.full((n,), float(material["elasticity"]["E"])),
                nu=torch.full((n,), float(material["elasticity"]["nu"])),
                particle_id=torch.arange(n), object_id=torch.zeros(n, dtype=torch.int64),
                material_id=torch.full((n,), MATERIAL_VOCAB.index("unknown"), dtype=torch.int64),
                selection=torch.zeros(n, dtype=torch.int32), material_vocab=list(MATERIAL_VOCAB),
                time=torch.arange(len(names), dtype=torch.float64) * dt)
    # Match the source's grid-index tests (<bound, >grid-bound) with planes
    # halfway between nodes. The generic bounding_box uses a different upper test.
    boundaries = []
    for axis in range(3):
        for coordinate, sign in (((sim["bound"] - .5) / grid, 1.),
                                 ((grid - sim["bound"] + .5) / grid, -1.)):
            point, normal = [0., 0., 0.], [0., 0., 0.]
            point[axis], normal[axis] = coordinate, sign
            boundaries.append(dict(type="surface_collider", point=point, normal=normal,
                                   surface="separate", friction=0., start_time=0.,
                                   end_time=float(data["time"][-1]) + dt))
    data["mpm_config"] = dict(material="jelly", n_grid=grid, grid_lim=1., g=sim["gravity"],
                              rpic_damping=0., grid_v_damping_scale=1., substep_dt=dt,
                              transfer_scheme="mls", grid_mass_epsilon=float(sim["eps"]),
                              boundary_conditions=boundaries)
    data["episode_id"] = "uniphy_nclaw_jelly"
    data["source"] = dict(dataset="UniPhy original NCLaw jelly", archive=Path(archive).name,
                          deformation_mapping="F_trial[t]=F[t]; identity plasticity",
                          material_label="unknown; numerical material properties from hydra.yaml",
                          grid_mass_epsilon=float(sim["eps"]),
                          checks=dict(x_recurrence_max_abs=x_error, F_recurrence_max_abs=f_error))
    _validate_trajectory(data)
    print(f"validated frames={len(names)} particles={n} x_error={x_error:.8g} F_error={f_error:.8g}")
    print(f"position_range={float(data['x'].min()):.8g}..{float(data['x'].max()):.8g}; volume={volume}; mass={volume * blob['rho']}")
    return data


def convert_demo(directory):
    """Read a complete, independent UniPhy demo; never fill missing internal states."""
    root = Path(directory)
    config = yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8"))
    if root.name not in ("Elastic_Toy", "Elastic_Blobby") or len(config["objects"]) != 1:
        raise ValueError("Expected a verified single-object UniPhy elastic demo")
    material = next(iter(config["objects"].values()))["material"]
    if material["object_material"] != "elastic" or material["material_type"] != "MPMSimulator.elasticity":
        raise ValueError("Only the original elastic demo constitutive branch is supported")
    dt, frame_dt = 1. / float(material["inv_dt"]), 1. / float(material["inv_frame_dt"])
    substeps = round(frame_dt / dt)
    if not math.isclose(substeps * dt, frame_dt, rel_tol=1e-7):
        raise ValueError("Demo video intervals must contain an integer number of saved substeps")
    frames = int(config["visualization_cfg"]["num_frames"]) * substeps
    raw = {key: torch.load(root / name, map_location="cpu", weights_only=True) for key, name in
           (("x", "GtX.pt"), ("v", "GtV.pt"), ("C", "GtC.pt"), ("F", "GtF.pt"), ("Ftmp", "GtFtmp.pt"))}
    n = raw["x"].shape[1]
    for key, value in raw.items():
        shape = (frames + 1, n, 3, 3) if key in ("F", "Ftmp", "C") else (frames + 1, n, 3)
        if value.dtype != torch.float32 or tuple(value.shape) != shape or not torch.isfinite(value).all():
            raise ValueError(f"Unexpected original demo state {key}")
    if any(torch.count_nonzero(raw[key][frames:]) for key in ("x", "v", "C")):
        raise ValueError("The expected final x/v/C padding is not zero; recheck saved time indexing")
    if torch.count_nonzero(raw["F"][0]) or not torch.equal(raw["F"], raw["Ftmp"]):
        raise ValueError("Expected the verified unwritten F[0] and identity-plasticity F/Ftmp mapping")
    # Source writes x/v/C at i, F at i+1; clone also removes oversized torch.save view storage.
    data = {key: raw[key][:frames].clone() for key in ("x", "v", "C")}
    data["F_trial"] = raw["F"][1:frames + 1].clone()
    torch.testing.assert_close(data["F_trial"][0], torch.eye(3).expand(n, 3, 3), rtol=0, atol=0)
    torch.testing.assert_close(data["v"][0], torch.tensor(material["velocity"]).expand(n, 3), rtol=0, atol=1e-7)
    x_error = float((data["x"][1:] - data["x"][:-1] - dt * data["v"][1:]).abs().max())
    f_error = float((data["F_trial"][1:] - (torch.eye(3) + dt * data["C"][1:]) @ data["F_trial"][:-1]).abs().max())
    if x_error > 2e-6 or f_error > 2e-5:
        raise ValueError(f"Demo substep phase mismatch: x={x_error}, F={f_error}")
    ground = config["simulator_cfg"]["BC"]
    if set(ground) != {"ground"} or ground["ground"] != [[0, 0, 0], [0, 1, 0]]:
        raise ValueError("Only the original sticky ground plane has been verified")
    # One common integer-grid translation preserves metres, velocities, material units and grid phase.
    shift = torch.tensor([.20, .04, .20])
    data["x"] += shift
    dx, volume = .02, (.02 * .5) ** 3  # Original MPMWrapper/simulator constants, not an AABB volume estimate.
    mu, lam, density = float(material["mu"]), float(material["lam"]), float(material["rho"])
    E, nu = mu * (3 * lam + 2 * mu) / (lam + mu), lam / (2 * (lam + mu))
    data.update(volume=torch.full((n,), volume), mass=torch.full((n,), volume * density),
                density=torch.full((n,), density), E=torch.full((n,), E), nu=torch.full((n,), nu),
                particle_id=torch.arange(n), object_id=torch.zeros(n, dtype=torch.int64),
                material_id=torch.full((n,), MATERIAL_VOCAB.index("unknown"), dtype=torch.int64),
                selection=torch.zeros(n, dtype=torch.int32), material_vocab=list(MATERIAL_VOCAB),
                time=torch.arange(frames, dtype=torch.float64) * dt, episode_id="uniphy_demo_" + root.name)
    # Source uses distance <=1e-6, including the y=0 grid row; the native plane uses distance <0.
    point = [0., float(shift[1]) + 1e-6, 0.]
    data["mpm_config"] = dict(material="jelly", n_grid=round(1. / dx), grid_lim=1.,
        g=config["simulator_cfg"]["gravity"], rpic_damping=0., grid_v_damping_scale=1.,
        substep_dt=dt, transfer_scheme="mls", grid_mass_epsilon=0., boundary_conditions=[dict(
            type="surface_collider", point=point, normal=[0., 1., 0.], surface="sticky", friction=0.,
            start_time=0., end_time=frames * dt)])
    data["source"] = dict(dataset="UniPhy CVPR2025 original gradio demo", episode_id=data["episode_id"],
        config=config, is_measured=False, physical_parameter_source="original config.yaml mu/lam/rho",
        coordinate_translation_m=shift.tolist(), coordinate_scaling=1.,
        deformation_mapping="F_trial[t]=GtF[t+1]; verified against C[t+1] recurrence",
        discarded_zero_padding_states=1, source_video_interval=frame_dt,
        source_sticky_condition="signed_distance <= 1e-6", source_particle_volume=volume,
        elasticity="NeoHookeanElasticity", plasticity="IdentityPlasticity",
        source_constitutive_branch="NeoHookeanElasticity + identity plasticity + MLS",
        checks=dict(x_recurrence_max_abs=x_error, F_recurrence_max_abs=f_error))
    _validate_trajectory(data)
    print(f"validated {data['episode_id']}: states={frames}, particles={n}, E={E:.8g}, nu={nu:.8g}, "
          f"x_phase_error={x_error:.8g}, F_phase_error={f_error:.8g}")
    return data


def convert_pac_probe(input_dir, device):
    """Replay the historical compression validation scene, exporting real PAC x/v/F/C."""
    import hashlib
    import importlib
    import types

    import numpy as np
    import taichi as ti

    root = Path(input_dir)
    scene_path = root if root.is_file() else root / "controlled_scene.npz"
    pac_root = Path(__file__).resolve().parents[2] / "PAC-NeRF"
    if not (pac_root / "lib/engine/dynamic_observer.py").is_file():
        raise FileNotFoundError(f"Original PAC-NeRF engine required at {pac_root}")
    device = torch.device(device)
    if device.type != "cuda" or device.index not in (None, 0):
        raise ValueError("Use cuda:0 with CUDA_VISIBLE_DEVICES selecting the PAC export GPU")
    with np.load(scene_path, allow_pickle=False) as scene:
        initial = torch.from_numpy(scene["initial"].copy()).to(device)
        velocity = torch.from_numpy(scene["velocity"].copy()).to(device)
        target = torch.from_numpy(scene["target"].copy())
        times = torch.from_numpy(scene["times"].copy())
        rho, nu, E = (float(scene[key]) for key in ("rho", "nu", "true_E"))
    if (initial.shape != (216, 3) or initial.dtype != torch.float64 or velocity.shape != initial.shape
            or target.shape != (9, 216, 3) or not torch.isfinite(initial).all()
            or not torch.isfinite(velocity).all() or not torch.isfinite(target).all()
            or (rho, nu, E) != (1000., .3, 2000.)):
        raise ValueError("Expected the original 216-particle PAC compression validation fixture")
    torch.testing.assert_close(times, torch.arange(9, dtype=torch.float64) * .01)
    torch.testing.assert_close(initial.cpu(), target[0], atol=2e-7, rtol=0.)
    package = types.ModuleType("physprp_pac")
    package.__path__ = [str(pac_root / "lib")]
    sys.modules.setdefault(package.__name__, package)
    engine = importlib.import_module("physprp_pac.engine")
    ti.init(arch=ti.cuda, default_fp=ti.f64, fast_math=False, device_memory_GB=1., offline_cache=False)
    # ponytail: this fixed historical fixture has dt=.001, dx=.05 and no contacts; no new teacher is invented.
    dt, dx, steps = .001, .05, 80
    observer = engine.DynamicObserver(dtype="float64", rgbnet_dim=0, dt=dt, frame_dt=dt,
        particle_chunk_size=256, cuda_chunk_size=32, gravity=[0., 0., 0.],
        material=engine.MPMSimulator.elasticity, tensor_float64=True)
    n = len(initial)
    observer.initialize(initial, initial.new_zeros((n, 1)), velocity,
        initial.new_full((n,), rho), initial.new_full((n,), E / (2 * (1 + nu))),
        initial.new_full((n,), E * nu / ((1 + nu) * (1 - 2 * nu))), dx)
    simulator = observer.simulator

    @ti.kernel
    def export_state(index: ti.i32, x: ti.types.ndarray(), v: ti.types.ndarray(),
                     F: ti.types.ndarray(), C: ti.types.ndarray()):
        for p in range(n):
            for i in ti.static(range(3)):
                x[p, i] = simulator.x[p, index][i]
                v[p, i] = simulator.v[p, index][i]
                for j in ti.static(range(3)):
                    F[p, i, j] = simulator.F[p, index][i, j]
                    C[p, i, j] = simulator.C[p, index][i, j]

    states = {key: [] for key in ("x", "v", "F_trial", "C")}
    for step in range(steps + 1):
        if step:
            simulator.advance(step - 1)
        if not observer.succeed():
            raise RuntimeError(f"Historical PAC replay failed its CFL check at substep {step}")
        arrays = [np.empty((n, 3), dtype=np.float64) for _ in range(2)]
        arrays += [np.empty((n, 3, 3), dtype=np.float64) for _ in range(2)]
        export_state(step % simulator.cuda_chunk_size, *arrays)
        for key, value in zip(states, arrays):
            states[key].append(torch.from_numpy(value))
    raw = {key: torch.stack(values) for key, values in states.items()}
    reproduction_error = float((raw["x"][::10] - target).abs().max())
    torch.testing.assert_close(raw["x"][::10], target, atol=2e-7, rtol=0.)
    torch.testing.assert_close(raw["x"][1:], raw["x"][:-1] + dt * raw["v"][1:], atol=1e-12, rtol=0.)
    torch.testing.assert_close(raw["F_trial"][1:],
        (torch.eye(3, dtype=torch.float64) + dt * raw["C"][:-1]) @ raw["F_trial"][:-1],
        atol=1e-12, rtol=0.)
    translation = initial.cpu().new_tensor([.5, .5, -1.5])
    torch.testing.assert_close(translation / dx, (translation / dx).round(), atol=1e-12, rtol=0.)
    data = {key: value.float() for key, value in raw.items()}
    # PAC updates F before P2G; expose the stress-input phase used by our baseline state contract.
    stress_F = (torch.eye(3, dtype=torch.float64) + dt * raw["C"]) @ raw["F_trial"]
    torch.testing.assert_close(stress_F[:-1], raw["F_trial"][1:], atol=1e-12, rtol=0.)
    torch.testing.assert_close(stress_F[1:],
        (torch.eye(3, dtype=torch.float64) + dt * raw["C"][1:]) @ stress_F[:-1], atol=1e-12, rtol=0.)
    data["F_trial"] = stress_F.float()
    data["x"] = (raw["x"] + translation).float()
    if not ((data["x"] > .1) & (data["x"] < .9)).all():
        raise ValueError("Translated historical PAC scene must remain away from the unit-box boundaries")
    volume = float(simulator.p_vol[None])
    assert abs(volume - (dx / 2) ** 3) < 1e-15
    data.update(volume=torch.full((n,), volume), mass=torch.full((n,), volume * rho),
        density=torch.full((n,), rho), E=torch.full((n,), E), nu=torch.full((n,), nu),
        particle_id=torch.arange(n), object_id=torch.zeros(n, dtype=torch.int64),
        material_id=torch.full((n,), MATERIAL_VOCAB.index("unknown"), dtype=torch.int64),
        selection=torch.zeros(n, dtype=torch.int32), material_vocab=list(MATERIAL_VOCAB),
        time=torch.arange(steps + 1, dtype=torch.float64) * dt,
        episode_id="pac_material_compression_20260922")
    data["mpm_config"] = dict(material="jelly", n_grid=round(1 / dx), grid_lim=1.,
        g=[0., 0., 0.], rpic_damping=0., grid_v_damping_scale=1., substep_dt=dt,
        transfer_scheme="mls", grid_mass_epsilon=0., boundary_conditions=[dict(
            type="surface_collider", point=[0., 0., 0.], normal=[0., 1., 0.],
            surface="separate", friction=0., start_time=0., end_time=(steps + 1) * dt)])
    sources = [scene_path, pac_root / "lib/engine/mpm_simulator.py", pac_root / "lib/engine/dynamic_observer.py"]
    data["source"] = dict(dataset="historical PAC compression validation; simulated, not measured",
        role="validation", original_scene=str(scene_path), original_dt=dt, original_dx=dx,
        original_frame_dt=.01, original_frames=9, complete_substeps=steps + 1,
        elasticity="NeoHookeanElasticity", plasticity="IdentityPlasticity",
        deformation_mapping="F_trial[t]=(I+dt*C[t])@raw_PAC_F[t]; verified next stress-input phase",
        translation=translation.tolist(), translation_kind="integer grid-cell offset; no scaling",
        floor="inactive adapter plane y=0; original replay has no collider",
        material_ground_truth=True, initial_state_ground_truth=True,
        grid_mass_threshold="1e-8*dx^3; no added epsilon", is_real_world=False,
        checks=dict(old_target_max_abs=reproduction_error),
        source_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources})
    _validate_trajectory(data)
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--format", choices=("post-step", "physcore"), default="post-step")
    parser.add_argument("--pac-probe", action="store_true", help="Re-export the archived independent compression validation scene")
    parser.add_argument("--device", default="cuda:0", help="PAC export GPU, selected with CUDA_VISIBLE_DEVICES")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    data = (convert_pac_probe(args.archive, args.device) if args.pac_probe else
            convert_demo(args.archive) if args.archive.is_dir() else convert(args.archive))
    frames = len(data["x"])
    if args.format == "physcore":
        data = to_physcore_episode(data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as destination:
        torch.save(data, destination)
    print(f"saved {args.output}; format={args.format}; transitions={frames - 1}")


if __name__ == "__main__":
    main()

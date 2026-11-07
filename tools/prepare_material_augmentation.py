"""Quality-filter original PhysCoRe augmentation; never train on material labels."""
import argparse
import contextlib
import copy
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def directory_bytes(path):
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file() and not p.is_symlink())


def remaining(target, records):
    hashes = [r["sample_sha256"] for r in records]
    if len(set(hashes)) != len(hashes):
        raise ValueError("Duplicate accepted observations in augmentation state")
    return max(0, target - len(records))


def next_index(attempts):
    return max((a["index"] for a in attempts), default=-1) + 1


def collect(output):
    records = []
    for path in sorted((output / "state").glob("*.json")):
        records.extend(json.loads(path.read_text())["records"])
    remaining(len(records), records)
    if len({r["case"] for r in records}) != len(records):
        raise ValueError("Augmented case names must be unique")
    save_json(output / "episode_manifest.json", records)
    return records


def visible_frame(visible, frame):
    if visible.ndim == 3:
        return visible[:, frame].reshape(-1)
    if visible.ndim == 2:
        return visible[frame]
    raise ValueError("Visibility must have V,T,M or T,M axes")


def quality(path):
    import torch
    raw = torch.load(path / "episode_data.pt", map_location="cpu", weights_only=False)
    xyz = raw["particle_coords"].float()
    controller = raw["controller_grid_points"].float()
    if xyz.ndim != 3 or xyz.shape[0] < 6 or xyz.shape[-1] != 3:
        raise ValueError("Need >=6 frames of 3D coordinates")
    if controller.shape[0] != xyz.shape[0] or not controller.shape[1]:
        raise ValueError("Controller does not match generated frame count")
    if not torch.isfinite(xyz).all() or not torch.isfinite(controller).all():
        raise ValueError("Nonfinite generated coordinates/controller")
    extent = xyz.amax((0, 1)) - xyz.amin((0, 1))
    motion = (xyz - xyz[:1]).norm(dim=-1)
    if xyz.min() < -.02 or xyz.max() > 1.02 or motion.max() > .45:
        raise ValueError("Rollout escaped the 1 m domain or displacement limit")
    if motion.mean() < 1e-6:
        raise ValueError("No measurable generated motion")
    observed = raw["observation_data"]
    points = observed["object_points_clean"]
    valid = observed["object_valid_mask"].bool()
    if not torch.isfinite(points[valid]).all():
        raise ValueError("Nonfinite valid observed point")
    tracked = int(raw["tracked_particle_count"])
    visible = raw["tracked_visible_particle_indices"]
    motion_valid = raw["particle_motion_valid"].bool()
    for frame in range(4, xyz.shape[0]):
        ids = visible_frame(visible, frame)
        ids = ids[(ids >= 0) & (ids < tracked)].unique()
        if int(motion_valid[frame, ids].sum()) < 8 or int(valid[:, frame].sum()) < 8:
            raise ValueError(f"Insufficient visible tracks/depth at frame {frame}")
    digest = hashlib.sha256()
    # ponytail: hash full coordinates/controls; random observation noise cannot make an identical trajectory unique.
    for tensor in (xyz, controller):
        tensor = tensor.detach().cpu().contiguous()
        digest.update(str((str(tensor.dtype), tuple(tensor.shape))).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest(), dict(frames=len(xyz), particles=xyz.shape[1],
        mean_motion=float(motion.mean()), max_motion=float(motion.max()), extent=extent.tolist())


def prepare_group(group, record, raw):
    import torch
    from PhysGaussian.gaussian_property_adapter import material_parameters
    prior = json.loads(Path(record["prior"]).read_text())
    prior = next(p for p in prior if int(p["object_id"]) == int(record.get("object_id", 0)))
    density = material_parameters(prior)["actual"]["density"]
    group = copy.deepcopy(group)
    group["dataset"]["real_world_domain_center"] = [.5, .5, .5]
    rollout = group["rollout"]
    radius = float(record.get("rollout", {}).get("manipulation_controller_grid_contact_radius", .04))
    rollout.update(rho=density, damping=1., gravity=[0., 0., -9.81], num_grids=32,
        use_explicit_rigid_contact=False, manipulation_controller_grid_contact_radius=radius,
        manipulation_controller_grid_velocity_blend=1., ground_tangent_damping=0.,
        ground_static_velocity_threshold=0.)
    group.setdefault("ground", {}).update(friction=.5, elasticity=0., clearance_cells=0.)
    max_E = math.exp(float(group["material"]["log_E"]["clamp"][1]))
    max_nu = float(group["material"]["nu"]["clamp"][1])
    speed = math.sqrt(max_E * (1-max_nu) / ((1+max_nu)*(1-2*max_nu)*density))
    chunks = int(group["dataset"]["rollout_steps"])
    steps = math.ceil(float(raw["frame_dt"]) / (chunks * min(2e-4, .35/(32*speed))))
    rollout["default_steps_per_frame"] = max(int(rollout.get("default_steps_per_frame", 1)), steps)
    x0 = raw["particle_coords"][0].float()
    half = ((x0.amax(0)-x0.amin(0))/2).clamp_min(1e-6)
    if not torch.isfinite(x0).all() or (half >= .48).any():
        raise ValueError("Source initial shape cannot fit the centered 1 m domain")
    volume = float(half.prod()) / len(x0)
    physics = dict(volume=volume, density=density, damping=1., gravity=[0., 0., -9.81])
    return group, physics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--recipe", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--source", nargs="*")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--budget-gb", type=float, default=100.)
    parser.add_argument("--limit", type=int, help="Maximum newly accepted episodes in this invocation")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--pilot", action="store_true", help="One accepted episode per group from the first source")
    mode.add_argument("--all", action="store_true")
    mode.add_argument("--collect", action="store_true", help="Merge completed per-job state; no simulation")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        import numpy as np
        assert np.array_equal(visible_frame(np.arange(12).reshape(1, 6, 2), 4), [8, 9])
        assert np.array_equal(visible_frame(np.arange(24).reshape(2, 6, 2), 4), [8, 9, 20, 21])
        assert np.array_equal(visible_frame(np.arange(12).reshape(6, 2), 4), [8, 9])
        assert remaining(80, [{"sample_sha256": "a"}]) == 79
        assert next_index([{"index": 0, "status": "rejected"}, {"index": 3, "status": "accepted"}]) == 4
        try:
            remaining(2, [{"sample_sha256": "a"}, {"sample_sha256": "a"}])
        except ValueError:
            pass
        else:
            raise AssertionError("Duplicate content was accepted")
        print("PASS: single/multi-view visibility, content uniqueness, target count and retry seed progression")
        return
    if not args.output:
        parser.error("Require --output")
    output = args.output.resolve()
    data_root = (ROOT / "data").resolve()
    if data_root not in output.parents or args.budget_gb <= 0:
        parser.error("Output must be below project data/ and budget must be positive")
    if args.collect:
        print(json.dumps(dict(collected=len(collect(output)), output=str(output))))
        return
    if not args.manifest or not args.recipe or not (args.all or args.pilot):
        parser.error("Require --manifest --recipe and either --pilot or --all")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    sys.path[:0] = [str(ROOT), str(ROOT / "third_party/PhysCoRe")]
    import torch
    import yaml
    from physcore.particle_flow.augmentation import generate_smoothfield_episodes
    from datagen.augment.run_recipe import validate_recipe
    recipe = yaml.safe_load(args.recipe.read_text())
    validate_recipe(recipe, args.recipe)
    sources = [r for r in json.loads(args.manifest.read_text()) if r["case"].startswith("phystwin__")]
    # ponytail: parallel workers own disjoint --source sets; merge once with --collect.
    if args.source:
        wanted = set(args.source)
        sources = [r for r in sources if r["case"] in wanted]
        if {r["case"] for r in sources} != wanted:
            raise ValueError("Requested source is not an available PhysTwin case")
    if not sources or (not args.source and len(sources) != 14):
        raise ValueError("Expected all 14 original PhysTwin sources")
    if args.pilot:
        sources = sources[:1]
    output.mkdir(parents=True, exist_ok=True)
    (output / "logs").mkdir(exist_ok=True)
    (output / "state").mkdir(exist_ok=True)
    existing = [json.loads(p.read_text()) for p in (output / "state").glob("*.json")]
    seen = {r["sample_sha256"] for s in existing for r in s["records"]}
    budget = int(args.budget_gb * 1e9)
    # ponytail: other datasets stay unchanged during this run; restart to refresh this baseline if they change.
    baseline_bytes = directory_bytes(data_root) - directory_bytes(output)
    used = baseline_bytes + directory_bytes(output)
    reserve = 256 * 1024**2
    new_count = 0
    started = time.monotonic()
    planned = len(sources) * sum(1 if args.pilot else int(g["num_episodes"]) for g in recipe["groups"])
    print(json.dumps(dict(phase="start", sources=len(sources), planned=planned,
        data_bytes=used, data_budget_bytes=budget, disk_free_bytes=shutil.disk_usage(data_root).free)), flush=True)
    for record in sources:
        source = Path(record["episode"])
        if args.source_root:
            source = args.source_root / record["case"] / "episode_0000"
        raw = torch.load(source / "episode_data.pt", map_location="cpu", weights_only=False)
        groups = [prepare_group(g, record, raw) for g in recipe["groups"]]
        frame_count = len(raw["particle_coords"])
        if frame_count < 6:
            raise ValueError("Source has fewer than six camera frames")
        source_bytes = (source / "episode_data.pt").stat().st_size
        del raw
        for group, physics in groups:
            target = 1 if args.pilot else int(group["num_episodes"])
            job = record["case"] + "_" + group["name"]
            if not all(c.isalnum() or c in "_-" for c in job):
                raise ValueError("Unsafe source/group name")
            job_path = output / "data" / job
            state_path = output / "state" / (job + ".json")
            signature = hashlib.sha256(json.dumps(dict(group=group, physics=physics,
                source=str(source.resolve()), source_bytes=source_bytes,
                prior_sha256=hashlib.sha256(Path(record["prior"]).read_bytes()).hexdigest()), sort_keys=True).encode()).hexdigest()
            state = json.loads(state_path.read_text()) if state_path.exists() else dict(
                recipe_sha256=signature, source_case=record["case"], group=group, physics=physics,
                records=[], attempts=[])
            if state["recipe_sha256"] != signature:
                raise ValueError(f"Preserve existing {job}: recipe/source/prior changed")
            if any(not (Path(r["episode"]) / "episode_data.pt").is_file() for r in state["records"]):
                raise ValueError(f"Accepted episode missing in {job}; preserve state for inspection")
            while remaining(target, state["records"]):
                if args.limit and new_count >= args.limit:
                    print(json.dumps(dict(phase="limited", new_accepted=new_count, elapsed_seconds=time.monotonic()-started)), flush=True)
                    if not args.source:
                        collect(output)
                    return
                index = next_index(state["attempts"])
                if index >= max(20, target * 5):
                    raise RuntimeError(f"Too many rejected episodes for {job}; inspect logs")
                if index % 10 == 0:
                    used = baseline_bytes + directory_bytes(output)
                allowance = max(reserve, source_bytes * 3)
                if used + allowance > budget or shutil.disk_usage(data_root).free < allowance:
                    raise RuntimeError("Data budget/free-space guard stopped before generating another episode")
                attempt = dict(index=index, status="started")
                state["attempts"].append(attempt)
                save_json(state_path, state)
                candidate = job_path / f"episode_{index:04d}"
                if candidate.exists():
                    raise FileExistsError(f"Preserve unexpected existing episode: {candidate}")
                generated = False
                try:
                    with (output / "logs" / (job + ".log")).open("a", encoding="utf-8") as log:
                        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                            generate_smoothfield_episodes(source, job_path, group, [index], args.device)
                    generated = True
                    digest, stats = quality(candidate)
                    if digest in seen:
                        raise ValueError("Duplicate generated observations/trajectory")
                    row = dict(record, case=f"{job}_{index:04d}", episode=str(candidate), split="train",
                        source_case=record["case"], data_origin="synthetic_mpm", augmentation_group=group["name"],
                        augmentation_index=index, rollout=group["rollout"], physics=physics,
                        sample_sha256=digest, generation_recipe_sha256=signature)
                    state["records"].append(row)
                    seen.add(digest)
                    attempt.update(status="accepted", sample_sha256=digest, quality=stats)
                    new_count += 1
                except ValueError as error:
                    if not generated:
                        attempt.update(status="failed", reason=str(error))
                        save_json(state_path, state)
                        raise
                    attempt.update(status="rejected", reason=str(error))
                except Exception as error:
                    attempt.update(status="failed", reason=f"{type(error).__name__}: {error}")
                    save_json(state_path, state)
                    raise
                finally:
                    size = directory_bytes(candidate) if candidate.exists() else 0
                    used += size + 65536
                    reserve = max(reserve, size * 2)
                save_json(state_path, state)
                print(json.dumps(dict(phase="generated", source=record["case"], group=group["name"],
                    attempt=attempt, accepted=len(state["records"]), target=target, data_bytes_estimate=used)), flush=True)
    records = collect(output) if not args.source else [r for p in (output / "state").glob("*.json") for r in json.loads(p.read_text())["records"]]
    print(json.dumps(dict(phase="complete", accepted=len(records), new_accepted=new_count,
        elapsed_seconds=time.monotonic()-started, output=str(output))), flush=True)


if __name__ == "__main__":
    main()

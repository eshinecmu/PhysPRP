"""Resume per-scene preparation using the original PhysCoRe and GaussianProperty tools."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VAL = {
    "physcore__double_clift_cloth", "physcore__double_fold_rope",
    "physcore__double_squeeze_plastic", "phystwin__single_clift_cloth_3",
    "phystwin__single_push_rope_1", "phystwin__double_lift_sloth",
    "phystwin__single_lift_zebra",
}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2))


def audit_case(source):
    meta = json.loads((source / "metadata.json").read_text())
    with (source / "sampled_tracks.pkl").open("rb") as handle:
        tracks = pickle.load(handle)
    with (source / "calibrate.pkl").open("rb") as handle:
        cameras = np.asarray(pickle.load(handle))
    xyz = np.asarray(tracks["object_points"])
    controls = np.asarray(tracks["controller_points"])
    valid = np.asarray(tracks["object_motions_valid"], dtype=bool)
    if min(int(meta["frame_num"]), len(xyz), len(controls)) < 12:
        raise ValueError("Needs at least 12 synchronized frames")
    if cameras.shape != (3, 4, 4) or not np.isfinite(cameras).all():
        raise ValueError("Missing finite three-view 4x4 camera calibration")
    if not np.isfinite(controls[:12]).all() or controls.shape[1] == 0:
        raise ValueError("Missing finite observed hand controllers")
    initial_valid = valid[0] & np.isfinite(xyz[0]).all(-1)
    if initial_valid.sum() < 8:
        raise ValueError("Fewer than 8 measured initial tracks")
    for view in range(3):
        labels = json.loads((source / f"mask/mask_info_{view}.json").read_text())
        objects = [key for key, value in labels.items() if value.lower() != "hand"]
        hands = [key for key, value in labels.items() if value.lower() == "hand"]
        if not objects or not hands:
            raise ValueError(f"view {view}: object/hand masks missing")
        for frame in range(12):
            for filename in (f"color/{view}/{frame}.png", f"depth/{view}/{frame}.npy"):
                if not (source / filename).is_file():
                    raise FileNotFoundError(source / filename)
            for mask_id in (*objects, *hands):
                if not (source / f"mask/{view}/{mask_id}/{frame}.png").is_file():
                    raise FileNotFoundError(f"view {view}, mask {mask_id}, frame {frame}")
    return dict(frames=int(meta["frame_num"]), fps=float(meta["fps"]),
                measured_tracks=int(xyz.shape[1]), initial_valid_tracks=int(initial_valid.sum()),
                initial_extent_metres=np.ptp(xyz[0, initial_valid], axis=0).tolist(),
                calibration_sha256=hashlib.sha256((source / "calibrate.pkl").read_bytes()).hexdigest())


def make_prior(source, key, output):
    import cv2
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "GaussianProperty"))
    from PhysGaussian.gaussian_property_adapter import parse_predictions
    import vlm_predict
    from utils.vlm_utils import Qwen

    asset = output / "assets" / key
    result = output / "gaussian_property" / key
    result.mkdir(parents=True, exist_ok=True)
    prior = result / "object_properties.json"
    if prior.is_file():
        return
    asset.mkdir(parents=True, exist_ok=True)
    (asset / "masks").mkdir(exist_ok=True)
    manifest = [{"object_id": 0, "name": "object"}]
    write_json(asset / "manifest.json", manifest)
    labels = json.loads((source / "mask/mask_info_0.json").read_text())
    mask = None
    for mask_id, label in labels.items():
        if label.lower() == "hand":
            continue
        current = cv2.imread(str(source / f"mask/0/{mask_id}/0.png"), cv2.IMREAD_UNCHANGED) > 0
        mask = current if mask is None else mask | current
    if mask is None or not mask.any():
        raise ValueError("Empty source object mask")
    cv2.imwrite(str(asset / "masks/object.png"), mask.astype(np.uint8) * 255)
    command = [sys.executable, "-B", str(ROOT / "prepare_rest3d_gaussian_property.py"),
               "--image", str(source / "color/0/0.png"), "--mask_dir", str(asset / "masks"),
               "--manifest", str(asset / "manifest.json"), "--output_root", str(output / "gaussian_property"),
               "--case_name", key]
    subprocess.run(command, check=True)
    clarification = ("\nUse numeric Shore bounds between 0 and 100, e.g. hardness 10-30. "
                     "The words low and high are placeholders, not output values. "
                     "Return exactly four comma-separated fields and no markdown.")
    # ponytail: preserve the original prompt/client; only clarify its numeric parser contract.
    vlm_predict.Qwen = lambda image, prompt: Qwen(image, prompt + clarification)
    os.environ["LAB_LLM_MODEL"] = "qwen3-vl-plus"
    prediction_file = result / f"{key}.txt"
    last_error = None
    for attempt in range(1, 4):
        vlm_predict.query_vlm(str(output / "gaussian_property"), key, vlm_type="qwen")
        try:
            records = parse_predictions(prediction_file.read_text(), manifest)
            write_json(prior, records)
            panel = result / "gpt_input/01/0000.png"
            write_json(result / "prior_generation.json", dict(
                pipeline="original GaussianProperty prompt/Qwen/adapter", model="qwen3-vl-plus",
                parameter_source="semantic_material_lookup", is_measured=False,
                numeric_format_clarification=clarification.strip(), attempts=attempt,
                source_image=str(source / "color/0/0.png"),
                input_sha256=hashlib.sha256(panel.read_bytes()).hexdigest(),
                source_image_sha256=hashlib.sha256((source / "color/0/0.png").read_bytes()).hexdigest()))
            return
        except ValueError as error:
            last_error = error
            print(json.dumps(dict(case=key, prior_attempt=attempt, error=str(error))), flush=True)
    raise last_error


def make_episode(source, key, output):
    import torch
    target = output / "episodes" / key
    episode = target / "episode_0000/episode_data.pt"
    if episode.is_file():
        raw = torch.load(episode, map_location="cpu", weights_only=False)
        if raw.get("is_real_world") and len(raw["particle_coords"]) >= 12:
            return
        raise ValueError("Existing episode failed readiness check; preserve for inspection")
    cloth = "cloth" in source.name
    command = [sys.executable, "-B", str(ROOT / "third_party/PhysCoRe/datagen/convert3d/convert_to_episode.py"),
        "--source_dir", str(source), "--output_dir", str(target), "--frame_limit", "12",
        "--max_observed_points", "256", "--target_total_particles", "16000" if cloth else "12000",
        "--ground_clip_reference_frames", "1", "--dpsr_device", "cpu",
        "--controller_mask_label", "hand", "--particle_voxel_size", ".004" if cloth else ".008"]
    if cloth:
        command.extend(["--interior_to_shell_max_distance", ".004"])
    print(json.dumps(dict(case=key, converter_command=command)), flush=True)
    subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reuse", type=Path, required=True)
    parser.add_argument("--phase", choices=("audit", "gp", "episodes", "manifest"), required=True)
    args = parser.parse_args()
    args.data, args.output, args.reuse = args.data.resolve(), args.output.resolve(), args.reuse.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    old = {row["case"]: row for row in json.loads((args.reuse / "episode_manifest.json").read_text())}
    statuses, records = [], []
    if args.phase == "manifest":
        import torch
        from omegaconf import OmegaConf
        official = OmegaConf.to_container(OmegaConf.load(ROOT / "third_party/PhysCoRe/configs/train_RfD.yaml"), resolve=True)
    for family in ("physcore", "phystwin"):
        for source in sorted((args.data / family / "different_types").iterdir()):
            key = f"{family}__{source.name}"
            row = dict(case=key, official_case=source.name, source_dataset=family,
                       source=str(source), split="val" if key in VAL else "train", object_id=0)
            status = dict(row, phase=args.phase, status="PENDING")
            try:
                status["raw"] = audit_case(source)
                reuse = old.get(source.name) if family == "physcore" else None
                if reuse:
                    row.update({field: reuse[field] for field in ("episode", "prior", "rollout", "prior_generation")})
                    panel = Path(row["prior"]).parent / "gpt_input/01/0000.png"
                    row["prior_generation"] = dict(row["prior_generation"],
                        input_sha256=hashlib.sha256(panel.read_bytes()).hexdigest(),
                        source_image_sha256=hashlib.sha256((source / "color/0/0.png").read_bytes()).hexdigest())
                    row["reused_from"] = str(args.reuse)
                else:
                    row.update(episode=str(args.output / "episodes" / key / "episode_0000"),
                               prior=str(args.output / "gaussian_property" / key / "object_properties.json"))
                    if args.phase == "gp":
                        make_prior(source, key, args.output)
                    if args.phase == "episodes":
                        make_episode(source, key, args.output)
                if args.phase == "manifest":
                    if not Path(row["episode"], "episode_data.pt").is_file() or not Path(row["prior"]).is_file():
                        raise FileNotFoundError("Episode or GaussianProperty prior not yet prepared")
                    raw = torch.load(Path(row["episode"]) / "episode_data.pt", map_location="cpu", weights_only=False)
                    prior = json.loads(Path(row["prior"]).read_text())
                    if not prior or len(raw["particle_coords"]) < 12:
                        raise ValueError("Incomplete episode/prior")
                    row.setdefault("rollout", official.get("per_sample_rollout_config", {}).get(source.name, {}))
                    if not reuse:
                        row["prior_generation"] = json.loads(Path(row["prior"]).with_name("prior_generation.json").read_text())
                    status["episode"] = dict(particles=int(raw["particle_coords"].shape[1]),
                        tracked=int(raw["tracked_particle_count"]), shape_completion=raw["shape_completion"],
                        target_tracks={str(f): int(((raw["tracked_visible_particle_indices"][f] >= 0)).sum()) for f in (4, 5)})
                    records.append(row)
                status["status"] = "READY"
            except Exception as error:
                status.update(status="FAILED", error=f"{type(error).__name__}: {error}")
            statuses.append(status)
            print(json.dumps(dict(case=key, phase=args.phase, status=status["status"], error=status.get("error"))), flush=True)
            if args.phase == "manifest":
                write_json(args.output / "episode_manifest.json", records)
            write_json(args.output / ("case_status.json" if args.phase == "manifest" else f"case_status_{args.phase}.json"), statuses)
    if args.phase == "manifest":
        write_json(args.output / "split.json", dict(
            rule="Fixed source/semantic-object stratification before examining losses; preserve the earlier validation case and earlier two training cases. This is held-out-scene evaluation, not held-out-object-identity evaluation.",
            validation_cases=sorted(VAL), train_cases=[r["case"] for r in records if r["split"] == "train"],
            validation_ready=[r["case"] for r in records if r["split"] == "val"],
            attempted_cases=len(statuses), ready_cases=len(records)))
    print(json.dumps(dict(phase=args.phase, ready=sum(s["status"] == "READY" for s in statuses), total=len(statuses))), flush=True)


if __name__ == "__main__":
    main()

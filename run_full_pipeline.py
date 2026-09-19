"""Thin runner for REST3D -> GaussianProperty -> relational PhysGaussian -> GT evaluation."""

import argparse
import os
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def run(command, cwd=None, env=None):
    print("+", " ".join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), cwd=cwd, env=env, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--camera_json", required=True, help="Single-view pose/intrinsics in PhysGaussian cameras.json format")
    parser.add_argument("--config", required=True)
    parser.add_argument("--work_dir", required=True)
    parser.add_argument("--gt_video")
    parser.add_argument("--relation_checkpoint")
    parser.add_argument("--rest_python", default=str(ROOT / "REST3D/envs/rest3d/bin/python"))
    parser.add_argument("--gp_python", default="python")
    parser.add_argument("--phys_python", default=str(ROOT / "PhysGaussian/.venv/bin/python"))
    parser.add_argument("--vlm", choices=("qwen", "gpt"), default="qwen")
    args = parser.parse_args()

    image = Path(args.image).resolve()
    work = Path(args.work_dir).resolve()
    rest_output, gp_output = work / "rest3d", work / "gaussian_property"
    manifest, relations = work / "object_manifest.json", work / "object_relations.json"
    properties = work / "object_properties.json"
    model = work / "phys_model"
    point_dir = model / "point_cloud" / "iteration_60000"
    point_dir.mkdir(parents=True, exist_ok=True)

    rest_env = os.environ.copy()
    rest_env["PYTHONPATH"] = str(ROOT / "REST3D") + os.pathsep + rest_env.get("PYTHONPATH", "")
    run((args.rest_python, "scripts/infer_scenetree.py", "--image_folder", image,
         "--output_folder", rest_output), ROOT / "REST3D", rest_env)
    run((args.rest_python, "scripts/infer_scene3d.py", "--image_folder", image,
         "--output_folder", rest_output), ROOT / "REST3D", rest_env)

    scene_root = rest_output / image.stem
    run((args.phys_python, ROOT / "rest3d_physgaussian_bridge.py",
         "--scene_tree", scene_root / "stage1/scene_tree.json",
         "--mesh_dir", scene_root / "stage2/scene_canon/obj_files",
         "--manifest_out", manifest, "--relations_out", relations))
    run((args.phys_python, ROOT / "prepare_rest3d_gaussian_property.py",
         "--image", image, "--mask_dir", scene_root / "stage1/segemented_obj",
         "--manifest", manifest, "--output_root", gp_output, "--case_name", image.stem))
    run((args.gp_python, "vlm_predict.py", "--dataset_path", gp_output, "--vlm", args.vlm),
        ROOT / "GaussianProperty")
    predictions = gp_output / image.stem / f"{image.stem}.txt"
    run((args.phys_python, ROOT / "PhysGaussian/gaussian_property_adapter.py",
         predictions, properties, "--manifest", manifest))

    point_cloud = point_dir / "point_cloud.ply"
    run((args.phys_python, ROOT / "PhysGaussian/replica_mesh_to_gaussian.py", manifest, point_cloud))
    shutil.copy2(args.camera_json, model / "cameras.json")
    shutil.copy2(image, model / "input.png")
    object_ids = point_cloud.with_suffix(".object_id.npy")

    simulation = work / "relation_simulation"
    command = [args.phys_python, ROOT / "PhysGaussian/gs_simulation.py",
               "--model_path", model, "--output_path", simulation, "--config", args.config,
               "--object_id_path", object_ids, "--property_path", properties,
               "--scene_relation_path", relations, "--enable_relation",
               "--render_img", "--compile_video", "--output_ply", "--output_h5"]
    if args.relation_checkpoint:
        command += ["--relation_full_checkpoint", args.relation_checkpoint]
    run(command, ROOT / "PhysGaussian")

    if args.gt_video:
        run((args.phys_python, ROOT / "evaluate_video.py", "--pred", simulation / "output.mp4",
             "--gt", args.gt_video, "--output", work / "relation_vs_gt.json"))


if __name__ == "__main__":
    main()

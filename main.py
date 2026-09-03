"""PhysPRP pipeline: orchestration only; model work stays in upstream code."""

import argparse
import json
import subprocess
from pathlib import Path

from dataloader.pac_nerf import PACNeRFStaticRGB


STAGES = (
    "rest3d_scene_tree",
    "rest3d_reconstruction",
    "rest3d_multiview",
    "pixie_material",
    "pac_nerf_train",
    "pac_nerf_render",
)


def run_stage(name: str, stage: dict, values: dict, root: Path) -> None:
    argv = [str(value).format_map(values) for value in stage["argv"]]
    cwd = (root / stage.get("cwd", ".")).resolve()
    print(f"[{name}] {' '.join(argv)}")
    subprocess.run(argv, cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the existing REST3D → PIXIE → PAC-NeRF interfaces")
    parser.add_argument("config", type=Path, help="Pipeline JSON configuration")
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    root = Path(__file__).resolve().parent
    sample = PACNeRFStaticRGB(
        config["scene_dir"], config.get("view_id", 0), config.get("frame_id", 0)
    )[0]

    work_dir = Path(config.get("work_dir", root / "outputs")).resolve()
    sample_name = Path(sample["image_path"]).stem
    values = {
        "rgb": sample["image_path"],
        "scene_dir": str(Path(config["scene_dir"]).resolve()),
        "sample_name": sample_name,
        "object_name": config["object_name"],
        "work_dir": str(work_dir),
        "rest3d_output": str(work_dir / "rest3d"),
        "rest3d_obj": str(work_dir / "rest3d" / sample_name / "stage2" / "scene_canon" / "obj_files" / f"scene_canon_{config['object_name']}.obj"),
        "multiview_output": str(work_dir / "multiview"),
        "material_output": str(work_dir / "material" / "mapped_preds.npz"),
        "pac_output": str(work_dir / "pac_nerf"),
        "video_output": str(work_dir / "video"),
    }

    commands = config["commands"]
    missing = [name for name in STAGES if name not in commands]
    if missing:
        raise ValueError(f"Missing pipeline commands: {', '.join(missing)}")
    for name in STAGES:
        run_stage(name, commands[name], values, root)


if __name__ == "__main__":
    main()

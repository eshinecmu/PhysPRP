"""Export native Z-up Gaussians and the exact SAM3D object-to-scene map."""
import json
from pathlib import Path
import numpy as np
import torch
from pytorch3d.transforms import quaternion_to_matrix
from sam3d_objects.data.dataset.tdfy.transforms_3d import compose_transform


def export_native_object(output, name, mask_path, directory):
    directory = Path(directory)
    gaussian = output.get("gs")
    if gaussian is None:
        gaussian = output.get("gaussian", [None])[0]
    if gaussian is None:
        raise RuntimeError(f"No native Gaussian output for {name}")
    if output.get("apply_orientation_fix", False):
        raise ValueError("Native export requires unmodified inferred scene placement")
    transform = compose_transform(output["scale"], quaternion_to_matrix(output["rotation"]), output["translation"])
    # Transform origin and basis with the actual upstream API; no row/column guess.
    basis = torch.cat((torch.zeros(1, 3), torch.eye(3))).to(output["rotation"])
    points = transform.transform_points(basis[None])[0].detach().cpu().numpy()
    affine = np.eye(4)
    affine[:3, :3] = (points[1:] - points[0]).T
    affine[:3, 3] = points[0]
    gaussian.save_ply(str(directory / f"{name}.ply"))
    arrays = {key: output[key].detach().cpu().numpy() for key in ("scale", "rotation", "translation", "pointmap") if key in output and torch.is_tensor(output[key])}
    np.savez(directory / f"{name}_source_pose.npz", **arrays)
    metadata = {"schema_version": 1, "name": name, "status": "complete", "mask": str(Path(mask_path).resolve()),
                "gaussian_frame": "sam3d_object_z_up", "scene_frame": "sam3d_camera_scene",
                "object_to_scene": affine.tolist(), "orientation_fix_applied": False,
                "ply_convention": {"scale": "log", "opacity": "logit", "rotation": "wxyz", "sh_degree": 0},
                "transform_source": "sam3d compose_transform(scale, quaternion_to_matrix(rotation), translation)"}
    (directory / f"{name}_pose.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

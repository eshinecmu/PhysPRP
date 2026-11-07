"""Adapt native SAM3D Gaussians using explicitly recorded object-to-scene pose."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np


def read_obj_vertices(path):
    rows = [list(map(float, line.split()[1:4])) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.startswith("v ")]
    value = np.asarray(rows, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 3 or len(value) < 3 or not np.isfinite(value).all():
        raise ValueError(f"Invalid OBJ vertices: {path}")
    return value


def rigid_transform(src, dst):
    if src.shape != dst.shape:
        raise ValueError("OBJ vertices must correspond")
    u, _, vt = np.linalg.svd((src-src.mean(0)).T @ (dst-dst.mean(0)))
    correction = np.eye(3); correction[-1,-1] = np.linalg.det(vt.T @ u.T)
    rotation = vt.T @ correction @ u.T
    translation = dst.mean(0) - rotation @ src.mean(0)
    if not np.allclose(src @ rotation.T + translation, dst, atol=1e-5):
        raise ValueError("OBJ mapping is not rigid")
    return rotation, translation


def quaternion_from_matrix(rotation):
    from scipy.spatial.transform import Rotation
    q = Rotation.from_matrix(rotation).as_quat()
    return q[..., [3, 0, 1, 2]]


def transform_vertices(vertex, linear, translation, sh_degree=3):
    """Transform means and full covariance, supporting anisotropic scene scale."""
    from scipy.spatial.transform import Rotation
    names = vertex.dtype.names or ()
    required = ["x", "y", "z", "opacity"] + [f"f_dc_{i}" for i in range(3)] + [f"scale_{i}" for i in range(3)] + [f"rot_{i}" for i in range(4)]
    if not set(required).issubset(names) or len(vertex) == 0:
        raise ValueError("Native Gaussian PLY missing required fields or empty")
    if np.isnan(vertex["opacity"]).any() or not all(np.isfinite(vertex[n]).all() for n in names if n != "opacity"):
        raise ValueError("Non-finite Gaussian field")
    if np.shape(linear) != (3,3) or not np.isfinite(linear).all() or np.linalg.det(linear) <= 0:
        raise ValueError("Expected invertible, orientation-preserving scene transform")
    rest = [n for n in names if n.startswith("f_rest_")]
    if any(np.any(vertex[n] != 0) for n in rest):
        raise ValueError("Nonzero higher-degree SH requires SH rotation; native SAM3D export must be degree 0")
    dtype = [(n, vertex.dtype[n]) for n in names]
    dtype += [(f"f_rest_{i}", "f4") for i in range(3*((sh_degree+1)**2-1)) if f"f_rest_{i}" not in names]
    result = np.zeros(len(vertex), dtype=dtype)
    for n in names: result[n] = vertex[n]
    # SAM3D can export logit(1)=+inf; preserve saturation to output dtype precision.
    saturated = np.isinf(result["opacity"])
    epsilon = np.finfo(result.dtype["opacity"]).eps
    limit = np.nextafter(np.asarray(np.log((1-epsilon)/epsilon), dtype=result.dtype["opacity"]), np.asarray(np.inf, dtype=result.dtype["opacity"]))
    result["opacity"][saturated] = np.sign(result["opacity"][saturated]) * limit
    xyz = np.column_stack([vertex[n] for n in ("x","y","z")])
    xyz = xyz @ linear.T + np.asarray(translation)
    for i,n in enumerate(("x","y","z")): result[n] = xyz[:,i]
    q = np.column_stack([vertex[f"rot_{i}"] for i in range(4)])
    if np.any(np.linalg.norm(q,axis=1) < 1e-12): raise ValueError("Zero Gaussian quaternion")
    rotation = Rotation.from_quat(q[:,[1,2,3,0]]).as_matrix()
    scales = np.exp(np.column_stack([vertex[f"scale_{i}"] for i in range(3)]).astype(float))
    if not np.isfinite(scales).all() or np.any(scales <= 0): raise ValueError("Invalid Gaussian scale")
    basis = linear[None] @ (rotation * scales[:,None,:])
    covariance = basis @ basis.transpose(0,2,1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if np.any(eigenvalues <= 0): raise ValueError("Non-positive Gaussian covariance")
    eigenvectors[:,:,2] *= np.where(np.linalg.det(eigenvectors)<0,-1,1)[:,None]
    transformed_q = quaternion_from_matrix(eigenvectors)
    for i in range(3): result[f"scale_{i}"] = .5*np.log(eigenvalues[:,i])
    for i in range(4): result[f"rot_{i}"] = transformed_q[:,i]
    if all(n in names for n in ("nx","ny","nz")):
        normals = np.column_stack([vertex[n] for n in ("nx","ny","nz")]) @ np.linalg.inv(linear)
        normals /= np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-12)
        for i,n in enumerate(("nx","ny","nz")): result[n] = normals[:,i]
    return result


def transform_ply(source, target, rotation, translation):
    from plyfile import PlyData, PlyElement
    vertex = transform_vertices(PlyData.read(str(source))["vertex"].data, rotation, translation)
    Path(target).parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(vertex,"vertex")]).write(str(target))
    return len(vertex)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--raw-obj",type=Path,required=True)
    p.add_argument("--canonical-obj",type=Path,help="Explicit diagnostic placement override; default preserves raw scene")
    p.add_argument("--raw-gaussian",type=Path,required=True)
    p.add_argument("--pose",type=Path,required=True)
    p.add_argument("--model-path",type=Path,required=True)
    p.add_argument("--iteration",type=int,default=1)
    a=p.parse_args()
    pose=json.loads(a.pose.read_text(encoding="utf-8"))
    transform=np.asarray(pose["object_to_scene"],dtype=float)
    if transform.shape != (4,4) or not np.allclose(transform[3],[0,0,0,1]): raise ValueError("Invalid affine pose")
    if a.canonical_obj:
        r,t=rigid_transform(read_obj_vertices(a.raw_obj),read_obj_vertices(a.canonical_obj))
        transform[:3,3]=r@transform[:3,3]+t; transform[:3,:3]=r@transform[:3,:3]
    target=a.model_path/"point_cloud"/f"iteration_{a.iteration}"/"point_cloud.ply"
    count=transform_ply(a.raw_gaussian,target,transform[:3,:3],transform[:3,3])
    (a.model_path/"rest3d_alignment.json").write_text(json.dumps({"object_to_output":transform.tolist(),"gaussian_count":count,"placement":"canonical_override" if a.canonical_obj else "raw_scene","pose_source":str(a.pose)},indent=2),encoding="utf-8")
    print(target)

if __name__ == "__main__": main()

"""Validate native Gaussian numerical format; projection is a separate renderer check."""
import argparse
import json
from pathlib import Path
import numpy as np
from plyfile import PlyData


def validate(ply, identities=None):
    vertex=PlyData.read(str(ply))["vertex"].data
    required=["x","y","z","opacity"]+[f"scale_{i}" for i in range(3)]+[f"rot_{i}" for i in range(4)]+[f"f_dc_{i}" for i in range(3)]
    if not len(vertex) or not set(required).issubset(vertex.dtype.names): raise ValueError("Empty or incomplete Gaussian PLY")
    if not all(np.isfinite(vertex[n]).all() for n in vertex.dtype.names): raise ValueError("Non-finite Gaussian fields")
    scales=np.exp(np.column_stack([vertex[f"scale_{i}"] for i in range(3)]).astype(float))
    q=np.column_stack([vertex[f"rot_{i}"] for i in range(4)])
    if not np.isfinite(scales).all() or np.any(scales<=0): raise ValueError("Invalid Gaussian covariance eigenvalues")
    if not np.allclose(np.linalg.norm(q,axis=1),1,atol=1e-4): raise ValueError("Quaternion normalization invalid")
    if identities is not None:
        ids=np.load(identities)
        if ids.shape!=(len(vertex),) or ids.dtype.kind not in "iu" or np.any(ids<0): raise ValueError("Invalid object identity array")
    return {"status":"numerical_format_pass","gaussians":len(vertex),"minimum_scale":float(scales.min()),"maximum_scale":float(scales.max()),"projection_alignment":"requires_actual_first_frame_render"}


def main():
    p=argparse.ArgumentParser();p.add_argument("--model-path",type=Path,required=True);p.add_argument("--iteration",type=int,default=60000);p.add_argument("--canonical-obj",type=Path,help="Deprecated; AABB is not projection validation")
    a=p.parse_args();path=a.model_path/"point_cloud"/f"iteration_{a.iteration}"/"point_cloud.ply"
    ids=path.with_suffix(".object_id.npy")
    print(json.dumps(validate(path,ids if ids.exists() else None)))

if __name__ == "__main__": main()

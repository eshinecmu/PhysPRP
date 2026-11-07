"""Run: python tests/check_native_scene.py (numpy, scipy, plyfile)."""
import sys
import tempfile
from pathlib import Path
import json
import numpy as np
from scipy.spatial.transform import Rotation
from plyfile import PlyData,PlyElement
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
from prepare_rest3d_object import transform_vertices
from prepare_rest3d_scene import prepare_scene
from validate_rest3d_object import validate


def main():
    fields=["x","y","z","opacity"]+[f"f_dc_{i}" for i in range(3)]+[f"scale_{i}" for i in range(3)]+[f"rot_{i}" for i in range(4)]
    v=np.zeros(4,dtype=[(n,"f4") for n in fields]); v["x"]=[0,.1,.2,.3];v["rot_0"]=1
    for i,s in enumerate([.01,.02,.03]):v[f"scale_{i}"]=np.log(s)
    r=Rotation.from_euler("xyz",[.3,.8,-.4]).as_matrix();a=r@np.diag([2.,3.,4.]);t=np.array([1.,2.,3.])
    out=transform_vertices(v,a,t)
    xyz=np.column_stack([out[n] for n in ("x","y","z")]);assert np.allclose(xyz,np.column_stack([v[n] for n in ("x","y","z")])@a.T+t)
    q=np.column_stack([out[f"rot_{i}"] for i in range(4)]);rr=Rotation.from_quat(q[:,[1,2,3,0]]).as_matrix()
    ss=np.exp(np.column_stack([out[f"scale_{i}"] for i in range(3)]));b=rr*ss[:,None,:]
    assert np.allclose(b@b.transpose(0,2,1),a@np.diag([.01**2,.02**2,.03**2])@a.T,atol=1e-8)
    assert all(np.all(out[f"f_rest_{i}"]==0) for i in range(45))
    saturated=v.copy();saturated['opacity'][:2]=[np.inf,-np.inf]
    fixed=transform_vertices(saturated,a,t)
    assert np.isfinite(fixed['opacity']).all()
    assert np.allclose(1/(1+np.exp(-fixed['opacity'][:2])),[1,0],atol=np.finfo('f4').eps)
    assert np.isinf(saturated['opacity'][:2]).all(), 'Source must remain unchanged'
    for field in ('opacity','x'):
        bad=v.copy();bad[field][0]=np.nan if field=='opacity' else np.inf
        try:transform_vertices(bad,a,t)
        except ValueError:pass
        else:raise AssertionError('Invalid Gaussian accepted')
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);raw=root/"stage2/scene_raw/obj_files";raw.mkdir(parents=True);(root/"stage1").mkdir()
        for name,delta in (("left",0),("right",3)):
            PlyData([PlyElement.describe(v,"vertex")]).write(str(raw/f"{name}.ply"))
            (raw/f"{name}.obj").write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\n")
            affine=np.eye(4);affine[0,3]=delta
            (raw/f"{name}_pose.json").write_text(json.dumps({"object_to_scene":affine.tolist(),"gaussian_frame":"sam3d_object_z_up","orientation_fix_applied":False}))
        (root/"stage1/scene_tree.json").write_text(json.dumps({"edges":[{"child":"left","parent":"right","relation":"on"}]}))
        cfg={"scene_scale":2,"objects":{"left":{"role":"dynamic"},"right":{"role":"static"}}}
        prepare_scene(root,cfg,root/"model",root/"manifest.json",root/"relations.json")
        path=root/"model/point_cloud/iteration_60000/point_cloud.ply"
        validate(path,path.with_suffix(".object_id.npy"))
        vertices=PlyData.read(str(path))["vertex"].data
        assert np.allclose(vertices["x"][4:]-vertices["x"][:4],6)
        assert np.array_equal(np.load(path.with_suffix(".object_id.npy")),[0]*4+[1]*4)
        cfg["required_objects"]=["missing"]
        try:prepare_scene(root,cfg,root/"model",root/"manifest.json",root/"relations.json")
        except ValueError:pass
        else:raise AssertionError("Missing object silently accepted")
    print("PASS: affine covariance, SH layout, scene placement, identity, missing-object rejection")

if __name__=="__main__":main()

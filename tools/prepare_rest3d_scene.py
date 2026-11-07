"""Merge native objects in one shared scene frame, preserving identity and roles."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from plyfile import PlyData, PlyElement
from prepare_rest3d_object import transform_vertices


def prepare_scene(scene_root, config, model_path, manifest_out, relations_out, selection_path=None):
    raw=scene_root/"stage2"/"scene_raw"/"obj_files"
    selection=json.loads(Path(selection_path).read_text(encoding="utf-8")) if selection_path else None
    selected=set(selection["objects"]) if selection else None
    poses=sorted(p for p in raw.glob("*_pose.json") if selected is None or p.name.removesuffix("_pose.json") in selected)
    if not poses: raise FileNotFoundError(f"No native Gaussian pose exports in {raw}; mesh fallback disabled")
    scale=float(config["scene_scale"])
    if not np.isfinite(scale) or scale <= 0: raise ValueError("scene_scale must be positive meters/source unit")
    roles=config.get("objects", {})
    names={p.name.removesuffix("_pose.json") for p in poses}
    masks={p.stem for p in (scene_root/"stage1"/"segemented_obj").glob("*") if p.suffix.lower() in (".png",".jpg",".jpeg")}
    if selected is not None:
        if selected - masks:
            raise ValueError(f"Selected objects have no REST3D masks: {sorted(selected-masks)}")
        masks &= selected
    required=set(config.get("required_objects", []))
    configured=set(roles)
    if selected is not None:
        required &= selected
        configured &= selected
    missing=(required | masks | configured)-names
    if missing: raise ValueError(f"Required or segmented objects missing native exports: {sorted(missing)}")
    rows=[]; blocks=[]; ids=[]; offset=0
    for oid, pose_path in enumerate(poses):
        name=pose_path.name.removesuffix("_pose.json")
        settings=roles.get(name,{})
        role=settings.get("role",config.get("default_role"))
        if selection and "role" not in settings:
            object_type=selection.get("object_types",{}).get(name)
            if object_type == "movable": role="dynamic"
            elif object_type == "fixed":
                if "collider" in settings:
                    role="static"
                elif selection.get("scope") == "interaction":
                    raise ValueError(f"REST3D marked interacting object {name!r} fixed; configure an explicit MPM plane collider")
        if role not in ("dynamic","static","floor","background"): raise ValueError(f"Explicit valid role required for {name}")
        pose=json.loads(pose_path.read_text(encoding="utf-8"))
        if pose.get("gaussian_frame") != "sam3d_object_z_up" or pose.get("orientation_fix_applied") is not False: raise ValueError(f"Unverified Gaussian frame for {name}")
        matrix=np.asarray(pose["object_to_scene"],dtype=float)
        if matrix.shape != (4,4) or not np.allclose(matrix[3],[0,0,0,1]): raise ValueError(f"Invalid pose for {name}")
        source=raw/f"{name}.ply"; mesh=raw/f"{name}.obj"
        if not mesh.is_file(): raise FileNotFoundError(mesh)
        vertex=PlyData.read(str(source))["vertex"].data
        saturated_opacity=int(np.isinf(vertex["opacity"]).sum())
        data=transform_vertices(vertex,scale*matrix[:3,:3],scale*matrix[:3,3])
        blocks.append(data); ids.append(np.full(len(data),oid,dtype=np.int32))
        rows.append({"name":name,"object_id":oid,"role":role,"path":str(mesh.resolve()),"gaussian_source":str(source.resolve()),"pose_source":str(pose_path.resolve()),"gaussian_count":len(data),"saturated_opacity_logits_encoded":saturated_opacity,"gaussian_start":offset,"gaussian_end":offset+len(data),"scene_scale":scale})
        offset+=len(data)
    if not any(r["role"] == "dynamic" for r in rows): raise ValueError("Scene has no explicit dynamic object")
    if len({b.dtype for b in blocks})!=1: raise ValueError("Incompatible Gaussian field layouts")
    tree=json.loads((scene_root/"stage1"/"scene_tree.json").read_text(encoding="utf-8"))
    lookup={r["name"]:r["object_id"] for r in rows}; edges=[]; root_edges=[]
    for edge in tree.get("edges",[]):
        child,parent=edge["child"],edge["parent"]
        if child not in lookup:
            if selected is not None:
                continue
            raise ValueError(f"Unmapped relation child {child}")
        if parent not in lookup:
            if parent not in ("floor","wall","ceiling","floor-wall"): raise ValueError(f"Unmapped relation parent {parent}")
            root_edges.append({**edge,"status":"semantic_root_only_requires_explicit_collider"}); continue
        edges.append({"source":lookup[child],"target":lookup[parent],"source_name":child,"target_name":parent,"relation":edge.get("relation","other")})
    target=model_path/"point_cloud"/"iteration_60000"/"point_cloud.ply"
    target.parent.mkdir(parents=True,exist_ok=True)
    PlyData([PlyElement.describe(np.concatenate(blocks),"vertex")]).write(str(target))
    np.save(target.with_suffix(".object_id.npy"),np.concatenate(ids))
    for path,value in ((manifest_out,rows),(relations_out,{"edges":edges,"root_edges":root_edges})):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2)+"\n",encoding="utf-8")
    report={"status":"native_scene_assembled","objects":len(rows),"gaussians":offset,"placement":"raw_inferred_scene_without_canonicalization","length_unit":"meter","scene_scale":scale,"ply_sha256":hashlib.sha256(target.read_bytes()).hexdigest(),"first_frame_projection":"not_checked_by_this_adapter"}
    (model_path/"rest3d_alignment.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    return report


def main():
    p=argparse.ArgumentParser()
    for flag in ("scene-root","scene-config","model-path","manifest-out","relations-out"): p.add_argument("--"+flag,type=Path,required=True)
    p.add_argument("--selection-file",type=Path)
    a=p.parse_args()
    print(json.dumps(prepare_scene(a.scene_root,json.loads(a.scene_config.read_text(encoding="utf-8")),a.model_path,a.manifest_out,a.relations_out,a.selection_file)))

if __name__ == "__main__": main()

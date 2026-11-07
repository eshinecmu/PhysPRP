"""Fail before reconstruction when segmentation omits objects or invents a floor."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image


def validate_segmentation(scene_root, config):
    stage=Path(scene_root)/"stage1"
    masks={}
    for path in sorted((stage/"segemented_obj").glob("*.png")):
        with Image.open(path) as image:
            if "A" not in image.getbands(): raise ValueError(f"Mask requires explicit alpha: {path.name}")
            mask=np.asarray(image.getchannel("A"))>127
        if not mask.any(): raise ValueError(f"Empty object mask: {path.stem}")
        masks[path.stem]=mask
    if not masks: raise ValueError("No segmentation masks")
    if len({m.shape for m in masks.values()})!=1: raise ValueError("Mask dimensions disagree")
    required=set(config.get("required_objects",[]))|set(config.get("objects",{}))
    missing=required-set(masks)
    if missing: raise ValueError(f"Required objects missing masks: {sorted(missing)}")
    floors=[n for n in masks if n.startswith("the_floor") or config.get("objects",{}).get(n,{}).get("role")=="floor"]
    if not floors: raise ValueError("Missing genuine floor segmentation")
    overlaps=[]
    for floor in floors:
        for name,mask in masks.items():
            if name in floors: continue
            iou=float(np.count_nonzero(mask&masks[floor])/np.count_nonzero(mask|masks[floor]))
            overlaps.append({"floor":floor,"object":name,"iou":iou})
            if iou>.95: raise ValueError(f"Floor duplicates object mask {name}: IoU={iou:.6f}")
    tree=json.loads((stage/"scene_tree.json").read_text(encoding="utf-8"))
    roots={"floor","wall","ceiling","floor-wall"}
    nodes=tree.get("nodes",[])
    for name in nodes:
        if not isinstance(name,str) or name not in masks: raise ValueError(f"Scene node has no mask: {name}")
    for edge in tree.get("edges",[]):
        if edge.get("child") not in masks: raise ValueError(f"Relation child has no mask: {edge}")
        if edge.get("parent") not in set(masks)|roots: raise ValueError(f"Relation parent has no mask/root: {edge}")
    return {"status":"passed","mask_pixels":{n:int(m.sum()) for n,m in masks.items()},"required_objects":sorted(required),"floor_overlaps":overlaps,"note":"Structural segmentation checks; semantic accuracy requires visual review"}


def main():
    p=argparse.ArgumentParser();p.add_argument("--scene-root",type=Path,required=True);p.add_argument("--scene-config",type=Path,required=True);p.add_argument("--output",type=Path)
    a=p.parse_args()
    try:report=validate_segmentation(a.scene_root,json.loads(a.scene_config.read_text(encoding="utf-8")))
    except Exception as exc:
        if a.output:
            a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps({"status":"failed","error":str(exc)},indent=2),encoding="utf-8")
        raise
    if a.output:
        a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report))

if __name__=="__main__":main()

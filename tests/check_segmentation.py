"""Run: python tests/check_segmentation.py; no model or GPU required."""
import json
import sys
import tempfile
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
from validate_segmentation import validate_segmentation


def main():
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);stage=root/"stage1";masks=stage/"segemented_obj";masks.mkdir(parents=True)
        object_mask=np.zeros((10,10,4),dtype=np.uint8);object_mask[:3,:,3]=255
        floor_mask=np.zeros_like(object_mask);floor_mask[5:,:,3]=255
        Image.fromarray(object_mask).save(masks/"toy_000.png");Image.fromarray(floor_mask).save(masks/"the_floor.png")
        (stage/"scene_tree.json").write_text(json.dumps({"nodes":["toy_000"],"edges":[{"child":"toy_000","parent":"floor"}]}))
        config={"required_objects":["toy_000"]};assert validate_segmentation(root,config)["status"]=="passed"
        config["required_objects"].append("box_000")
        try:validate_segmentation(root,config)
        except ValueError as exc:assert "missing" in str(exc)
        else:raise AssertionError("Missing object accepted")
        config["required_objects"].remove("box_000")
        Image.fromarray(object_mask).save(masks/"the_floor.png")
        try:validate_segmentation(root,config)
        except ValueError as exc:assert "duplicates" in str(exc)
        else:raise AssertionError("Copied floor accepted")
    print("PASS: valid masks, required-object failure, duplicated-floor failure")

if __name__=="__main__":main()

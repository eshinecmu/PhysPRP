"""Select REST3D objects for local interaction or whole-scene MPM inference."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOTS = {"floor", "wall", "ceiling", "floor-wall"}
FLOORS = {"the_floor", "floor", "ground"}


def select_scope(tree_path, mask_dir, scope, subject=None, margin=0.08):
    if not 0 <= margin <= 1:
        raise ValueError("margin must be between 0 and 1")
    tree = json.loads(Path(tree_path).read_text(encoding="utf-8"))
    masks = {p.stem: p for p in Path(mask_dir).glob("*.png")}
    names = set(masks)
    kinds = {edge["child"]: edge.get("type", "movable")
             for edge in tree.get("edges", []) if edge.get("child") in names}
    if scope == "full":
        return {"scope": scope, "subject": subject, "objects": sorted(names),
                "object_types": {n: kinds[n] for n in sorted(kinds)},
                "reasons": {n: ["all_visible_masks"] for n in sorted(names)}}
    if scope != "interaction" or not subject or subject not in names:
        raise ValueError(f"interaction scope requires an exact subject mask; available: {sorted(names)}")

    parents = {}
    children = {}
    for edge in tree.get("edges", []):
        child, parent = edge.get("child"), edge.get("parent")
        if child not in names:
            continue
        if parent in names and parent not in ROOTS and parent not in FLOORS:
            parents[child] = parent
            children.setdefault(parent, set()).add(child)

    selected = {subject}
    reasons = {subject: ["subject"]}
    # Include the subject's support chain, its dependents, and objects sharing a non-floor support.
    frontier = [subject]
    while frontier:
        current = frontier.pop()
        related = set(children.get(current, ()))
        parent = parents.get(current)
        if parent:
            related.add(parent)
            related.update(children.get(parent, ()))
        for name in related - selected:
            selected.add(name)
            reasons[name] = ["REST3D_non_floor_relation"]
            frontier.append(name)

    # Add conservatively nearby image-space candidates; this is a recall-oriented proxy,
    # not a physical distance estimate. Floor masks never expand the candidate set.
    diagonal = 0.0
    boxes = {}
    for name, path in masks.items():
        if name.lower().startswith("the_floor") or name.lower() in FLOORS:
            continue
        with Image.open(path) as image:
            if "A" not in image.getbands():
                raise ValueError(f"Mask requires alpha: {path.name}")
            mask = np.asarray(image.getchannel("A")) > 127
        y, x = np.where(mask)
        if x.size:
            boxes[name] = (x.min(), y.min(), x.max(), y.max())
            diagonal = max(diagonal, float(np.hypot(mask.shape[1], mask.shape[0])))

    limit = diagonal * float(margin)
    changed = True
    while changed:
        changed = False
        anchors = [n for n in selected if n in boxes]
        for name, box in boxes.items():
            if name in selected:
                continue
            for anchor in anchors:
                other = boxes[anchor]
                dx = max(other[0] - box[2], box[0] - other[2], 0)
                dy = max(other[1] - box[3], box[1] - other[3], 0)
                if np.hypot(dx, dy) <= limit:
                    selected.add(name)
                    reasons[name] = ["nearby_mask_candidate"]
                    changed = True
                    break

    for name in names:
        if name.lower().startswith("the_floor") or name.lower() in FLOORS:
            selected.add(name)
            reasons[name] = ["scene_contact_surface"]
    return {"scope": scope, "subject": subject, "objects": sorted(selected),
            "object_types": {n: kinds[n] for n in sorted(selected) if n in kinds},
            "reasons": reasons, "image_margin_fraction": float(margin),
            "spatial_rule": "2D mask-box proximity; candidates require review"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--scope", choices=("interaction", "full"), required=True)
    parser.add_argument("--subject")
    parser.add_argument("--margin", type=float, default=0.08)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = select_scope(args.tree, args.mask_dir, args.scope, args.subject, args.margin)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"scope": result["scope"], "selected": result["objects"]}))


if __name__ == "__main__":
    main()

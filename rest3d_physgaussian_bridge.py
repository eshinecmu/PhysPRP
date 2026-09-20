"""Convert REST3D scene outputs into PhysGaussian mesh and relation manifests."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_tree", required=True)
    parser.add_argument("--mesh_dir", required=True)
    parser.add_argument("--manifest_out", required=True)
    parser.add_argument("--relations_out", required=True)
    args = parser.parse_args()

    tree = json.loads(Path(args.scene_tree).read_text(encoding="utf-8"))
    mesh_dir = Path(args.mesh_dir)
    meshes = sorted(mesh_dir.glob("scene_canon_*.obj"))
    if not meshes:
        meshes = sorted(mesh_dir.glob("*.obj"))
    if not meshes:
        raise FileNotFoundError(f"No REST3D OBJ meshes in {mesh_dir}")

    tree_objects = {
        str(obj.get("name", ""))
        for obj in tree.get("objects", [])
        if obj.get("name")
    }

    rows = []
    name_to_id = {}
    for object_id, mesh in enumerate(meshes):
        name = mesh.stem.removeprefix("scene_canon_")
        if not name:
            raise ValueError(f"Could not derive an object name from {mesh}")
        if name in name_to_id:
            raise ValueError(f"Duplicate REST3D object name: {name}")
        if tree_objects and name not in tree_objects:
            raise ValueError(f"Mesh {name!r} is missing from the REST3D scene tree")
        name_to_id[name] = object_id
        rows.append({"path": str(mesh.resolve()), "object_id": object_id, "name": name})

    relations = []
    skipped = []
    for edge in tree.get("edges", []):
        child, parent = edge.get("child"), edge.get("parent")
        if child not in name_to_id or parent not in name_to_id:
            skipped.append(edge)
            continue
        relations.append({
            "source": name_to_id[child],
            "target": name_to_id[parent],
            "relation": edge.get("relation", "other"),
            "source_name": child,
            "target_name": parent,
        })

    Path(args.manifest_out).write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    Path(args.relations_out).write_text(json.dumps({"edges": relations, "skipped_roots": skipped}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {len(rows)} objects and {len(relations)} object relations")


if __name__ == "__main__":
    main()

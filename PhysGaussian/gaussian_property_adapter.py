import argparse
import json
import math
import re
from pathlib import Path


DEFAULTS = {
    "wood": (8e8, .30, 700), "metal": (5e9, .28, 7800),
    "plastic": (8e7, .40, 950), "glass": (6e9, .22, 2500),
    "fabric": (2e6, .45, 300), "foam": (5e5, .45, 80),
    "food": (2e5, .45, 900), "ceramic": (8e9, .22, 2200),
    "paper": (1e8, .30, 700), "leather": (5e6, .45, 900),
}
# Ordered checkpoint vocabulary; never derive embedding IDs from scene contents.
MATERIAL_VOCAB = ("unknown", *DEFAULTS)
MATERIAL_LIMITS = {"E": (1e4, 1e7), "nu": (.05, .45), "density": (50., 3000.)}


def material_parameters(record):
    """Validate physical input and expose the solver's numerical clipping explicitly."""
    requested = {name: float(record[name]) for name in MATERIAL_LIMITS}
    if not all(math.isfinite(v) for v in requested.values()):
        raise ValueError("material values must be finite")
    if requested["E"] <= 0 or requested["density"] <= 0 or not 0 <= requested["nu"] < .5:
        raise ValueError("require E > 0, density > 0, and 0 <= nu < 0.5")
    actual = {name: min(max(requested[name], low), high)
              for name, (low, high) in MATERIAL_LIMITS.items()}
    return {"requested": requested, "actual": actual,
            "clamped": [name for name in requested if requested[name] != actual[name]]}


def particle_material_fields(records, object_id):
    import torch

    fields = {
        name: torch.empty(
            object_id.shape, device=object_id.device, dtype=torch.float32
        )
        for name in MATERIAL_LIMITS
    }

    fields["material_id"] = torch.zeros_like(object_id)
    audits = []

    for oid in torch.unique(object_id).tolist():
        record = records[int(oid)]
        audit = material_parameters(record)
        mask = object_id == oid

        for name in MATERIAL_LIMITS:
            fields[name][mask] = audit["actual"][name]

        fields["material_id"][mask] = MATERIAL_VOCAB.index(
            str(record.get("material", "unknown"))
        )
        audits.append(dict(object_id=int(oid), **audit))

    return fields, audits


def normalize(value):
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def parse_predictions(text, manifest):
    names = {normalize(str(row.get("name", ""))): int(row["object_id"]) for row in manifest}
    expected = {int(row["object_id"]) for row in manifest}
    if len(expected) != len(manifest):
        raise ValueError("Duplicate object IDs in manifest")
    records = []
    seen = set()
    for line in text.splitlines():
        if not line.strip():
            continue
        # GP emits unquoted comma-separated captions; split material fields from the right.
        filename, separator, response = line.partition(",")
        parts = [item.strip() for item in response.strip().strip("()").rsplit(",", 3)]
        if not separator or len(parts) != 4:
            raise ValueError(f"Malformed GaussianProperty prediction: {filename}")
        stem = Path(filename.strip()).stem
        if stem.isdigit():
            object_id = int(stem)
        else:
            matches = [oid for name, oid in names.items() if name and name in normalize(stem)]
            if len(matches) != 1:
                raise ValueError(f"Cannot map GaussianProperty result to one REST3D object: {filename}")
            object_id = matches[0]
        if object_id not in expected or object_id in seen:
            raise ValueError(f"Unexpected or duplicate GaussianProperty object ID: {object_id}")
        seen.add(object_id)
        raw_material = parts[1]
        material = normalize(raw_material)
        if material not in DEFAULTS:
            material = "unknown"
        fallback = "plastic" if material == "unknown" else None
        hardness = re.fullmatch(r"(?:hardness\s*)?(\d+(?:\.\d+)?)(?:\s*[-–]\s*(\d+(?:\.\d+)?))?", parts[2], re.I)
        if not hardness:
            raise ValueError(f"Invalid hardness for object {object_id}: {parts[2]}")
        low = float(hardness[1]); high = float(hardness[2] or hardness[1])
        shore = parts[3].rstrip(".。 ").strip("<> ").title()
        if not 0 <= low <= high <= 100 or shore not in ("Shore A", "Shore D"):
            raise ValueError(f"Invalid Shore hardness for object {object_id}")
        E, nu, density = DEFAULTS[fallback or material]
        record = {"object_id": object_id, "caption": parts[0], "material": material,
                  "material_id": MATERIAL_VOCAB.index(material), "material_label_raw": raw_material,
                  "fallback_material": fallback, "hardness_range": [low, high], "shore": shore,
                  "E": E, "nu": nu, "density": density,
                  "parameter_source": "semantic_material_lookup", "is_measured": False,
                  "hardness_used_for_parameters": False}
        record["material_parameters"] = material_parameters(record)
        records.append(record)
    missing = sorted(expected - seen)
    if missing:
        raise ValueError(f"Missing GaussianProperty predictions for object IDs: {missing}")
    return sorted(records, key=lambda row: row["object_id"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions")
    parser.add_argument("output")
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    records = parse_predictions(Path(args.predictions).read_text(encoding="utf-8"), manifest)
    Path(args.output).write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"wrote {len(records)} per-object property records to {args.output}")


if __name__ == "__main__":
    main()

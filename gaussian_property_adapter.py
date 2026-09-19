import argparse
import json
import re
from pathlib import Path


DEFAULTS = {
    "wood": (8e8, .30, 700), "metal": (5e9, .28, 7800),
    "plastic": (8e7, .40, 950), "glass": (6e9, .22, 2500),
    "fabric": (2e6, .45, 300), "foam": (5e5, .45, 80),
    "food": (2e5, .45, 900), "ceramic": (8e9, .22, 2200),
    "paper": (1e8, .30, 700), "leather": (5e6, .45, 900),
}


def normalize(value):
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions")
    parser.add_argument("output")
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    names = {normalize(str(row.get("name", ""))): int(row["object_id"]) for row in manifest}
    records = []
    for line in Path(args.predictions).read_text(encoding="utf-8").splitlines():
        parts = [item.strip() for item in line.split(",")]
        if len(parts) < 5:
            continue
        stem = Path(parts[0]).stem
        if stem.isdigit():
            object_id = int(stem)
        else:
            normalized_stem = normalize(stem)
            matches = [object_id for name, object_id in names.items() if name and name in normalized_stem]
            if len(matches) != 1:
                raise ValueError(f"Cannot map GaussianProperty result to one REST3D object: {parts[0]}")
            object_id = matches[0]
        material = next((key for key in DEFAULTS if key in parts[-3].lower()), "plastic")
        numbers = re.findall(r"[0-9]+(?:\.[0-9]+)?", parts[-2])
        low, high = (numbers + ["0", "0"])[:2]
        E, nu, density = DEFAULTS[material]
        records.append({"object_id": object_id, "caption": parts[-4], "material": material,
                        "hardness_range": [float(low), float(high)], "shore": parts[-1],
                        "E": E, "nu": nu, "density": density})
    seen = {record["object_id"] for record in records}
    expected = {int(row["object_id"]) for row in manifest}
    missing = sorted(expected - seen)
    if missing:
        raise ValueError(f"Missing GaussianProperty predictions for object IDs: {missing}")
    Path(args.output).write_text(json.dumps(sorted(records, key=lambda row: row["object_id"]), indent=2), encoding="utf-8")
    print(f"wrote {len(records)} per-object property records to {args.output}")


if __name__ == "__main__":
    main()

"""Validate an LLM/VLM material prior and write a PhysGaussian config."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


MODELS = {"jelly", "metal", "sand", "foam", "snow", "plasticine"}


def value(prior: dict, name: str) -> float:
    raw = prior[name]
    return float(raw["value"] if isinstance(raw, dict) else raw)


def validate(prior: dict) -> dict:
    model = prior.get("physgaussian_model")
    if model not in MODELS:
        raise ValueError(f"physgaussian_model must be one of {sorted(MODELS)}")
    result = {"material": model, "E": value(prior, "young_modulus_pa"), "nu": value(prior, "poisson_ratio"), "density": value(prior, "density_kg_m3")}
    if not all(math.isfinite(v) for v in result.values() if isinstance(v, (int, float))):
        raise ValueError("material values must be finite")
    if result["E"] <= 0 or result["density"] <= 0 or not 0 <= result["nu"] < 0.5:
        raise ValueError("require E > 0, density > 0, and 0 <= nu < 0.5")
    for name in ("young_modulus_pa", "poisson_ratio", "density_kg_m3"):
        raw = prior[name]
        if isinstance(raw, dict) and "range" in raw:
            low, high = map(float, raw["range"])
            if not low <= result[{"young_modulus_pa": "E", "poisson_ratio": "nu", "density_kg_m3": "density"}[name]] <= high:
                raise ValueError(f"{name}.value is outside its declared range")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior", type=Path, required=True, help="LLM/VLM JSON output")
    parser.add_argument("--template", type=Path, required=True, help="Existing PhysGaussian config")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prior = json.loads(args.prior.read_text(encoding="utf-8"))
    config = json.loads(args.template.read_text(encoding="utf-8"))
    config.update(validate(prior))
    config["material_prior"] = prior
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

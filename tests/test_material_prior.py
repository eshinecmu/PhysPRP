import json
import subprocess
import sys
from pathlib import Path


def test_material_prior_adapter(tmp_path):
    root = Path(__file__).parents[1]
    output = tmp_path / "config.json"
    subprocess.run([
        sys.executable, "tools/apply_material_prior.py",
        "--prior", "config/material_prior.example.json",
        "--template", "config/rest3d_soft_object_jelly.json",
        "--output", str(output),
    ], cwd=root, check=True)
    config = json.loads(output.read_text(encoding="utf-8"))
    assert config["material"] == "jelly"
    assert config["E"] == 100000.0
    assert config["material_prior"]["is_measured"] is False

"""Load one static RGB frame from a PAC-NeRF multi-view sequence."""

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def _view_frame(file_path: str) -> tuple[int, int]:
    try:
        _, view_id, frame_id = Path(file_path).stem.rsplit("_", 2)
        return int(view_id), int(frame_id)
    except ValueError as exc:
        raise ValueError(f"Invalid PAC-NeRF file_path: {file_path!r}") from exc


class PACNeRFStaticRGB(Dataset):
    """A one-item dataset for a selected camera view and video frame.

    PAC-NeRF reserves frame -1 for the background and -2 for the mask, so only
    non-negative video frame IDs are accepted here.
    """

    def __init__(self, scene_dir: str | Path, view_id: int = 0, frame_id: int = 0):
        if view_id < 0 or frame_id < 0:
            raise ValueError("view_id and frame_id must be non-negative")

        self.scene_dir = Path(scene_dir).resolve()
        metadata_path = self.scene_dir / "all_data.json"
        with metadata_path.open(encoding="utf-8") as file:
            entries = json.load(file)

        self.entry = next(
            (entry for entry in entries if _view_frame(entry["file_path"]) == (view_id, frame_id)),
            None,
        )
        if self.entry is None:
            raise IndexError(f"PAC-NeRF frame not found: view={view_id}, frame={frame_id}")

        self.image_path = (self.scene_dir / self.entry["file_path"]).resolve()
        try:
            self.image_path.relative_to(self.scene_dir)
        except ValueError as exc:
            raise ValueError("Image path escapes the PAC-NeRF scene directory") from exc

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> dict:
        if index not in (0, -1):
            raise IndexError(index)

        rgb = np.asarray(Image.open(self.image_path).convert("RGB"), dtype=np.float32).copy() / 255.0
        view_id, frame_id = _view_frame(self.entry["file_path"])
        return {
            "rgb": torch.from_numpy(rgb).permute(2, 0, 1),
            "view_id": view_id,
            "frame_id": frame_id,
            "c2w": torch.tensor(self.entry["c2w"], dtype=torch.float32),
            "intrinsic": torch.tensor(self.entry["intrinsic"], dtype=torch.float32),
            "image_path": str(self.image_path),
        }

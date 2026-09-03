import json
import sys
import tempfile
from pathlib import Path
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parents[1]))
from dataloader.pac_nerf import PACNeRFStaticRGB


class PACNeRFStaticRGBTest(unittest.TestCase):
    def test_selects_requested_view_and_frame(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "data").mkdir()
            Image.fromarray(np.full((2, 3, 4), [64, 128, 255, 7], dtype=np.uint8), "RGBA").save(
                root / "data" / "r_2_4.png"
            )
            (root / "all_data.json").write_text(
                json.dumps([{
                    "file_path": "data/r_2_4.png",
                    "c2w": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
                    "intrinsic": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                }]),
                encoding="utf-8",
            )

            sample = PACNeRFStaticRGB(root, view_id=2, frame_id=4)[0]
            self.assertEqual(sample["rgb"].shape, (3, 2, 3))
            self.assertEqual((sample["view_id"], sample["frame_id"]), (2, 4))
            self.assertAlmostEqual(sample["rgb"][0, 0, 0].item(), 64 / 255)


if __name__ == "__main__":
    unittest.main()

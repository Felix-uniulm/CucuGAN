"""Maskenvertrag ohne Download von Modellgewichten prüfen."""

import sys
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from face_mask import skin_alpha
from infer import convert_image


class FaceMaskTests(unittest.TestCase):
    def test_only_skin_and_nose_even_with_feathering(self):
        labels = torch.arange(19).repeat(19, 1)
        for radius in (0, 1, 3):
            mask = skin_alpha(labels, [1, 2], radius)
            self.assertTrue((mask[:, 1:3] > 0).all())
            self.assertEqual(mask[:, 3:].count_nonzero().item(), 0)
            self.assertEqual(mask[:, 0].count_nonzero().item(), 0)

    def test_compositing_keeps_excluded_pixels(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, dest = Path(tmp) / "in.png", Path(tmp) / "out.png"
            Image.new("RGB", (128, 128), (30, 60, 90)).save(source)
            mask = torch.zeros(1, 1, 128, 128)
            mask[:, :, 40:80, 40:80] = 1
            convert_image(lambda x: torch.ones_like(x), source, dest,
                          torch.device("cpu"), lambda image: mask, True)
            with Image.open(dest) as result:
                self.assertEqual(result.getpixel((0, 0)), (30, 60, 90))
                self.assertEqual(result.getpixel((50, 50)), (255, 255, 255))
            self.assertTrue((Path(tmp) / "out_mask.png").is_file())


if __name__ == "__main__":
    unittest.main()

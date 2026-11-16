import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dataset import UnpairedDataset, mirror_tiles
from models import Discriminator
from utils import masked_l1


class SkinTrainingTests(unittest.TestCase):
    def test_outside_loss_normalization_and_gradient(self):
        fake = torch.ones(2, 3, 8, 8, requires_grad=True)
        mask = torch.zeros(2, 1, 8, 8)
        mask[0, :, :2] = 1
        mask[1, :, :6] = 1
        loss = masked_l1(fake, torch.zeros_like(fake), 1 - mask)
        self.assertAlmostEqual(loss.item(), 1)
        loss.backward()
        self.assertEqual((fake.grad * mask).count_nonzero().item(), 0)
        self.assertGreater((fake.grad * (1 - mask)).sum().item(), 0)

    def test_texture_gradient_cannot_change_excluded_pixels(self):
        torch.set_num_threads(2)
        fake = torch.randn(1, 3, 64, 64, requires_grad=True)
        mask = torch.zeros(1, 1, 64, 64)
        mask[:, :, 16:48, 16:48] = 1
        discriminator = Discriminator(ndf=4)
        (discriminator(fake * mask) - 1).square().mean().backward()
        self.assertEqual((fake.grad * (1 - mask)).count_nonzero().item(), 0)
        self.assertGreater((fake.grad * mask).abs().sum().item(), 0)

    def test_mirror_tiles_geometry(self):
        array = np.arange(64, dtype=np.uint8).reshape(8, 8)
        image = Image.fromarray(array).convert("RGB")
        output = np.array(mirror_tiles(image))
        self.assertEqual(output.shape, (8, 8, 3))
        np.testing.assert_array_equal(output, output[:, ::-1])
        np.testing.assert_array_equal(output, output[::-1])
        np.testing.assert_array_equal(np.array(mirror_tiles(image, 1)), np.array(image))

    def test_dataset_flips_mask_with_face(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for folder in ("A", "B", "masks_A"):
                (root / "train" / folder).mkdir(parents=True)
            data = np.zeros((32, 32), dtype=np.uint8)
            data[:, :12] = 255
            Image.fromarray(data).convert("RGB").save(root / "train/A/face.png")
            Image.fromarray(data).save(root / "train/masks_A/face.png.png")
            Image.new("RGB", (32, 32)).save(root / "train/B/skin.png")
            dataset = UnpairedDataset(root, skin_loss=True, texture_tiles=2)
            with patch("dataset.random.random", return_value=0):
                sample = dataset[0]
            torch.testing.assert_close((sample["A"][:1] + 1) / 2, sample["mask_A"])
            self.assertEqual(sample["mask_A"][0, 0, -1].item(), 1)
            (root / "train/masks_A/face.png.png").unlink()
            with self.assertRaises(FileNotFoundError):
                UnpairedDataset(root, skin_loss=True)



if __name__ == "__main__":
    unittest.main()

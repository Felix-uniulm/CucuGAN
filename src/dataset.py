"""Ungepaartes Dataset für CycleGAN: zu jedem Gesicht (A) eine Gurke (B).

Erwartete Ordnerstruktur (erzeugt von scripts/prepare_data.py):
    <root>/<split>/A/*.png   Gesichter
    <root>/<split>/B/*.png   Gurken

Selbsttest (speichert einen Batch als Bild):
    python src/dataset.py --root data/processed/squares128
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image

from utils import denormalize

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


class UnpairedDataset(Dataset):
    def __init__(self, root: str | Path, split: str = "train", max_images: int | None = None,
                 skin_loss: bool = False, texture_tiles: int = 1):
        folder = Path(root) / split
        # max_images: nur die ersten N Bilder pro Domäne (für den Overfit-Test)
        self.paths_a = list_images(folder / "A")[:max_images]
        self.paths_b = list_images(folder / "B")[:max_images]
        if not self.paths_a or not self.paths_b:
            raise FileNotFoundError(f"Keine Bilder in {folder}/A oder {folder}/B")
        self.train = split == "train"
        self.mask_folder = folder / "masks_A"
        self.skin_loss = skin_loss
        if texture_tiles not in (1, 2):
            raise ValueError("texture_tiles muss 1 oder 2 sein")
        self.texture_tiles = texture_tiles
        if skin_loss:
            for path in self.paths_a:
                mask = self.mask_folder / (path.name + ".png")
                if not mask.is_file():
                    raise FileNotFoundError(f"Hautmaske fehlt: {mask}; scripts/prepare_masks.py ausführen")

        steps = [
            transforms.ToTensor(),                                   # [0, 255] -> [0, 1], HxWxC -> CxHxW
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),  # [0, 1] -> [-1, 1]
        ]
        self.transform = transforms.Compose(steps)

    def __len__(self) -> int:
        return len(self.paths_a)

    def __getitem__(self, index: int) -> dict:
        path_a = self.paths_a[index]
        if self.train:
            path_b = random.choice(self.paths_b)                      # ungepaart: zufällige Gurke
        else:
            path_b = self.paths_b[index % len(self.paths_b)]          # test: reproduzierbar
        a = self._load(path_a)
        with Image.open(path_b) as image:
            b = self.transform(mirror_tiles(image.convert("RGB"), self.texture_tiles))
        result = {"A": a, "B": b}
        if self.skin_loss:
            with Image.open(self.mask_folder / (path_a.name + ".png")) as image:
                mask = transforms.ToTensor()(image.convert("L"))
            if mask.shape[-2:] != a.shape[-2:]:
                raise ValueError(f"Maskengröße passt nicht zu {path_a}")
            mask = (mask >= 0.5).float()
            if not mask.any() or mask.all():
                raise ValueError(f"Leere oder vollflächige Hautmaske für {path_a}")
            result["mask_A"] = mask
        if self.train and random.random() < 0.5:
            result["A"] = result["A"].flip(-1)
            if self.skin_loss:
                result["mask_A"] = result["mask_A"].flip(-1)
        if self.train and random.random() < 0.5:
            result["B"] = result["B"].flip(-1)
        return result

    def _load(self, path: Path):
        with Image.open(path) as img:
            return self.transform(img.convert("RGB"))


def mirror_tiles(image: Image.Image, count: int = 2) -> Image.Image:
    """Gespiegeltes 2×2-Raster auf Originalgröße verkleinern; keine Split-Mischung."""
    if count == 1:
        return image
    if count != 2:
        raise ValueError("Nur 1×1 oder 2×2 unterstützt")
    w, h = image.size
    canvas = Image.new("RGB", (2 * w, 2 * h))
    canvas.paste(image, (0, 0))
    canvas.paste(ImageOps.mirror(image), (w, 0))
    canvas.paste(ImageOps.flip(image), (0, h))
    canvas.paste(ImageOps.flip(ImageOps.mirror(image)), (w, h))
    return canvas.resize(image.size, Image.Resampling.LANCZOS)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Einen Batch laden und als Bild speichern")
    parser.add_argument("--root", default="data/processed/squares128")
    parser.add_argument("--split", default="train")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--out", default="outputs/batch_check.png")
    args = parser.parse_args()

    ds = UnpairedDataset(args.root, args.split)
    print(f"{args.split}: {len(ds.paths_a)} Gesichter, {len(ds.paths_b)} Gurken, len(ds) = {len(ds)}")

    batch = next(iter(DataLoader(ds, batch_size=args.batch_size, shuffle=True)))
    a, b = batch["A"], batch["B"]
    print(f"A: {tuple(a.shape)}, Werte {a.min():.2f} bis {a.max():.2f}")
    print(f"B: {tuple(b.shape)}, Werte {b.min():.2f} bis {b.max():.2f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save_image(denormalize(torch.cat([a, b])), args.out, nrow=args.batch_size)
    print(f"Gespeichert: {args.out} (obere Reihe A, untere Reihe B)")

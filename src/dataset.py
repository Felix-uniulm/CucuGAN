"""Ungepaartes Dataset für CycleGAN: zu jedem Gesicht (A) eine Gurke (B).

Erwartete Ordnerstruktur (erzeugt von scripts/prepare_data.py):
    <root>/<split>/A/*.png   Gesichter
    <root>/<split>/B/*.png   Gurken

Selbsttest (speichert einen Batch als Bild):
    python src/dataset.py --root data/processed/128
"""

import argparse
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image

from utils import denormalize

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


class UnpairedDataset(Dataset):
    def __init__(self, root: str | Path, split: str = "train", max_images: int | None = None):
        folder = Path(root) / split
        # max_images: nur die ersten N Bilder pro Domäne (für den Overfit-Test)
        self.paths_a = list_images(folder / "A")[:max_images]
        self.paths_b = list_images(folder / "B")[:max_images]
        if not self.paths_a or not self.paths_b:
            raise FileNotFoundError(f"Keine Bilder in {folder}/A oder {folder}/B")
        self.train = split == "train"

        steps = [transforms.RandomHorizontalFlip()] if self.train else []
        steps += [
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
        return {"A": self._load(path_a), "B": self._load(path_b)}

    def _load(self, path: Path):
        with Image.open(path) as img:
            return self.transform(img.convert("RGB"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Einen Batch laden und als Bild speichern")
    parser.add_argument("--root", default="data/processed/128")
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

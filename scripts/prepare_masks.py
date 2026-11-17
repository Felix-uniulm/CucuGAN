"""Face-Parsing-Masken einmalig neben A-Bildern speichern, ohne Training des Parsers."""

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision.utils import save_image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dataset import list_images
from face_mask import DEFAULT_MODEL, MASK_FOLDER, SKIN_CLASSES, FaceMask
from infer import pick_device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/processed/squares128"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--mask-folder", default=MASK_FOLDER)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-images", type=int, help="Nur für kleine Funktionstests, pro Split")
    parser.add_argument("--threads", type=int, default=4, help="CPU-Threads")
    args = parser.parse_args()
    if args.threads < 1 or (args.max_images is not None and args.max_images < 1):
        parser.error("--threads und --max-images müssen positiv sein")
    torch.set_num_threads(args.threads)
    model = None  # Kein Modell laden, wenn bereits alle Masken vorhanden sind.
    failures = []
    for split in ("train", "test"):
        folder = args.data / split / args.mask_folder
        folder.mkdir(parents=True, exist_ok=True)
        metadata = {"model": args.model, "classes": list(SKIN_CLASSES), "feather": 0}
        metadata_path = folder / "config.json"
        if metadata_path.exists() and json.loads(metadata_path.read_text()) != metadata:
            raise ValueError(f"Andere Maskenkonfiguration in {folder}; separaten Datensatz verwenden")
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
        for path in tqdm(list_images(args.data / split / "A")[:args.max_images], desc=split):
            destination = folder / (path.name + ".png")
            if destination.exists():
                continue
            if model is None:
                model = FaceMask(pick_device(args.device), args.model, feather=0)
            with Image.open(path) as image:
                try:
                    mask = model(image)
                    if mask.all():
                        raise ValueError("Vollflächige Maske")
                except ValueError as exc:
                    failures.append(f"{path}: {exc}")
                    continue
            temporary = destination.with_suffix(".tmp.png")
            save_image(mask, temporary)
            temporary.replace(destination)
    if failures:
        raise RuntimeError("Maskenerstellung fehlgeschlagen; Bilder prüfen:\n" + "\n".join(failures))
    print("Masken gespeichert. Vorhandene Dateien wurden übersprungen.")


if __name__ == "__main__":
    main()

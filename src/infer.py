"""Gesichter mit einem trainierten CycleGAN-Generator in Gurkengesichter verwandeln.

Für den aktuellen Stand werden quadratische 128x128-Bilder erwartet:

    python src/infer.py \
        --checkpoint checkpoints/baseline_squares_2k/latest.pt \
        --input input/face.png \
        --output outputs/inference/face_cucumber.png

Ein Ordner als Eingabe wird ebenfalls unterstützt:

    python src/infer.py \
        --checkpoint checkpoints/baseline_squares_2k/latest.pt \
        --input input/faces \
        --output outputs/inference
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

from models import Generator

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Checkpoint, z. B. checkpoints/name/latest.pt")
    parser.add_argument("--input", type=Path, required=True,
                        help="128x128-Bild oder Ordner mit solchen Bildern")
    parser.add_argument("--output", type=Path, required=True,
                        help="Ausgabedatei oder Ausgabeordner")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda oder mps")
    return parser.parse_args()


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def image_paths(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        paths = sorted(p for p in input_path.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
        if not paths:
            raise FileNotFoundError(f"Keine unterstützten Bilder in {input_path}")
        return paths
    raise FileNotFoundError(f"Eingabe nicht gefunden: {input_path}")


def load_generator(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint nicht gefunden: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    try:
        config = checkpoint["config"]
        state_dict = checkpoint["nets"]["G_AB"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"Ungültiger CycleGAN-Checkpoint: {checkpoint_path} "
            "(erwartet werden config und nets.G_AB)"
        ) from exc

    generator = Generator(
        ngf=config["ngf"],
        n_downsampling=config["n_downsampling"],
        n_res_blocks=config["n_res_blocks"],
    )
    generator.load_state_dict(state_dict)
    generator.to(device)
    generator.eval()
    return generator


def output_path(output: Path, input_path: Path, multiple: bool) -> Path:
    if multiple or output.is_dir() or output.suffix == "":
        return output / f"{input_path.stem}_cucumber.png"
    return output


def transform_image(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB")
        if image.size != (128, 128):
            raise ValueError(
                f"{path} hat die Größe {image.size}; erwartet wird 128x128."
            )
        tensor = transforms.ToTensor()(image)
    return transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(tensor).unsqueeze(0)


def convert_image(generator: torch.nn.Module, input_path: Path,
                  output_path_: Path, device: torch.device) -> None:
    image = transform_image(input_path).to(device)
    with torch.inference_mode():
        result = generator(image)
    output_path_.parent.mkdir(parents=True, exist_ok=True)
    save_image(((result + 1) / 2).clamp(0, 1), output_path_)


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    inputs = image_paths(args.input)
    generator = load_generator(args.checkpoint, device)
    multiple = len(inputs) > 1

    for input_path in inputs:
        destination = output_path(args.output, input_path, multiple)
        convert_image(generator, input_path, destination, device)
        print(f"{input_path} -> {destination}")


if __name__ == "__main__":
    main()

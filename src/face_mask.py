"""Hautmaske aus SegFormer Face Parsing (optionale transformers-Abhängigkeit)."""

import numpy as np
import torch
from PIL import Image, ImageFilter
from torch.nn import functional as F

DEFAULT_MODEL = "jonathandinu/face-parsing"


def skin_alpha(labels: torch.Tensor, skin_ids: list[int], feather: float) -> torch.Tensor:
    """Weiche Übergänge ausschließlich innerhalb der ausgewählten Hautfläche."""
    if feather < 0:
        raise ValueError("Maskenradius muss >= 0 sein")
    selected = torch.zeros_like(labels, dtype=torch.bool)
    for label_id in skin_ids:
        selected |= labels == label_id
    hard = selected.float()
    if feather == 0:
        return hard
    image = Image.fromarray((hard.cpu().numpy() * 255).astype("uint8"))
    blurred = image.filter(ImageFilter.GaussianBlur(feather))
    soft = torch.tensor(np.array(blurred), dtype=torch.float32,
                        device=labels.device) / 255
    # Kein Einbluten in Augen, Lippen, Haare oder Hintergrund.
    return soft * hard


class FaceMask:
    def __init__(self, device: torch.device, model: str = DEFAULT_MODEL,
                 feather: float = 1.0):
        from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

        if feather < 0:
            raise ValueError("Maskenradius muss >= 0 sein")
        self.device = device
        self.feather = feather
        self.processor = SegformerImageProcessor.from_pretrained(model)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model).to(device).eval()
        labels = {name.lower(): int(i) for i, name in self.model.config.id2label.items()}
        if not {"skin", "nose"} <= labels.keys():
            raise ValueError("Face-Parsing-Modell muss Klassen 'skin' und 'nose' enthalten")
        self.skin_ids = [labels["skin"], labels["nose"]]

    @torch.inference_mode()
    def __call__(self, image: Image.Image) -> torch.Tensor:
        inputs = self.processor(images=image.convert("RGB"), return_tensors="pt").to(self.device)
        logits = self.model(**inputs).logits
        labels = F.interpolate(logits, size=(image.height, image.width),
                               mode="bilinear", align_corners=False).argmax(1)[0]
        mask = skin_alpha(labels, self.skin_ids, self.feather)
        if not mask.any():
            raise ValueError("Keine Gesichtshaut erkannt; Bildausschnitt oder Maske prüfen")
        return mask[None, None]

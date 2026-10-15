"""Hilfsfunktionen für CucuGAN."""

import random
from pathlib import Path

import torch
from torch import nn
from torchvision.utils import save_image


def denormalize(x: torch.Tensor) -> torch.Tensor:
    """Bilder von [-1, 1] (Netz) zurück nach [0, 1] (Anzeigen, Speichern)."""
    return ((x + 1) / 2).clamp(0, 1)


class ImagePool:
    """Speicher für alte Fakes (Shrivastava et al. 2017).

    Der Diskriminator bekommt mit 50 % Wahrscheinlichkeit nicht das neueste Fake,
    sondern ein älteres aus dem Speicher. So "vergisst" er frühere Fehler des
    Generators nicht, und das Training schwingt weniger hin und her.
    """

    def __init__(self, size: int = 50):
        self.size = size
        self.images: list[torch.Tensor] = []

    def query(self, images: torch.Tensor) -> torch.Tensor:
        if self.size == 0:
            return images
        out = []
        for img in images.detach():
            img = img.unsqueeze(0)
            if len(self.images) < self.size:       # Speicher noch nicht voll: merken und benutzen
                self.images.append(img)
                out.append(img)
            elif random.random() < 0.5:            # altes Bild ausgeben, neues dafür merken
                i = random.randrange(self.size)
                out.append(self.images[i])
                self.images[i] = img
            else:                                  # neues Bild direkt benutzen
                out.append(img)
        return torch.cat(out)


def linear_decay(n_epochs: int, n_epochs_decay: int):
    """Faktor für die Lernrate: n_epochs lang 1, danach linear auf 0 (CycleGAN-Paper)."""
    def factor(epoch: int) -> float:
        return 1.0 - max(0, epoch + 1 - n_epochs) / (n_epochs_decay + 1)
    return factor


def set_requires_grad(nets: list[nn.Module], requires_grad: bool) -> None:
    """Gradienten für Netze ein- oder ausschalten (spart Rechenzeit, verhindert falsche Updates)."""
    for net in nets:
        for p in net.parameters():
            p.requires_grad = requires_grad


@torch.no_grad()
def sample_grid(G_AB: nn.Module, G_BA: nn.Module, real_A: torch.Tensor, real_B: torch.Tensor) -> torch.Tensor:
    """Kontrollbild: Zeilen echt A, fake B, rekonstruiert A, echt B, fake A, rekonstruiert B."""
    was_training = G_AB.training
    G_AB.eval()
    G_BA.eval()
    fake_B = G_AB(real_A)
    fake_A = G_BA(real_B)
    rows = [real_A, fake_B, G_BA(fake_B), real_B, fake_A, G_AB(fake_A)]
    if was_training:
        G_AB.train()
        G_BA.train()
    return denormalize(torch.cat(rows))


def save_grid(grid: torch.Tensor, path: Path, nrow: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, path, nrow=nrow)

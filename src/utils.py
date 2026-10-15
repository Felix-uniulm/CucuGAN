"""Hilfsfunktionen für CucuGAN."""

import torch


def denormalize(x: torch.Tensor) -> torch.Tensor:
    """Bilder von [-1, 1] (Netz) zurück nach [0, 1] (Anzeigen, Speichern)."""
    return ((x + 1) / 2).clamp(0, 1)

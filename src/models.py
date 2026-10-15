"""Netze für CycleGAN: ResNet-Generator (Johnson et al. 2016) und 70x70-PatchGAN-Diskriminator.

Alle Größen (Kanäle, Anzahl Blöcke/Schichten) kommen als Parameter, nichts ist auf eine
Bildgröße festgelegt.

Selbsttest (Shapes für mehrere Auflösungen, Parameterzahlen):
    python src/models.py
"""

import torch
from torch import nn


class ResidualBlock(nn.Module):
    """x + f(x): der Block lernt nur die Änderung am Bild und reicht den Rest durch."""

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class Generator(nn.Module):
    """Encoder -> Residual-Blöcke -> Decoder. Ausgabe hat dieselbe Größe wie die Eingabe."""

    def __init__(self, in_channels: int = 3, out_channels: int = 3, ngf: int = 64,
                 n_downsampling: int = 2, n_res_blocks: int = 6):
        super().__init__()
        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, ngf, kernel_size=7),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True),
        ]

        # Encoder: jede Stufe halbiert die Auflösung und verdoppelt die Kanäle
        channels = ngf
        for _ in range(n_downsampling):
            layers += [
                nn.Conv2d(channels, channels * 2, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(channels * 2),
                nn.ReLU(inplace=True),
            ]
            channels *= 2

        # Transformer: hier passiert die eigentliche Übersetzung, bei kleiner Auflösung
        layers += [ResidualBlock(channels) for _ in range(n_res_blocks)]

        # Decoder: spiegelbildlich zum Encoder
        for _ in range(n_downsampling):
            layers += [
                nn.ConvTranspose2d(channels, channels // 2, kernel_size=3, stride=2,
                                   padding=1, output_padding=1),
                nn.InstanceNorm2d(channels // 2),
                nn.ReLU(inplace=True),
            ]
            channels //= 2

        layers += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(channels, out_channels, kernel_size=7),
            nn.Tanh(),  # Ausgabe in [-1, 1], passend zur Normalisierung der Bilder
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class Discriminator(nn.Module):
    """PatchGAN: gibt eine Karte aus, jeder Wert bewertet einen ~70x70-Ausschnitt als echt/fake."""

    def __init__(self, in_channels: int = 3, ndf: int = 64, n_layers: int = 3):
        super().__init__()
        # Erste Schicht ohne Normalisierung
        layers = [
            nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        # Weitere Schichten mit stride 2: Auflösung halbieren, Kanäle verdoppeln (max. 8 * ndf)
        channels = ndf
        for n in range(1, n_layers):
            next_channels = ndf * min(2 ** n, 8)
            layers += [
                nn.Conv2d(channels, next_channels, kernel_size=4, stride=2, padding=1),
                nn.InstanceNorm2d(next_channels),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            channels = next_channels

        # Eine Schicht mit stride 1, dann auf einen Kanal: die Echt/Fake-Karte
        next_channels = ndf * min(2 ** n_layers, 8)
        layers += [
            nn.Conv2d(channels, next_channels, kernel_size=4, stride=1, padding=1),
            nn.InstanceNorm2d(next_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(next_channels, 1, kernel_size=4, stride=1, padding=1),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def init_weights(net: nn.Module, std: float = 0.02) -> None:
    """Gewichte aller Faltungen ~ N(0, std), Bias = 0 (wie im CycleGAN-Paper)."""
    for m in net.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight, mean=0.0, std=std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def count_parameters(net: nn.Module) -> int:
    return sum(p.numel() for p in net.parameters())


if __name__ == "__main__":
    G = Generator(n_res_blocks=6)
    D = Discriminator()
    init_weights(G)
    init_weights(D)

    print(f"Generator:     {count_parameters(G):>10,} Parameter (Plan: 7.837.699)")
    print(f"Diskriminator: {count_parameters(D):>10,} Parameter (Plan: 2.764.737)")
    print(f"Gesamt 2G+2D:  {2 * count_parameters(G) + 2 * count_parameters(D):>10,}")

    with torch.no_grad():
        for size in (64, 128, 256):
            x = torch.randn(2, 3, size, size)
            g, d = G(x), D(x)
            assert g.shape == x.shape, f"Generator: {g.shape} != {x.shape}"
            assert -1 <= g.min() and g.max() <= 1, "Generator-Ausgabe nicht in [-1, 1]"
            print(f"{size:>3} px: G {tuple(x.shape)} -> {tuple(g.shape)}, D -> {tuple(d.shape)}")
    print("Alle Shape-Tests bestanden.")

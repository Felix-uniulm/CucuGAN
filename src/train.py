"""CycleGAN-Training: Gesichter (A) <-> Gurken (B).

Beispiele:
    # Overfit-Test auf 20 Bildern (CPU)
    python src/train.py --name overfit --max-images 20 --n-epochs 100 --n-epochs-decay 0 --pool-size 0

    # Baseline mit Paper-Werten
    python src/train.py --name baseline --data data/processed/collage128

    # Abgebrochenes Training fortsetzen
    python src/train.py --name baseline --resume

Ablage:
    checkpoints/<name>/latest.pt        letzter Stand (zum Fortsetzen)
    checkpoints/<name>/epoch_XXX.pt     Zwischenstände
    outputs/<name>/config.json          verwendete Einstellungen
    outputs/<name>/samples/             Kontrollbilder
    outputs/<name>/tb/                  TensorBoard-Logs (tensorboard --logdir outputs)
"""

import argparse
import itertools
import json
import random
import time
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from tqdm import tqdm

from dataset import UnpairedDataset
from models import Discriminator, Generator, init_weights
from utils import ImagePool, StructureLoss, denormalize, linear_decay, sample_grid, save_grid, set_requires_grad


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True, help="Name des Laufs (Ordner für Checkpoints und Ausgaben)")
    p.add_argument("--data", default="data/processed/collage128", help="Ordner mit train/ und test/")
    p.add_argument("--max-images", type=int, default=None, help="nur N Bilder pro Domäne (Overfit-Test)")
    # Netze
    p.add_argument("--ngf", type=int, default=64, help="Basiskanäle Generator")
    p.add_argument("--ndf", type=int, default=64, help="Basiskanäle Diskriminator")
    p.add_argument("--n-downsampling", type=int, default=2)
    p.add_argument("--n-res-blocks", type=int, default=6, help="6 für 128 px, 9 für 256 px")
    p.add_argument("--n-layers-d", type=int, default=3)
    p.add_argument("--attention", action="store_true",
                   help="Generator lernt eine Maske, wo er ändert; außerhalb wird die Eingabe durchgereicht")
    # Losses
    p.add_argument("--lambda-cycle", type=float, default=10.0)
    p.add_argument("--lambda-identity", type=float, default=0.5, help="relativ zu lambda-cycle")
    p.add_argument("--lambda-struct", type=float, default=0.0,
                   help="Gesichtsstruktur (grobe Kanten) muss im Gurkenbild sichtbar bleiben; 0 = aus")
    p.add_argument("--cycle-noise", type=float, default=0.0,
                   help="Rauschen (Standardabweichung) vor dem Rückweg gegen verstecktes Kodieren; 0 = aus")
    # Optimierung
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--beta1", type=float, default=0.5)
    p.add_argument("--n-epochs", type=int, default=100, help="Epochen mit konstanter Lernrate")
    p.add_argument("--n-epochs-decay", type=int, default=100, help="Epochen mit linear fallender Lernrate")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--pool-size", type=int, default=50, help="0 = kein Image-Pool")
    # Technik
    p.add_argument("--device", default="auto", help="auto, cpu, cuda, mps")
    p.add_argument("--amp", action="store_true", help="Mixed Precision (nur CUDA): Faltungen in 16 Bit, deutlich schneller")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=50, help="Losses alle N Schritte loggen")
    p.add_argument("--sample-every", type=int, default=0, help="zusätzlich alle N Schritte Bilder (0 = nur pro Epoche)")
    p.add_argument("--save-every", type=int, default=10, help="alle N Epochen einen Zwischenstand behalten")
    p.add_argument("--resume", action="store_true", help="von checkpoints/<name>/latest.pt fortsetzen")
    return p.parse_args()


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_models(cfg: dict) -> dict[str, nn.Module]:
    """Die vier Netze. G_AB: Gesicht -> Gurke, G_BA: Gurke -> Gesicht,
    D_A: echtes Gesicht?, D_B: echte Gurke?"""
    gen = dict(ngf=cfg["ngf"], n_downsampling=cfg["n_downsampling"], n_res_blocks=cfg["n_res_blocks"],
               attention=cfg.get("attention", False))   # .get: ältere Checkpoints kennen die Option nicht
    disc = dict(ndf=cfg["ndf"], n_layers=cfg["n_layers_d"])
    return {
        "G_AB": Generator(**gen),
        "G_BA": Generator(**gen),
        "D_A": Discriminator(**disc),
        "D_B": Discriminator(**disc),
    }


def main() -> None:
    args = parse_args()
    cfg = vars(args)
    ckpt_dir = Path("checkpoints") / args.name
    out_dir = Path("outputs") / args.name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    use_amp = args.amp and device.type == "cuda"
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True    # schnellste Faltungs-Algorithmen für feste Bildgröße suchen

    # --- Daten -------------------------------------------------------------------------
    train_ds = UnpairedDataset(args.data, "train", args.max_images)
    test_ds = UnpairedDataset(args.data, "test")
    with Image.open(train_ds.paths_a[0]) as img:
        cfg["image_size"] = img.width                       # für infer.py im Checkpoint
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, drop_last=True,
                        pin_memory=device.type == "cuda", persistent_workers=args.num_workers > 0)
    # Feste Testbilder: in jeder Epoche dieselben 8 Gesichter und 8 Gurken
    n_fixed = min(8, len(test_ds))
    fixed = [test_ds[i] for i in range(n_fixed)]
    fixed_A = torch.stack([f["A"] for f in fixed]).to(device)
    fixed_B = torch.stack([f["B"] for f in fixed]).to(device)

    # --- Netze, Optimizer, Lernraten-Plan -----------------------------------------------
    nets = build_models(cfg)
    for net in nets.values():
        init_weights(net)
        net.to(device)
    G_AB, G_BA, D_A, D_B = nets["G_AB"], nets["G_BA"], nets["D_A"], nets["D_B"]

    # Ein Optimizer für beide Generatoren, einer für beide Diskriminatoren
    opt_G = torch.optim.Adam(itertools.chain(G_AB.parameters(), G_BA.parameters()),
                             lr=args.lr, betas=(args.beta1, 0.999))
    opt_D = torch.optim.Adam(itertools.chain(D_A.parameters(), D_B.parameters()),
                             lr=args.lr, betas=(args.beta1, 0.999))
    decay = linear_decay(args.n_epochs, args.n_epochs_decay)
    sched_G = torch.optim.lr_scheduler.LambdaLR(opt_G, decay)
    sched_D = torch.optim.lr_scheduler.LambdaLR(opt_D, decay)

    gan_loss = nn.MSELoss()   # LSGAN: Abstand zum Sollwert 1 (echt) bzw. 0 (fake)
    l1 = nn.L1Loss()          # Cycle und Identity: mittlere absolute Pixelabweichung
    struct_loss = StructureLoss().to(device)

    def noisy(x: torch.Tensor) -> torch.Tensor:
        """Leichtes Rauschen: zerstört unsichtbar versteckte Feinmuster (Steganografie)."""
        return x + args.cycle_noise * torch.randn_like(x) if args.cycle_noise > 0 else x
    pool_A, pool_B = ImagePool(args.pool_size), ImagePool(args.pool_size)
    # Mixed Precision: autocast wählt pro Operation 16 oder 32 Bit; der GradScaler vergrößert den Loss
    # vor backward(), damit kleine Gradienten in 16 Bit nicht zu 0 werden. Ohne --amp tun beide nichts.
    autocast = lambda: torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp)  # noqa: E731
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    start_epoch, step = 0, 0
    if args.resume:
        ckpt = torch.load(ckpt_dir / "latest.pt", map_location=device)
        for k, net in nets.items():
            net.load_state_dict(ckpt["nets"][k])
        opt_G.load_state_dict(ckpt["opt_G"])
        opt_D.load_state_dict(ckpt["opt_D"])
        sched_G.load_state_dict(ckpt["sched_G"])
        sched_D.load_state_dict(ckpt["sched_D"])
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch, step = ckpt["epoch"] + 1, ckpt["step"]
        print(f"Fortgesetzt ab Epoche {start_epoch + 1}")

    with open(out_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    writer = SummaryWriter(out_dir / "tb")
    total_epochs = args.n_epochs + args.n_epochs_decay
    print(f"Gerät: {device}, {len(train_ds)} Gesichter / {len(train_ds.paths_b)} Gurken, "
          f"{len(loader)} Schritte pro Epoche, {total_epochs} Epochen")

    # --- Training ----------------------------------------------------------------------
    for epoch in range(start_epoch, total_epochs):
        running: dict[str, float] = {}
        t_epoch = time.time()
        for batch in tqdm(loader, desc=f"Epoche {epoch + 1}/{total_epochs}", leave=False):
            real_A = batch["A"].to(device, non_blocking=True)
            real_B = batch["B"].to(device, non_blocking=True)

            # 1) Generatoren: Diskriminatoren nur bewerten lassen, nicht mitlernen
            set_requires_grad([D_A, D_B], False)
            opt_G.zero_grad(set_to_none=True)

            with autocast():
                fake_B = G_AB(real_A)                 # Gesicht -> Gurke
                rec_A = G_BA(noisy(fake_B))           # ... -> wieder Gesicht
                fake_A = G_BA(real_B)                 # Gurke -> Gesicht
                rec_B = G_AB(noisy(fake_A))           # ... -> wieder Gurke

                pred_fake_B = D_B(fake_B)             # Generator will, dass D "echt" (1) sagt
                pred_fake_A = D_A(fake_A)
                loss_gan = gan_loss(pred_fake_B, torch.ones_like(pred_fake_B)) \
                    + gan_loss(pred_fake_A, torch.ones_like(pred_fake_A))
                loss_cycle = args.lambda_cycle * (l1(rec_A, real_A) + l1(rec_B, real_B))
                if args.lambda_identity > 0:          # echte Gurke in G_AB soll Gurke bleiben
                    loss_idt = args.lambda_cycle * args.lambda_identity * (
                        l1(G_AB(real_B), real_B) + l1(G_BA(real_A), real_A))
                else:
                    loss_idt = torch.zeros((), device=device)
                # Nur Richtung Gesicht -> Gurke: Augen, Nase, Mund sollen sichtbar bleiben
                loss_struct = args.lambda_struct * struct_loss(fake_B.float(), real_A.float()) \
                    if args.lambda_struct > 0 else torch.zeros((), device=device)
                loss_G = loss_gan + loss_cycle + loss_idt + loss_struct
            scaler.scale(loss_G).backward()           # ohne --amp: einfach loss_G.backward()
            scaler.step(opt_G)                        # ohne --amp: einfach opt_G.step()

            # 2) Diskriminatoren: echte Bilder -> 1, Fakes (aus dem Pool, ohne Gradient zum G) -> 0
            set_requires_grad([D_A, D_B], True)
            opt_D.zero_grad(set_to_none=True)
            with autocast():
                loss_D = 0.0
                for D, real, fake in ((D_A, real_A, pool_A.query(fake_A)), (D_B, real_B, pool_B.query(fake_B))):
                    pred_real = D(real)
                    pred_fake = D(fake.detach())
                    loss_D = loss_D + 0.5 * (gan_loss(pred_real, torch.ones_like(pred_real))
                                             + gan_loss(pred_fake, torch.zeros_like(pred_fake)))
            scaler.scale(loss_D).backward()
            scaler.step(opt_D)
            scaler.update()                           # Skalierungsfaktor für den nächsten Schritt anpassen

            # 3) Logging
            step += 1
            for k, v in (("G_gan", loss_gan), ("G_cycle", loss_cycle), ("G_identity", loss_idt),
                         ("G_struct", loss_struct), ("D", loss_D)):
                running[k] = running.get(k, 0.0) + v.item()
            if step % args.log_every == 0:
                for k, v in running.items():
                    writer.add_scalar(f"loss/{k}", v / args.log_every, step)
                running = {}
            if args.sample_every and step % args.sample_every == 0:
                save_grid(sample_grid(G_AB, G_BA, fixed_A, fixed_B),
                          out_dir / "samples" / f"step_{step:07d}.png", nrow=n_fixed)

        epoch_time = time.time() - t_epoch
        sched_G.step()
        sched_D.step()

        # Ende der Epoche: Kontrollbild, Lernrate, Checkpoint
        grid = sample_grid(G_AB, G_BA, fixed_A, fixed_B)
        save_grid(grid, out_dir / "samples" / f"epoch_{epoch + 1:03d}.png", nrow=n_fixed)
        writer.add_image("samples", make_grid(grid.cpu(), nrow=n_fixed), epoch + 1)
        writer.add_scalar("lr", opt_G.param_groups[0]["lr"], epoch + 1)
        if args.attention:   # gelernte Maske von G_AB: hell = wird zu Gurke, dunkel = bleibt
            with torch.no_grad():
                G_AB.eval()
                mask = G_AB.forward_with_mask(fixed_A)[1].repeat(1, 3, 1, 1)
                G_AB.train()
            save_grid(torch.cat([denormalize(fixed_A), mask]), out_dir / "samples" / f"mask_{epoch + 1:03d}.png",
                      nrow=n_fixed)

        ckpt = {
            "config": cfg, "epoch": epoch, "step": step,
            "nets": {k: net.state_dict() for k, net in nets.items()},
            "opt_G": opt_G.state_dict(), "opt_D": opt_D.state_dict(),
            "sched_G": sched_G.state_dict(), "sched_D": sched_D.state_dict(),
            "scaler": scaler.state_dict(),
        }
        torch.save(ckpt, ckpt_dir / "latest.pt")
        if (epoch + 1) % args.save_every == 0 or epoch + 1 == total_epochs:
            # Zwischenstände nur mit den Netzen (ohne Optimizer-Zustand: ~1/3 der Größe)
            light = {k: ckpt[k] for k in ("config", "epoch", "step", "nets")}
            torch.save(light, ckpt_dir / f"epoch_{epoch + 1:03d}.pt")
        print(f"Epoche {epoch + 1}/{total_epochs} fertig, Schritt {step}, "
              f"{epoch_time:.0f} s ({epoch_time / len(loader):.3f} s/Schritt), "
              f"lr {opt_G.param_groups[0]['lr']:.2e}", flush=True)

    writer.close()


if __name__ == "__main__":
    main()

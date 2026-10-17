"""CUT-Training (Park et al. 2020, Contrastive Unpaired Translation): Gesichter (A) -> Gurken (B).

Unterschied zu CycleGAN (train.py): nur EIN Generator und EIN Diskriminator, kein Rückweg.
Der Inhalt bleibt über PatchNCE erhalten: Jede Stelle im Gurkengesicht soll der Stelle an derselben
Position im Original entsprechen (Auge -> Gurken-Auge), nicht irgendeiner anderen. Ohne Rückweg gibt
es nichts unsichtbar zu verstecken.

Beispiele:
    python src/train_cut.py --name cut_test --n-epochs 20 --n-epochs-decay 0 --amp
    python src/train_cut.py --name cut --amp                    # voller Lauf
    python src/train_cut.py --name cut --amp --resume

Ablage wie bei train.py: checkpoints/<name>/, outputs/<name>/{config.json,samples,tb}
Kontrollbild-Zeilen: echtes Gesicht | Gurkengesicht | echte Collage | G(Collage) (soll gleich bleiben)
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
from models import Discriminator, Generator, PatchSampleF, init_weights
from train import pick_device
from utils import (ImagePool, StructureLoss, denormalize, linear_decay, patch_nce_loss, sample_grid_cut,
                   save_grid, set_requires_grad)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True, help="Name des Laufs")
    p.add_argument("--data", default="data/processed/collage128", help="Ordner mit train/ und test/")
    p.add_argument("--max-images", type=int, default=None, help="nur N Bilder pro Domäne (Test)")
    # Netze
    p.add_argument("--ngf", type=int, default=64)
    p.add_argument("--ndf", type=int, default=64)
    p.add_argument("--n-downsampling", type=int, default=2)
    p.add_argument("--n-res-blocks", type=int, default=6, help="6 für 128 px, 9 für 256 px")
    p.add_argument("--n-layers-d", type=int, default=3)
    p.add_argument("--attention", action="store_true", help="Generator lernt eine Maske (wie in train.py)")
    # CUT
    p.add_argument("--nce-layers", default="0,4,8,12,15",
                   help="Schichten des Generators für PatchNCE (Index in G.model): Eingabe, 2x Verkleinern, "
                        "Res-Block 3 und 6. Bei 9 Res-Blöcken wie im Paper: 0,4,8,12,16")
    p.add_argument("--num-patches", type=int, default=256, help="zufällige Stellen pro Schicht")
    p.add_argument("--nce-tau", type=float, default=0.07, help="Temperatur der Kreuzentropie")
    p.add_argument("--lambda-gan", type=float, default=1.0)
    p.add_argument("--lambda-nce", type=float, default=1.0)
    p.add_argument("--no-nce-idt", action="store_true",
                   help="Identitäts-NCE aus (Paper-Standard ist an: G(Collage) soll Collage bleiben)")
    p.add_argument("--lambda-struct", type=float, default=0.0, help="zusätzlich unser Struktur-Loss; 0 = aus")
    # Optimierung (Paper: lr 2e-4, betas (0.5, 0.999), kein Image-Pool)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--beta1", type=float, default=0.5)
    p.add_argument("--n-epochs", type=int, default=100)
    p.add_argument("--n-epochs-decay", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--pool-size", type=int, default=0)
    # Technik
    p.add_argument("--device", default="auto")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = vars(args)
    cfg["model"] = "cut"
    ckpt_dir = Path("checkpoints") / args.name
    out_dir = Path("outputs") / args.name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    use_amp = args.amp and device.type == "cuda"
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # --- Daten -------------------------------------------------------------------------
    train_ds = UnpairedDataset(args.data, "train", args.max_images)
    test_ds = UnpairedDataset(args.data, "test")
    with Image.open(train_ds.paths_a[0]) as img:
        cfg["image_size"] = img.width
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, drop_last=True,
                        pin_memory=device.type == "cuda", persistent_workers=args.num_workers > 0)
    n_fixed = min(8, len(test_ds))
    fixed = [test_ds[i] for i in range(n_fixed)]
    fixed_A = torch.stack([f["A"] for f in fixed]).to(device)
    fixed_B = torch.stack([f["B"] for f in fixed]).to(device)

    # --- Netze ---------------------------------------------------------------------------
    G = Generator(ngf=args.ngf, n_downsampling=args.n_downsampling, n_res_blocks=args.n_res_blocks,
                  attention=args.attention)
    D = Discriminator(ndf=args.ndf, n_layers=args.n_layers_d)
    nce_layers = [int(i) for i in args.nce_layers.split(",")]
    assert max(nce_layers) < len(G.model), f"--nce-layers: G hat nur {len(G.model)} Schichten"
    with torch.no_grad():   # Kanalzahlen der gewählten Schichten bestimmen
        channels = [f.shape[1] for f in G.encode(torch.zeros(1, 3, 64, 64), nce_layers)]
    F = PatchSampleF(channels)
    for net in (G, D, F):
        init_weights(net)
        net.to(device)

    opt_G = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    opt_F = torch.optim.Adam(F.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    decay = linear_decay(args.n_epochs, args.n_epochs_decay)
    scheds = [torch.optim.lr_scheduler.LambdaLR(o, decay) for o in (opt_G, opt_D, opt_F)]

    gan_loss = nn.MSELoss()
    struct_loss = StructureLoss().to(device)
    pool = ImagePool(args.pool_size)
    autocast = lambda: torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp)  # noqa: E731
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    def nce(src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """PatchNCE zwischen Eingabe src und Ergebnis tgt, gemittelt über die Schichten."""
        feat_q = G.encode(tgt, nce_layers)                 # Ergebnis: daran lernt G
        feat_k = G.encode(src, nce_layers)                 # Original: nur Zielwerte
        q, ids = F(feat_q, args.num_patches)
        k, _ = F(feat_k, args.num_patches, ids)            # dieselben Stellen
        return sum(patch_nce_loss(a, b, src.shape[0], args.nce_tau) for a, b in zip(q, k)) / len(q)

    start_epoch, step = 0, 0
    if args.resume:
        ckpt = torch.load(ckpt_dir / "latest.pt", map_location=device)
        for net, k in ((G, "G"), (D, "D"), (F, "F")):
            net.load_state_dict(ckpt["nets"][k])
        for o, k in ((opt_G, "opt_G"), (opt_D, "opt_D"), (opt_F, "opt_F")):
            o.load_state_dict(ckpt[k])
        for s, sd in zip(scheds, ckpt["scheds"]):
            s.load_state_dict(sd)
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch, step = ckpt["epoch"] + 1, ckpt["step"]
        print(f"Fortgesetzt ab Epoche {start_epoch + 1}")

    with open(out_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    writer = SummaryWriter(out_dir / "tb")
    total_epochs = args.n_epochs + args.n_epochs_decay
    print(f"CUT | Gerät: {device}, {len(train_ds)} Gesichter / {len(train_ds.paths_b)} Gurken, "
          f"{len(loader)} Schritte pro Epoche, {total_epochs} Epochen, NCE-Schichten {nce_layers} {channels}")

    # --- Training ----------------------------------------------------------------------
    for epoch in range(start_epoch, total_epochs):
        running: dict[str, float] = {}
        t_epoch = time.time()
        for batch in tqdm(loader, desc=f"Epoche {epoch + 1}/{total_epochs}", leave=False):
            real_A = batch["A"].to(device, non_blocking=True)
            real_B = batch["B"].to(device, non_blocking=True)

            # Gesicht und Collage zusammen durch G: fake_B = Gurkengesicht, idt_B = G(Collage)
            with autocast():
                both = G(torch.cat([real_A, real_B]) if not args.no_nce_idt else real_A)
            fake_B = both[:real_A.shape[0]]
            idt_B = both[real_A.shape[0]:] if not args.no_nce_idt else None

            # 1) Diskriminator: echte Collage -> 1, Gurkengesicht -> 0
            set_requires_grad([D], True)
            opt_D.zero_grad(set_to_none=True)
            with autocast():
                pred_real = D(real_B)
                pred_fake = D(pool.query(fake_B).detach())
                loss_D = 0.5 * (gan_loss(pred_real, torch.ones_like(pred_real))
                                + gan_loss(pred_fake, torch.zeros_like(pred_fake)))
            scaler.scale(loss_D).backward()
            scaler.step(opt_D)

            # 2) Generator und F: Diskriminator täuschen, Stellen sollen sich entsprechen
            set_requires_grad([D], False)
            opt_G.zero_grad(set_to_none=True)
            opt_F.zero_grad(set_to_none=True)
            with autocast():
                pred = D(fake_B)
                loss_gan = args.lambda_gan * gan_loss(pred, torch.ones_like(pred))
                loss_nce = args.lambda_nce * nce(real_A, fake_B)
                if idt_B is not None:   # Collage soll Collage bleiben (ersetzt den Identity-Loss)
                    loss_nce = 0.5 * (loss_nce + args.lambda_nce * nce(real_B, idt_B))
                loss_struct = args.lambda_struct * struct_loss(fake_B.float(), real_A.float()) \
                    if args.lambda_struct > 0 else torch.zeros((), device=device)
                loss_G = loss_gan + loss_nce + loss_struct
            scaler.scale(loss_G).backward()
            scaler.step(opt_G)
            scaler.step(opt_F)
            scaler.update()

            step += 1
            for k, v in (("G_gan", loss_gan), ("NCE", loss_nce), ("G_struct", loss_struct), ("D", loss_D)):
                running[k] = running.get(k, 0.0) + v.item()
            if step % args.log_every == 0:
                for k, v in running.items():
                    writer.add_scalar(f"loss/{k}", v / args.log_every, step)
                running = {}

        epoch_time = time.time() - t_epoch
        for s in scheds:
            s.step()

        grid = sample_grid_cut(G, fixed_A, fixed_B)
        save_grid(grid, out_dir / "samples" / f"epoch_{epoch + 1:03d}.png", nrow=n_fixed)
        writer.add_image("samples", make_grid(grid.cpu(), nrow=n_fixed), epoch + 1)
        writer.add_scalar("lr", opt_G.param_groups[0]["lr"], epoch + 1)
        if args.attention:
            with torch.no_grad():
                G.eval()
                mask = G.forward_with_mask(fixed_A)[1].repeat(1, 3, 1, 1)
                G.train()
            save_grid(torch.cat([denormalize(fixed_A), mask]), out_dir / "samples" / f"mask_{epoch + 1:03d}.png",
                      nrow=n_fixed)

        ckpt = {
            "config": cfg, "epoch": epoch, "step": step,
            "nets": {"G": G.state_dict(), "D": D.state_dict(), "F": F.state_dict()},
            "opt_G": opt_G.state_dict(), "opt_D": opt_D.state_dict(), "opt_F": opt_F.state_dict(),
            "scheds": [s.state_dict() for s in scheds], "scaler": scaler.state_dict(),
        }
        torch.save(ckpt, ckpt_dir / "latest.pt")
        if (epoch + 1) % args.save_every == 0 or epoch + 1 == total_epochs:
            light = {k: ckpt[k] for k in ("config", "epoch", "step")} | {"nets": {"G": ckpt["nets"]["G"]}}
            torch.save(light, ckpt_dir / f"epoch_{epoch + 1:03d}.pt")
        print(f"Epoche {epoch + 1}/{total_epochs} fertig, Schritt {step}, "
              f"{epoch_time:.0f} s ({epoch_time / len(loader):.3f} s/Schritt), "
              f"lr {opt_G.param_groups[0]['lr']:.2e}", flush=True)

    writer.close()


if __name__ == "__main__":
    main()

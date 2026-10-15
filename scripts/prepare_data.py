"""Datenaufbereitung für CucuGAN.

Reihenfolge: exakte Duplikate entfernen -> Split (train/test) -> Zuschneiden -> Skalieren.

Befehle: dedupe (exakte Duplikate), split (Aufteilung anzeigen), preview (Gurkenerkennung
prüfen), build (fertige Trainingsbilder erzeugen).

Beispiel:
    python scripts/prepare_data.py dedupe --src data/raw/cucumber/jin2023 --report outputs/dedupe/jin2023
    python scripts/prepare_data.py split --src data/raw/cucumber/jin2023
    python scripts/prepare_data.py preview --src data/raw/cucumber/jin2023
    python scripts/prepare_data.py build --size 128
"""

import argparse
import collections
import csv
import hashlib
import math
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Gurkenerkennung (Jin-Fotos: Gurke auf schwarzem Stoff, darüber Lineal und Farbkarten)
DETECT_WIDTH = 800        # Erkennung auf verkleinertem Bild, ca. so breit
MIN_SATURATION = 60       # schwarzer Stoff, Staub und weiße Etiketten sind kaum gesättigt
MIN_VALUE = 35            # schwarzer Stoff ist dunkel
HUE_RANGE = (42, 90)      # Gurkengrün (PIL-HSV 0-255). Gemessen: Lineal 19-26, gelbe Gurken und
                          # Etiketten 32-38, grüne Gurken 43-55, grüne Farbkarte 96-103
MIN_AREA_FRACTION = 0.002 # kleinere Flecken ignorieren
MAX_RECT_FILL = 0.9       # füllt eine Fläche ihr gedrehtes Rechteck stärker, ist es eine Farbkarte
# Quadrate
SQUARE_FRACTION = 0.8     # Kantenlänge = 0.8 * typische Gurkendicke
MIN_SKIN_FRACTION = 0.97  # so viel des Quadrats muss Gurke sein


def list_images(folder: Path) -> list[Path]:
    """Alle Bilder im Ordner, sortiert (damit Ergebnisse reproduzierbar sind)."""
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


def file_md5(path: Path) -> str:
    """Prüfsumme der Dateibytes: gleiche Prüfsumme = identische Datei."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_duplicates(paths: list[Path], workers: int = 4):
    """Exakte Duplikate finden.

    Gibt (behalten, entfernt) zurück; entfernt ist eine Liste von
    (entferntes_bild, behaltenes_bild). Von zwei Duplikaten bleibt das erste
    in der sortierten Liste.
    """
    with ProcessPoolExecutor(workers) as pool:
        md5s = list(pool.map(file_md5, paths, chunksize=16))

    kept, removed = [], []
    first_with_md5 = {}
    for path, m in zip(paths, md5s):
        if m in first_with_md5:
            removed.append((path, first_with_md5[m]))
        else:
            first_with_md5[m] = path
            kept.append(path)
    return kept, removed


def split_for(name: str, test_fraction: float, seed: int = 0) -> str:
    """Split eines Fotos, nur abhängig von seinem Namen.

    md5 macht aus dem Namen eine gleichverteilte Zahl in [0, 1). Dieselbe Datei
    landet damit immer im selben Split, egal welche anderen Dateien es gibt.
    Mit einem anderen seed entsteht eine andere, ebenso stabile Aufteilung.
    """
    digest = hashlib.md5(f"{seed}:{name}".encode()).hexdigest()
    u = int(digest, 16) / 16 ** len(digest)
    return "test" if u < test_fraction else "train"


def load_image(path: Path) -> Image.Image:
    """Bild laden, EXIF-Drehung anwenden (Handyfotos speichern sie oft nur dort)."""
    with Image.open(path) as img:
        return ImageOps.exif_transpose(img).convert("RGB")


def connected_components(mask: np.ndarray) -> list[np.ndarray]:
    """Zusammenhängende Flächen einer Maske als Listen von (y, x)-Koordinaten."""
    seen = np.zeros_like(mask)
    h, w = mask.shape
    components = []
    for y0, x0 in zip(*np.nonzero(mask)):
        if seen[y0, x0]:
            continue
        seen[y0, x0] = True
        queue, pixels = collections.deque([(y0, x0)]), []
        while queue:
            y, x = queue.popleft()
            pixels.append((y, x))
            for ny, nx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    queue.append((ny, nx))
        components.append(np.array(pixels))
    return components


def principal_axes(points: np.ndarray) -> np.ndarray:
    """Hauptachsen (PCA) einer Punktwolke als Spalten, längste Achse zuerst."""
    evals, evecs = np.linalg.eigh(np.cov(points.T.astype(float)))
    return evecs[:, np.argsort(evals)[::-1]]


def rect_fill(pixels: np.ndarray) -> float:
    """Anteil, den eine Fläche von ihrem an der Hauptachse ausgerichteten Rechteck füllt.

    Rechteck (Karte) ~1.0, Ellipse ~0.79, gebogene Gurke deutlich weniger.
    """
    proj = (pixels - pixels.mean(axis=0)) @ principal_axes(pixels)
    return len(pixels) / ((np.ptp(proj[:, 0]) + 1) * (np.ptp(proj[:, 1]) + 1))


def find_cucumber(img: Image.Image):
    """Gurke im Foto finden.

    Gibt (maske, faktor) zurück: maske ist ein bool-Array auf dem verkleinerten
    Bild, faktor der Verkleinerungsfaktor. None, wenn nichts gefunden wurde.
    """
    factor = max(1, img.width // DETECT_WIDTH)
    small = img.reduce(factor)
    h, s, v = (np.asarray(c, dtype=np.int16) for c in small.convert("HSV").split())
    mask = (s > MIN_SATURATION) & (v > MIN_VALUE) & (h >= HUE_RANGE[0]) & (h <= HUE_RANGE[1])

    # Kleine Lücken (helle Noppen, Glanzpunkte) schließen, einzelne Pixel entfernen
    m = Image.fromarray(mask.astype(np.uint8) * 255)
    m = m.filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.MinFilter(5))
    m = m.filter(ImageFilter.MinFilter(3)).filter(ImageFilter.MaxFilter(3))
    mask = np.asarray(m) > 0

    H, W = mask.shape
    comps = [c for c in connected_components(mask) if len(c) > MIN_AREA_FRACTION * H * W]

    # Größte grüne Fläche, die kein Rechteck (grüne Farbkarte) ist
    best = None
    for c in comps:
        if rect_fill(c) > MAX_RECT_FILL:
            continue
        if best is None or len(c) > len(best):
            best = c
    if best is None:
        return None

    out = np.zeros_like(mask)
    out[best[:, 0], best[:, 1]] = True
    return out, factor


def cut_squares(img: Image.Image, mask: np.ndarray, factor: int):
    """Gurke waagerecht drehen und quadratische Ausschnitte entlang der Gurke schneiden.

    Gibt (quadrate, info) zurück: quadrate sind PIL-Bilder in Originalauflösung,
    info enthält Winkel, Kantenlänge und das gedrehte Bild mit Markierungen.
    """
    ys, xs = np.nonzero(mask)
    # Hauptachse (PCA): Richtung, in der die Gurke am längsten ist
    dx, dy = principal_axes(np.stack([xs, ys], axis=1))[:, 0]
    angle = math.degrees(math.atan2(dy, dx))

    # Ausschnitt um die Gurke in Originalauflösung (mit Rand, damit beim Drehen nichts fehlt)
    margin = 0.15 * max(np.ptp(xs), np.ptp(ys)) + 10
    box = (int(max(0, (xs.min() - margin) * factor)), int(max(0, (ys.min() - margin) * factor)),
           int(min(img.width, (xs.max() + margin) * factor)), int(min(img.height, (ys.max() + margin) * factor)))
    crop = img.crop(box)
    mask_img = Image.fromarray(mask.astype(np.uint8) * 255).resize(
        (mask.shape[1] * factor, mask.shape[0] * factor), Image.NEAREST).crop(box)

    # Drehen, sodass die Hauptachse waagerecht liegt
    crop = crop.rotate(angle, resample=Image.BICUBIC, expand=True)
    rmask = np.asarray(mask_img.rotate(angle, resample=Image.NEAREST, expand=True)) > 127

    # Dicke und Mitte der Gurke pro Spalte
    cols = rmask.any(axis=0)
    top = np.where(cols, rmask.argmax(axis=0), 0)
    bottom = np.where(cols, rmask.shape[0] - rmask[::-1].argmax(axis=0), 0)
    thickness = bottom - top
    side = int(SQUARE_FRACTION * np.median(thickness[cols]))
    # Löcher (helle Noppen, Glanzlichter) füllen: quer zur Gurke ist alles zwischen
    # Ober- und Unterkante Gurke
    rows = np.arange(rmask.shape[0])[:, None]
    rmask = (rows >= top[None, :]) & (rows < bottom[None, :])

    squares, boxes = [], []
    x = int(np.argmax(cols))
    x_end = rmask.shape[1] - int(np.argmax(cols[::-1]))
    while side > 0 and x + side <= x_end:
        cy = int(np.mean((top[x:x + side] + bottom[x:x + side]) / 2))
        y = cy - side // 2
        if y >= 0 and y + side <= rmask.shape[0] and rmask[y:y + side, x:x + side].mean() >= MIN_SKIN_FRACTION:
            squares.append(crop.crop((x, y, x + side, y + side)))
            boxes.append((x, y, side))
            x += side
        else:
            x += max(1, side // 8)  # an den Enden: in kleinen Schritten weitersuchen

    return squares, {"angle": angle, "side": side, "rotated": crop, "boxes": boxes}


def process_cucumber_photo(path: Path):
    """Für Vorschau und Build: Gurke finden und Quadrate schneiden."""
    img = load_image(path)
    found = find_cucumber(img)
    if found is None:
        return path, None, None, img
    squares, info = cut_squares(img, *found)
    return path, squares, info, img


def cmd_preview(args) -> None:
    """Zufällige Fotos verarbeiten und Kontrollbilder speichern (schreibt nichts nach data/)."""
    paths = list_images(args.src)
    random.Random(args.seed).shuffle(paths)
    paths = paths[:args.n]
    args.out.mkdir(parents=True, exist_ok=True)

    with ProcessPoolExecutor(args.workers) as pool:
        results = list(pool.map(process_cucumber_photo, paths))

    sides, n_squares, failed = [], [], []
    for k, (path, squares, info, img) in enumerate(results):
        if squares is None:
            failed.append(path.name)
            img.thumbnail((600, 600))
            img.save(args.out / f"{k:03d}_FEHLT_{path.stem}.jpg")
            continue
        sides.append(info["side"])
        n_squares.append(len(squares))
        rot = info["rotated"].copy()
        draw = ImageDraw.Draw(rot)
        for x, y, s in info["boxes"]:
            draw.rectangle((x, y, x + s, y + s), outline="red", width=max(4, s // 40))
        rot.thumbnail((900, 300))
        tiles = [sq.resize((140, 140)) for sq in squares[:6]]
        sheet = Image.new("RGB", (900, 300 + 150), "gray")
        sheet.paste(rot, (0, 0))
        for i, t in enumerate(tiles):
            sheet.paste(t, (i * 150, 305))
        ImageDraw.Draw(sheet).text((5, 5), f"{path.name}  {info['angle']:.0f} Grad  Kante {info['side']} px  "
                                           f"{len(squares)} Quadrate", fill="white")
        sheet.save(args.out / f"{k:03d}_{path.stem}.jpg")

    print(f"{len(paths)} Fotos, Gurke nicht gefunden: {len(failed)} {failed}")
    if sides:
        print(f"Kantenlänge px: min {min(sides)}, median {int(np.median(sides))}, max {max(sides)}")
        print(f"Quadrate pro Foto: min {min(n_squares)}, median {int(np.median(n_squares))}, max {max(n_squares)}")
    print(f"Kontrollbilder: {args.out}")


def safe_stem(path: Path) -> str:
    """Dateiname ohne Leer- und Sonderzeichen ('PE107- (3)' -> 'PE107-_3_')."""
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in path.stem)


def build_face(path: Path, out: Path, size: int, test_fraction: float, seed: int):
    split = split_for(path.name, test_fraction, seed)
    img = load_image(path).resize((size, size), Image.LANCZOS)
    name = f"{safe_stem(path)}.png"
    img.save(out / split / "A" / name)
    return [("A", split, name, path.name, "", "")]


def build_cucumber(path: Path, out: Path, size: int, test_fraction: float, seed: int):
    """Alle Quadrate eines Fotos landen im Split des Fotos (Gruppierung)."""
    split = split_for(path.name, test_fraction, seed)
    _, squares, info, _ = process_cucumber_photo(path)
    rows = []
    for i, sq in enumerate(squares or []):
        if sq.width < size:
            continue  # nie hochskalieren
        name = f"{safe_stem(path)}_c{i:02d}.png"
        sq.resize((size, size), Image.LANCZOS).save(out / split / "B" / name)
        rows.append(("B", split, name, path.name, f"{info['angle']:.1f}", info["side"]))
    return rows


def cmd_build(args) -> None:
    """Fertige Trainingsbilder erzeugen: <out>/<size>/{train,test}/{A,B}/ plus manifest.csv."""
    out = args.out / str(args.size)
    if out.exists():
        raise SystemExit(f"{out} existiert schon. Zum Neuerzeugen den Ordner vorher löschen.")
    for split in ("train", "test"):
        for domain in ("A", "B"):
            (out / split / domain).mkdir(parents=True)

    jobs = [(build_face, p) for p in list_images(args.faces)]
    jobs += [(build_cucumber, p) for p in list_images(args.cucumbers)]
    rows = []
    with ProcessPoolExecutor(args.workers) as pool:
        futures = [pool.submit(fn, p, out, args.size, args.test_fraction, args.seed) for fn, p in jobs]
        for k, f in enumerate(futures, 1):
            rows += f.result()
            if k % 200 == 0 or k == len(futures):
                print(f"  {k}/{len(futures)} Fotos verarbeitet", flush=True)

    with open(out / "manifest.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["domain", "split", "datei", "quelle", "winkel", "kante_px"])
        w.writerows(rows)

    for domain, label in (("A", "Gesichter"), ("B", "Gurken")):
        for split in ("train", "test"):
            sel = [r for r in rows if r[0] == domain and r[1] == split]
            print(f"{label:9s} {split:5s}: {len(sel):5d} Bilder aus {len({r[3] for r in sel})} Fotos")
    used = {r[3] for r in rows if r[0] == "B"}
    print(f"Gurkenfotos ohne verwendbares Quadrat: {len(list_images(args.cucumbers)) - len(used)}")
    print(f"Fertig: {out}")


def cmd_dedupe(args) -> None:
    paths = list_images(args.src)
    kept, removed = find_duplicates(paths, args.workers)
    print(f"{len(paths)} Bilder, {len(removed)} exakte Duplikate, {len(kept)} bleiben")

    args.report.mkdir(parents=True, exist_ok=True)
    with open(args.report / "duplicates.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["entfernt", "behalten"])
        w.writerows((d.name, o.name) for d, o in removed)
    print(f"Bericht: {args.report / 'duplicates.csv'}")


def cmd_split(args) -> None:
    paths = list_images(args.src)
    counts = {"train": 0, "test": 0}
    for p in paths:
        counts[split_for(p.name, args.test_fraction, args.seed)] += 1
    n = len(paths)
    print(f"{n} Bilder: train {counts['train']} ({counts['train'] / n:.1%}), "
          f"test {counts['test']} ({counts['test'] / n:.1%})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("dedupe", help="Exakte Duplikate finden und Bericht schreiben")
    p.add_argument("--src", type=Path, required=True, help="Ordner mit Originalbildern")
    p.add_argument("--report", type=Path, required=True, help="Ausgabeordner für den CSV-Bericht")
    p.add_argument("--workers", type=int, default=4, help="Anzahl paralleler Prozesse")
    p.set_defaults(func=cmd_dedupe)

    p = sub.add_parser("split", help="Aufteilung train/test anzeigen")
    p.add_argument("--src", type=Path, required=True, help="Ordner mit Originalbildern")
    p.add_argument("--test-fraction", type=float, default=0.1, help="Anteil der Testbilder")
    p.add_argument("--seed", type=int, default=0, help="anderer Wert = andere Aufteilung")
    p.set_defaults(func=cmd_split)

    p = sub.add_parser("preview", help="Gurkenerkennung und Quadrate an zufälligen Fotos prüfen")
    p.add_argument("--src", type=Path, required=True, help="Ordner mit Gurkenfotos")
    p.add_argument("--out", type=Path, default=Path("outputs/preview"), help="Ordner für Kontrollbilder")
    p.add_argument("--n", type=int, default=24, help="Anzahl Fotos")
    p.add_argument("--seed", type=int, default=0, help="Auswahl der Fotos")
    p.add_argument("--workers", type=int, default=4, help="Anzahl paralleler Prozesse")
    p.set_defaults(func=cmd_preview)

    p = sub.add_parser("build", help="Trainingsbilder erzeugen (Gesichter skalieren, Gurkenquadrate schneiden)")
    p.add_argument("--size", type=int, default=128, help="Kantenlänge der fertigen Bilder in px")
    p.add_argument("--faces", type=Path, default=Path("data/raw/ffhq"), help="Ordner mit Gesichtern")
    p.add_argument("--cucumbers", type=Path, default=Path("data/raw/cucumber/jin2023"), help="Ordner mit Gurkenfotos")
    p.add_argument("--out", type=Path, default=Path("data/processed"), help="Zielordner")
    p.add_argument("--test-fraction", type=float, default=0.1, help="Anteil der Testbilder")
    p.add_argument("--seed", type=int, default=0, help="anderer Wert = andere Aufteilung")
    p.add_argument("--workers", type=int, default=4, help="Anzahl paralleler Prozesse")
    p.set_defaults(func=cmd_build)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

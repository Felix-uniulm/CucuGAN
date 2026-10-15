"""Datenaufbereitung für CucuGAN.

Reihenfolge: exakte Duplikate entfernen -> Split (train/test) -> Zuschneiden -> Skalieren.

Aktuell enthalten: Schritt 1 (exakte Duplikate finden).

Beispiel:
    python scripts/prepare_data.py dedupe --src data/raw/cucumber/jin2023 --report outputs/dedupe/jin2023
"""

import argparse
import csv
import hashlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("dedupe", help="Exakte Duplikate finden und Bericht schreiben")
    p.add_argument("--src", type=Path, required=True, help="Ordner mit Originalbildern")
    p.add_argument("--report", type=Path, required=True, help="Ausgabeordner für den CSV-Bericht")
    p.add_argument("--workers", type=int, default=4, help="Anzahl paralleler Prozesse")
    p.set_defaults(func=cmd_dedupe)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

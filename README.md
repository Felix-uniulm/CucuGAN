# CucuGAN

Verwandelt Gesichter mit einem selbst implementierten CycleGAN (PyTorch) in Gurkengesichter.
Status: in Arbeit, siehe [PLAN.md](PLAN.md).

## Daten

Die Bilder sind nicht im Repository enthalten. Sie liegen lokal unter `data/raw/`.

### Domäne A: Gesichter
| Datensatz | Quelle | Lizenz | Ordner |
|---|---|---|---|
| FFHQ small (Teilmenge von FFHQ, Karras et al. 2019) | [Kaggle](https://www.kaggle.com/datasets/tommykamaz/faces-dataset-small), Original: [NVlabs/ffhq-dataset](https://github.com/NVlabs/ffhq-dataset) | CC BY-NC-SA 4.0 | `data/raw/ffhq/` |

### Domäne B: Gurken
| Datensatz | Autor:innen | DOI | Lizenz | Ordner |
|---|---|---|---|---|
| Cucumber Dataset (2023) | T. Jin | [10.5281/zenodo.10081197](https://doi.org/10.5281/zenodo.10081197) | CC BY 4.0 | `data/raw/cucumber/jin2023/` |

Verwendet wird nur der Ordner „Fruit Dataset“ (ganze Gurken mit Schale), nicht „Flesh DataSet“ (Schnittflächen).

## Automatische Hautmaske bei der Inferenz

Die optionale Face-Parsing-Maske übernimmt Gesichtshaut, Nase, beide Ohren und Hals aus dem
CycleGAN-Ergebnis. Augen, Augenbrauen, Lippen, Mund, Haare, Schmuck, Kleidung und
Hintergrund bleiben aus dem Original erhalten. Die Maske wird aus dem Original
berechnet, ihre Ränder werden ausschließlich innerhalb der Hautfläche geglättet.
Das Training und bestehende Checkpoints bleiben unverändert.

```bash
pip install -r requirements-face-parsing.txt
python src/infer.py \
  --checkpoint checkpoints/baseline_squares_2k/latest.pt \
  --input input/face.png \
  --output outputs/inference/face_cucumber.png \
  --skin-mask --save-mask
```

Die bestehende Inferenz erwartet weiterhin 128×128-Bilder mit einem gut sichtbaren
Gesicht. `--save-mask` schreibt zusätzlich `face_cucumber_mask.png` (weiß = übernehmen).
`--mask-feather 1` steuert den weichen Rand in Pixeln; `0` deaktiviert ihn.
Ohne `--skin-mask` wird wie bisher das ganze Generatorbild gespeichert.
Wird keine Haut erkannt, bricht die Verarbeitung mit einer Fehlermeldung ab.
Falsch klassifizierte Bereiche können trotzdem verändert werden; Kontrollmaske prüfen.

Verwendet wird [jonathandinu/face-parsing](https://huggingface.co/jonathandinu/face-parsing),
ein auf CelebAMask-HQ trainiertes SegFormer-Modell. Beim ersten Aufruf werden
die Gewichte heruntergeladen, danach wird der lokale Hugging-Face-Cache genutzt.
Ein lokal gespeichertes Modell kann über `--face-model /pfad/zum/modell` geladen
werden. Laut Modellkarte ist das Modell für nichtkommerzielle Forschung und Lehre
vorgesehen. Das Face-Parsing-Modell erzeugt nur die Maske, die Gurkentextur erzeugt
weiterhin das eigene CycleGAN.

## Dichtere Schalentextur als Experiment

Eine 2×2-Kachelung vorhandener Schalenquadrate, wieder auf 128×128 verkleinert,
halbiert ungefähr die Größe der Texturmerkmale. Einfaches Wiederholen erzeugt
jedoch oft sichtbare Nähte. Abwechselnd gespiegelte Kacheln reduzieren die
Farbübergänge, können aber symmetrische Wiederholungen erzeugen. Für einen ersten
Vergleich 2×2 statt gleich 4×4 ausprobieren; feine Details gehen bei 128×128 schnell
verloren. Das garantiert noch keine entsprechend dichtere Textur im Generatorbild.

Solche Varianten in einem separaten Datensatz erzeugen, ursprüngliche train/test-
Zuordnung beibehalten und keine Ausschnitte verschiedener Splits mischen.
Die veränderten B-Bilder erfordern einen neuen Trainingsversuch; die Hautmaske
allein verändert die Texturdichte eines vorhandenen Checkpoints nicht.

## Training mit Haut-Loss und gespiegelter 2×2-Textur

Diese Variante begrenzt den Textur-Loss auf Gesichtshaut, Nase, Ohren und Hals. Zur Auswertung: `infer.py` ohne `--skin-mask` zeigt das rohe Ergebnis.
Der Generator bleibt ein RGB-Netz; die Maske ist hier ausschließlich Zusatzinformation
für Textur- und Erhaltungs-Loss, kein vierter Eingangskanal.

1. Masken einmalig für train und test vorbereiten (auf GPU deutlich schneller):

   ```bash
   pip install -r requirements-face-parsing.txt
   python scripts/prepare_masks.py --data data/processed/squares128 --device cuda
   ```

   Ohne GPU `--device cpu` verwenden. Die Dateien landen in
   `<split>/masks_A_skin_ears_neck/<ursprünglicher Dateiname>.png`, beispielsweise `00418.png.png`.
   Sie enthalten binär `skin`, `nose`, `l_ear`, `r_ear` und `neck`. Existierende Masken werden übersprungen;
   nach Änderungen an Bildern müssen die betroffenen Masken neu erzeugt werden.
   Fehler bei der Segmentierung werden gemeldet und dürfen nicht durch Vollmasken
   ersetzt werden. Vor dem langen Lauf Masken visuell prüfen.

2. Einen neuen Trainingslauf starten (Beispielwerte, noch nicht optimiert):

   ```bash
   python src/train.py --name skin_ears_neck_tiles2_full --data data/processed/squares128 \
     --skin-loss --texture-tiles 2 --lambda-outside 10 \
     --device cuda
   ```

   Auf dem Cluster übernimmt das Jobskript die Maskenvorbereitung automatisch,
   wenn `SKIN_LOSS=1` gesetzt ist. Dafür einmalig die zusätzlichen Abhängigkeiten
   in der Cluster-venv installieren (`pip install -r requirements-face-parsing.txt`).
   Im Workspace reichen zunächst die A/B-Bilder:

   ```bash
   mkdir -p logs
   sbatch jobs/train_uc3.sbatch
   ```

   Fehlende Masken werden zuerst auf der GPU direkt im Workspace gespeichert,
   danach werden Bilder und Masken auf die lokale SSD kopiert und das Training
   gestartet. Vorhandene Masken werden übersprungen; sind alle vorhanden, wird
   das Parsing-Modell gar nicht geladen. Die Vorbereitung zählt zur Joblaufzeit.
   Modellgewichte werden standardmäßig unter `$WS/huggingface` gecacht.
   Der erste Download benötigt Internetzugriff; alternativ lässt sich mit
   `FACE_MODEL=/pfad/zum/lokalen/modell` ein vorbereitetes Modell angeben.

`--texture-tiles 2` erzeugt pro B-Bild ein gespiegeltes 2×2-Raster und verkleinert es
auf die ursprüngliche Größe. Das geschieht auch für die festen Kontrollbilder.
Es werden keine Daten überschrieben und keine verschiedenen Quellen/Splits gemischt.
Gesicht und Hautmaske werden bei Augmentierung gemeinsam gespiegelt.

Änderung am bestehenden Training:

- **Textur:** D_B erhält `fake_B * mask` und `real_B * mask`. Beide Seiten besitzen
  dieselbe Maskengeometrie und außerhalb den neutralen Wert 0 (normalisierte Bilder).
  Der direkte Texturgradient zum Generator ist außerhalb der Haut null.
  Im Replay-Pool liegen bereits maskierte Fakes mit ihrer damaligen Geometrie.
- **Erhaltung:** `lambda_outside × masked_L1(fake_B, real_A, 1-mask)` bestraft
  Änderungen an Augen, Mund, Haaren und Hintergrund außerhalb der Hautmaske.
  Pro Bild wird nach ausgewählter Fläche und Kanalzahl normalisiert.
  Mit `--skin-loss` ist dieser Loss standardmäßig mit Gewicht 10 aktiv;
  `--lambda-outside 0` deaktiviert ihn. Ohne `--skin-loss` wird er nicht verwendet.
  TensorBoard protokolliert ihn unter `loss/G_outside`.
- Cycle-, Identity-Loss und D_A bleiben unverändert.
  Der bereits vorhandene optionale Struktur-Loss bleibt unverändert
  und ist standardmäßig deaktiviert (`--lambda-struct 0`). Das bleibt eine experimentelle
  CycleGAN-Variante: Insbesondere die Rückübersetzung reiner Schalentextur zu einem
  ganzen Gesicht ist weiterhin eine schwierige Aufgabe.

Der PatchGAN sieht auch Maskenränder; dieser Ansatz ist keine reine Bewertung
isolierter Hautpatches. Gleiche Maskierung beider Klassen reduziert das Risiko,
dass die Maskenform selbst als Echt/Fake-Merkmal dient. Außerhalb der Haut bleiben
Änderungen möglich: Der Erhaltungs-Loss bestraft Abweichungen, erzwingt aber
keine exakt unveränderten Ausgabepixel.

`skin_masks.png`
im Laufordner enthält die Masken der festen Testgesichter. Die normalen Samples
zeigen weiterhin rohe Generatorbilder, damit unerwünschte Änderungen sichtbar sind.
Für einen sinnvollen Vergleich zuerst `--texture-tiles 1` gegen `2` bei denselben
Loss-Gewichten und demselben Seed testen. Ein Funktionstest ersetzt noch keine
Bewertung eines vollständig trainierten Modells.

## Voller Clusterlauf mit Ohren und Hals

`jobs/train_uc3.sbatch` startet standardmäßig einen neuen Lauf
`skin_ears_neck_tiles2_full` mit folgenden Einstellungen:

| Einstellung | Wert |
|---|---|
| Datensatz | squares128, alle A- und B-Trainingsbilder (kein 2.000-Bilder-Limit) |
| Epochen | 100 konstante Lernrate + 100 mit linearem Decay |
| Hautmaske | Gesichtshaut, Nase, beide Ohren und Hals |
| Textur | gespiegeltes 2×2-Raster |
| Erhaltungs-Loss | Gewicht 10 außerhalb der Hautmaske |
| Struktur-Loss | aus |
| Jobzeitlimit | 48 Stunden inklusive Maskenvorbereitung und Datenkopie |
| Sicherung | latest.pt jede Epoche, zusätzlicher Checkpoint alle 10 Epochen |

Das Jobzeitlimit ist ein Budget, keine gemessene Laufzeitprognose.
Die [Clusterdokumentation](https://wiki.bwhpc.de/e/BwUniCluster3.0/Running_Jobs)
erlaubt bis zu 72 Stunden in `gpu_h100`.

Die erweiterten Masken werden in `train/masks_A_skin_ears_neck` und
`test/masks_A_skin_ears_neck` neu berechnet. Alte `masks_A`-Ordner werden weder
überschrieben noch für diesen Lauf geladen. Die SegFormer-Klassen umfassen keine
allgemeine Körperhaut: Hände und Arme werden damit nicht zuverlässig erfasst.
Auch Ohrringe und Halsketten sind keine ausgewählten Hautklassen.

Auf dem Cluster im Projektordner:

```bash
git pull --ff-only
source .venv/bin/activate
python -m pip install -r requirements-face-parsing.txt
mkdir -p logs
sbatch jobs/train_uc3.sbatch
```

Die Standardwerte gelten, sofern keine früher exportierten Variablen wie
`N_EPOCHS`, `N_EPOCHS_DECAY`, `MAX_IMAGES`, `SKIN_LOSS`, `TEXTURE_TILES` oder
`TRAIN_EXTRA` sie überschreiben. Das Joblog zeigt die verwendeten Werte.
Mit `MAX_IMAGES=20 N_EPOCHS=1 N_EPOCHS_DECAY=0 RUN=skin_ears_neck_smoke`
lässt sich weiterhin ein Kurztest starten.

Ein abgebrochener **voller Lauf mit derselben Konfiguration** wird so fortgesetzt:

```bash
RESUME=1 sbatch jobs/train_uc3.sbatch
```

Der alte 20-Epochen-Lauf wird nicht überschrieben oder als Startpunkt verwendet.
Ein existierender Checkpoint unter dem neuen Laufnamen verhindert einen
versehentlichen Neustart ohne `RESUME=1`.

# CucuGAN: Projektplan

Ziel: Menschliche Gesichter mit einem selbst implementierten **CycleGAN** (PyTorch) in „gurkige“ Gesichter verwandeln. Trainiert wird auf zwei **ungepaarten** Datensätzen: Gesichter (Domäne A) und Gurken (Domäne B).

Referenz: Zhu, Park, Isola, Efros, *Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks*, ICCV 2017 ([arXiv:1703.10593](https://arxiv.org/abs/1703.10593))

---

## 1. Architektur

### 1.1 Grundidee

Es gibt keine Paare (Gesicht, passende Gurke). Deshalb lernt CycleGAN **beide Richtungen** gleichzeitig und verlangt, dass ein Bild nach Hin- und Rückweg wieder sich selbst ergibt (*Cycle-Consistency*).

```
          G_AB                 F_BA
Gesicht a ────→ fake Gurke  ────→ rekonstr. Gesicht   ≈ a   (Cycle-Loss)
                   │
                  D_B: „echte Gurke oder fake?“

          F_BA                 G_AB
Gurke b   ────→ fake Gesicht ────→ rekonstr. Gurke    ≈ b   (Cycle-Loss)
                   │
                  D_A: „echtes Gesicht oder fake?“
```

Es gibt **vier Netze**, aber nur **zwei Klassen**: zweimal Generator und zweimal Diskriminator.

Alle Netze sind **fully convolutional CNNs**: Sie bestehen nur aus Faltungen, ohne Fully-Connected-Schichten.

### 1.2 Generator: Encoder → Residual-Blöcke → Decoder

Die Architektur folgt Johnson et al. 2016. Hier die Variante für 128×128 px:

| Stufe | Schicht | Ausgabe |
|---|---|---|
| Eingang | | 3 × 128 × 128 |
| Encoder | Conv 7×7, 64, InstanceNorm, ReLU | 64 × 128 × 128 |
| | Conv 3×3 stride 2, 128, IN, ReLU | 128 × 64 × 64 |
| | Conv 3×3 stride 2, 256, IN, ReLU | 256 × 32 × 32 |
| Transformer | 6 × Residual-Block (256 Kanäle) | 256 × 32 × 32 |
| Decoder | ConvTranspose 3×3 stride 2, 128, IN, ReLU | 128 × 64 × 64 |
| | ConvTranspose 3×3 stride 2, 64, IN, ReLU | 64 × 128 × 128 |
| Ausgang | Conv 7×7, 3, **tanh** | 3 × 128 × 128 |

- **Residual-Block:** `x + f(x)`, wobei `f` = Conv 3×3 → IN → ReLU → Conv 3×3 → IN ist. Das Netz lernt nur die *Änderung* am Bild und kann den Rest durchreichen. Das passt gut zur Bildübersetzung.
- **Reflection-Padding** statt Zero-Padding vermeidet Artefakte am Bildrand.
- **InstanceNorm** statt BatchNorm: normalisiert pro Bild, eignet sich für Stil und Textur und funktioniert auch bei Batch Size 1.
- **tanh** am Ausgang, weil die Bilder auf [-1, 1] normalisiert sind.
- Bei 256 px nimmt man 9 statt 6 Residual-Blöcke.

### 1.3 Diskriminator: 70×70-PatchGAN

CycleGAN hat diesen Diskriminator aus pix2pix (Isola et al. 2017) übernommen. pix2pix selbst wird in diesem Projekt nicht verwendet, weil es gepaarte Daten braucht.

| Schicht | Ausgabe (bei 128 px Eingang) |
|---|---|
| Conv 4×4 stride 2, 64, LeakyReLU(0.2), *ohne* Norm | 64 × 64 × 64 |
| Conv 4×4 stride 2, 128, IN, LeakyReLU | 128 × 32 × 32 |
| Conv 4×4 stride 2, 256, IN, LeakyReLU | 256 × 16 × 16 |
| Conv 4×4 stride 1, 512, IN, LeakyReLU | 512 × 15 × 15 |
| Conv 4×4 stride 1, 1 | **1 × 14 × 14** |

Der Diskriminator gibt keinen einzelnen Wert aus, sondern eine **Karte**. Jeder Eintrag bewertet einen etwa 70×70 px großen Ausschnitt als echt oder fake. Er beurteilt also **lokale Textur**, nicht die globale Form. Darum überträgt CycleGAN vor allem Farbe und Textur und verändert die Form kaum.

### 1.4 Losses

| Loss | Formel | Gewicht | Zweck |
|---|---|---|---|
| Adversarial (LSGAN) | `MSE(D(fake), 1)` für G; `0.5·[MSE(D(real),1) + MSE(D(fake),0)]` für D | 1 | Fakes sollen realistisch aussehen |
| Cycle | `L1(F(G(a)), a) + L1(G(F(b)), b)` | λ = 10 | Inhalt bleibt erhalten |
| Identity | `L1(G(b), b) + L1(F(a), a)` | 0,5 · λ = 5 | Farben bleiben stabil |

Gesamt-Loss der Generatoren: `L_GAN(G) + L_GAN(F) + λ·L_cyc + 0,5·λ·L_id`

### 1.5 Parametergröße (zum Selbsttest)

| Netz | Parameter |
|---|---|
| Generator (6 Res-Blöcke) | 7.837.699 |
| Diskriminator | 2.764.737 |
| **Gesamt (2 G + 2 D)** | **≈ 21,2 Mio.** |

Fast alle Generator-Parameter stecken in den Residual-Blöcken (je ≈ 1,18 Mio.). Wenn `sum(p.numel() for p in model.parameters())` diese Zahlen liefert, stimmt die Architektur (vorausgesetzt Conv mit Bias, InstanceNorm ohne affine-Parameter).

### 1.6 Training (Standardwerte aus dem Paper)

| Hyperparameter | Wert | Herkunft |
|---|---|---|
| Optimizer | Adam, lr = 2e-4, β = (0,5; 0,999) | DCGAN |
| Batch Size | 1 | CycleGAN |
| Epochen | 100 konstant + 100 lineares Decay auf 0 | CycleGAN |
| Image-Buffer | 50 alte Fakes für D | SimGAN |
| Gewichtsinit | N(0; 0,02) | CycleGAN / DCGAN |

**Ablauf eines Trainingsschritts:**
1. `fake_B = G(a)`, `fake_A = F(b)`, Rekonstruktionen und Identity-Bilder berechnen
2. **Generatoren updaten:** D-Gradienten aus, Gesamt-Loss der Generatoren, `backward`, `step`
3. **Diskriminatoren updaten:** Fakes aus dem Image-Buffer holen und `.detach()`-en, D-Loss berechnen, `backward`, `step`

Die meisten Fehler passieren hier: welcher Loss welches Netz updatet, wo `detach()` hingehört und wo `zero_grad()` steht.

---

## 2. Zu programmierende Bestandteile

| Datei | Inhalt |
|---|---|
| `src/dataset.py` | Unpaired Dataset (zu jedem A ein zufälliges B), Transforms |
| `src/models.py` | `ResidualBlock`, `Generator`, `Discriminator`, `init_weights` |
| `src/utils.py` | `ImagePool`, LR-Schedule, Bilder speichern, Denormalisierung |
| `src/train.py` | Losses, Trainingsloop, Logging, Checkpoints, Config |
| `src/infer.py` | Bild oder Ordner rein, Gurkengesicht raus |
| `src/evaluate.py` | KID/FID auf test |
| `scripts/prepare_data.py` | Zuschneiden, Skalieren, Deduplizieren, Split |

### Grundregel: alles konfigurierbar

Wir starten mit **128×128**. Die Auflösung wird später erhöht, deshalb darf **nichts hart codiert** sein. Über die Config (YAML oder argparse) einstellbar sein müssen:

| Parameter | Start | Später z. B. |
|---|---|---|
| `image_size` | 128 | 256 |
| `n_res_blocks` | 6 | 9 |
| `ngf` / `ndf` (Basiskanäle G / D) | 64 / 64 | 64 / 64 |
| `n_downsampling` (G) | 2 | 2 |
| `n_layers_d` (D) | 3 | 3 |
| `lambda_cycle`, `lambda_identity` | 10 / 0,5 | Experimente |
| `lr`, `n_epochs`, `n_epochs_decay`, `batch_size`, `pool_size` | Paper-Werte | |
| `device` | automatisch (`cuda` falls vorhanden) | |

Konkret heißt das:
- Netze bauen ihre Schichten in Schleifen aus diesen Parametern auf, ohne fest verdrahtete Kanal- oder Pixelzahlen.
- Keine Shape-Annahmen im Code (z. B. D-Ausgabe nicht auf 14×14 festlegen, sondern `torch.ones_like(pred)` als Label verwenden).
- Die Datenaufbereitung bekommt die Zielgröße als Argument und arbeitet idealerweise von den Originalbildern aus, damit später einfach in 256 neu erzeugt werden kann.
- Die Config wird in jedem Checkpoint mitgespeichert, damit `infer.py` das Modell passend wiederaufbauen kann.
- Shape-Tests laufen für mehrere Auflösungen (z. B. 64, 128, 256).

---

## 3. Vorgehen in Phasen

### Phase 0: Setup
- [x] `git init`, `.gitignore` (`data/`, `checkpoints/`, `outputs/`, `.venv/`)
- [x] venv, `requirements.txt` (torch, torchvision, pillow, tqdm, matplotlib, tensorboard)
- [x] Ordnerstruktur anlegen

**Fertig, wenn:** `import torch` läuft und der erste Commit steht.

### Phase 1: Daten
- [x] Gesichter: FFHQ small (Kaggle), 3.143 Bilder, 1024×1024
- [x] Gurken: Jin 2023 (Zenodo), Ordner „Fruit Dataset“, 1.811 Fotos (Quellen siehe README)
- [x] Gurkenbilder von Hand sichten
- [x] Exakte Duplikate entfernen (md5): FFHQ 0, Jin 0
- [x] Split train/test etwa 90/10 pro **Originalfoto** über einen Hash des Dateinamens (stabil, auch beim späteren Neuerzeugen in 256 px)
- [x] Gurken: Gurke finden (nur grüne Farbtöne, Farbkarten als Rechtecke ausgeschlossen), waagerecht drehen, quadratische Ausschnitte entlang der Gurke schneiden (Ausschnitte erben den Split ihres Fotos)
- [x] Auf 128×128 skalieren: `python scripts/prepare_data.py build --size 128`

Reihenfolge: Duplikate entfernen → Split → Zuschneiden. Wie in den Original-Datensätzen von CycleGAN gibt es nur train und test (kein val), siehe Phase 7.

Ergebnis `data/processed/squares128/` (früher `128/`) (Herkunft jedes Bildes in `manifest.csv`):

| | train | test |
|---|---|---|
| A: Gesichter | 2.856 | 287 |
| B: Gurkenquadrate | 8.220 (aus 1.549 Fotos) | 843 (aus 166 Fotos) |

96 Gurkenfotos liefern kein Quadrat (gelbe Gurken, zu dünne Gurken < 128 px).

#### Wechsel auf Collagen (Branch `gurken-collage`)
Mit den Schalenquadraten wird das **ganze Bild** Gurke, Haare und Hintergrund gehen verloren (siehe Abschnitt 4). Ziel ist aber: Gesicht wird Gurke, **Haare bleiben**. Deshalb drei vergleichbare Overfit-Tests (je 20 gleiche Gesichter, 40 Epochen CPU, nur die Gurkenbilder verschieden):

| Lauf | Gurkenbilder | Haare | Hintergrund | Grün wird |
|---|---|---|---|---|
| `overfit` | Schalenquadrate | weg | weg | alles |
| `overfit_umgebung` | 22 Gurkenfotos mit Umgebung (Commons) | bleiben | teils grünstichig | Gesicht + Umgebung |
| `overfit_collage` | 22 Collagen | bleiben | meist original | **gezielt das Gesicht** |

Prinzip (wie horse2zebra): CycleGAN ändert, worin sich die Domänen unterscheiden. Bei den Collagen sind Haare, Hals und Hintergrund in beiden Domänen gleich verteilt, nur Gesicht vs. Gurke unterscheidet sich. Gurkentextur (Noppen) war nach 800 Schritten bei keinem Lauf mit Umgebung sichtbar, nur Farbe; das muss das lange Training zeigen.

- [x] `build --mode collage`: freigestellte, senkrechte Jin-Gurke auf die Gesichtsposition eines FFHQ-Porträts (weiche Kante, Größe/Position leicht zufällig). Nur gerade Gurken ohne dunkle Stellen (391 von 1.811). Porträts werden per Hash zu Gesicht (A) oder Hintergrund (B), etwa 50/50.

Ergebnis `data/processed/collage128/` (Standard für `train.py`):

| | train | test |
|---|---|---|
| A: Gesichter | 1.431 | 150 |
| B: Collagen | 1.425 (351 verschiedene Gurken) | 137 (40 verschiedene Gurken) |

Geprüft: kein Porträt ist zugleich Gesicht und Hintergrund, keine Gurke in train und test.

Nach Baseline 1 (siehe Phase 5): Gurke vergrößert von 50×70 % auf **66×86 %** der Bildgröße (`--face-size`), damit das ganze Gesicht bedeckt ist und keine Hautreste an Wangen/Kinn bleiben. Haare bleiben seitlich/oben sichtbar. Der alte Datensatz liegt als `collage50_128/`. Jeder Build schreibt seine Einstellungen nach `build_config.json`.

**Fertig, wenn:** Ein Bildgrid beider Domänen sauber aussieht und die Bildzahlen pro Split dokumentiert sind.

### Phase 2: Dataset und DataLoader
- [x] `dataset.py` mit Normalisierung auf [-1, 1] und horizontalem Flip (nur train); zu jedem Gesicht eine zufällige Gurke
- [x] `utils.denormalize`; Selbsttest `python src/dataset.py` speichert einen Batch nach `outputs/batch_check.png`

**Fertig, wenn:** Ein Batch geladen, zurücktransformiert und angezeigt werden kann.

### Phase 3: Modelle
- [x] Generator, Diskriminator, Init (`src/models.py`)
- [x] Shape-Tests für 64, 128 und 256 px: G gibt dieselbe Shape zurück, D eine Patch-Karte (64: 6×6, 128: 14×14, 256: 30×30)
- [x] Parameterzahlen mit Abschnitt 1.5 vergleichen: G 7.837.699, D 2.764.737 (exakt); mit 9 Res-Blöcken G 11.378.179

Selbsttest: `python src/models.py`

**Fertig, wenn:** Shapes und Parameterzahlen stimmen.

### Phase 4: Trainingsloop
- [x] Losses, Optimizer, ImagePool, LR-Schedule (`src/train.py`, `src/utils.py`)
- [x] Logging (TensorBoard), Kontrollbilder pro Epoche (optional alle N Schritte), Checkpoints mit Resume
- [x] Config über argparse, gespeichert in `outputs/<name>/config.json` und jedem Checkpoint
- [ ] **Sanity-Check:** auf etwa 20 Bildern overfitten, bis die Rekonstruktionen fast perfekt sind
  - CPU-Lauf (20 Bilder, 40 Epochen = 800 Schritte, ~30 min): Training läuft stabil, Cycle-Loss fällt 9,4 → 3,0, D-Loss ~0,4 (kein Kollaps). Fake-Gurken: grüne Schalentextur mit erkennbarem Gesicht als Relief, auch Hintergrund vergurkt (siehe Abschnitt 4). Rekonstruktionen erkennbar, aber noch unscharf, leichte Schachbrettmuster.
  - GPU-Lauf auf Kaggle (collage128, 20 Bilder, 200 Epochen = 4.000 Schritte, 19 min): läuft stabil, auf ungesehenen Testgesichtern wird das Gesicht zum grünen Oval, Haare/Hintergrund bleiben. Rekonstruktion der Testgesichter erkennbar, aber unscharf (bei 20 Trainingsbildern erwartbar). Für die Baseline als Nachweis ausreichend.

**Fertig, wenn:** Ein Mini-Training auf der CPU durchläuft und sinnvolle Bilder speichert.

### Phase 5: Baseline-Training
- [x] GPU wählen: **Kaggle** (Tesla T4, 15 GB, 30 GPU-h/Woche, Sitzungen max. 12 h), Notebook `notebooks/kaggle_train.ipynb`. bwUniCluster 3.0 (A100/H100) als nächster Schritt: Anmeldung über kiz-Helpdesk (Entitlement), bwIDM, 2FA, Uni-VPN.
- [x] Geschwindigkeit gemessen (128 px, Batch 1, T4): 32 Bit 0,136 s/Schritt, Mixed Precision (`--amp`) 0,118 s/Schritt. Nur ~13 % schneller, weil die GPU bei Batch 1 vor allem auf Verwaltungsarbeit wartet. Ein Schritt kostet ~343 GFLOP (256 px: ~1.375 GFLOP).
- [x] Mixed-Precision-Test (`amp_test`, 1.000 Schritte): stabil, Bilder plausibel.
- [x] **Baseline 1** (`baseline`, Collage 50×70 %, Paper-Standard + `--amp`): 200 Epochen, 0,086 s/Schritt, ~7 h.
  - Ergebnis: Haare und Hintergrund perfekt erhalten, aber der Generator **klebt eine Gurke über das Gesicht**. Keine Gesichtszüge sichtbar, trotzdem fast perfekte Rekonstruktion: Das Gesicht wird unsichtbar in der Gurke versteckt (**Steganografie**, bekanntes CycleGAN-Phänomen, Chu et al. 2017, *CycleGAN, a Master of Steganography*). Ursache: Die Collagen haben keine Gesichtszüge, und der Cycle-Loss prüft nur, ob man zurückrechnen kann, nicht, ob man etwas sieht.
- [ ] **Lauf 2** (`collage66`): Collage 66×86 %, dazu
  - `--lambda-struct 10`: **Struktur-Loss**, grobe Kanten (Graustufen, 1/4 Auflösung, Sobel, pro Bild normiert) von Gesicht und Gurkengesicht müssen übereinstimmen. Umfärben kostet fast nichts (0,035), Gesicht verschwinden lassen kostet viel (0,38 × λ).
  - `--cycle-noise 0.1`: **Rauschen** vor dem Rückweg, zerstört versteckte Feinmuster
  - erst Kurztest `struct_test` (20 Epochen, ~45 min), dann voller Lauf
- [x] Fester Satz Testbilder in jeder Epoche umgewandelt und gespeichert (8 Gesichter + 8 Collagen)

**Fertig, wenn:** Ein Checkpoint vorliegt, der erkennbar Gurkiges erzeugt.

### Phase 6: Evaluation
- [ ] `infer.py`
- [ ] KID (bei wenigen Bildern zuverlässiger) und FID auf test, z. B. mit `torchmetrics` oder `torch-fidelity`

**Fertig, wenn:** Für die Baseline eine Zahl und ein Bildgrid vorliegen.

### Phase 7: Experimente (je nur **eine** Änderung gegenüber der Baseline)
Da es kein val-Set gibt, werden die Varianten auf test verglichen. Die Werte der besten Variante sind dadurch leicht optimistisch; bei Bedarf später einen Teil von train als val abzweigen.
- [ ] Collage vs. Schalenquadrate (`--data data/processed/squares128`) im vollen Training vergleichen
- [ ] Größere Batch Size (z. B. 4 oder 8): nutzt die GPU besser aus (bei Batch 1 nur ~1/6 der T4-Leistung), verändert aber das Lernverhalten
- [ ] Collage-Parameter variieren (Gurkengröße, Anteil Hintergrund-Porträts, auch krumme Gurken zulassen)
- [ ] λ_struct variieren: Regler zwischen „Gurke mit Gesichtszügen“ (klein) und „grün gefärbtes Gesicht“ (groß)
- [ ] Frühere Epochen der Baseline ansehen: Im Kurztest nach 1.000 Schritten waren die Gesichtszüge noch sichtbar, irgendwann kippte es zum Überkleben

- [ ] λ_cycle variieren (z. B. 5 oder 2): mehr Freiheit für Formänderungen
- [ ] λ_identity variieren
- [ ] DiffAugment (wenig Gurkendaten)
- [ ] **Auflösung auf 256×256 erhöhen** (mit 9 Res-Blöcken), Gurkendaten in passender Qualität nötig
- [ ] Optional: Vergleich mit CUT (Park et al. 2020) oder U-GAT-IT (Kim et al. 2020)

### Phase 8: Abschluss
- [ ] Finale Ergebnisse auf dem Test-Set
- [ ] README mit Beispielbildern, Ergebnissen und Anleitung zum Reproduzieren

---

## 4. Erwartungen und Risiken

- **Form vs. Textur:** CycleGAN verändert vor allem Farbe und Oberfläche. Realistisch sind grüne, genoppte Gesichter, keine Gesichter in Gurkenform.
- **Wenige Gurkenbilder:** Der Diskriminator kann overfitten. Gegenmittel sind Augmentierung bzw. DiffAugment.
- **GAN-Losses sind keine Qualitätsmetrik:** Bewertet wird über Bilder und KID/FID.
- **Mode Collapse / Instabilität:** Regelmäßig Bilder anschauen und Checkpoints behalten.
- **Ganzes Bild wird Gurke:** Die Gurkenbilder sind reine Schalentextur (Quadrate ohne Hintergrund), die Gesichtsbilder enthalten Haare, Kleidung und Hintergrund. Der PatchGAN-Diskriminator bewertet ~70×70-Ausschnitte und hält alles für unecht, was nicht nach Schale aussieht. Der Generator wird deshalb vermutlich auch Hintergrund und Haare grün und genoppt machen; die Form bleibt über den Cycle-Loss erhalten. Gegenmittel, vom einfachsten zum aufwendigsten:
  1. Erst die Baseline trainieren und ansehen.
  2. Maske beim Anwenden: In `infer.py` nur die Hautbereiche aus dem Generator übernehmen, den Rest aus dem Original (Hautmaske über ein fertiges Face-Parsing-Modell).
  3. Maske beim Training: Diskriminator-Loss nur auf Hautbereichen.
  4. Modell mit eigener Aufmerksamkeit, z. B. U-GAT-IT (siehe Phase 7).

  **Entscheidung:** keine Maske (nimmt den Gag), stattdessen die Gurkendomäne angepasst: Collagen (siehe Phase 1). Neues Risiko: Die Collagen haben keine Augen und keinen Mund. Der Diskriminator drängt darauf, sie verschwinden zu lassen, der Cycle-Loss will sie behalten. Außerdem kann der Collage-Look (Kanten, Licht) mitgelernt werden.
- **Schachbrettmuster:** `ConvTranspose2d` im Decoder kann Checkerboard-Artefakte erzeugen. Falls sichtbar: durch Hochskalieren + normale Faltung ersetzen.

## 5. Weiterführende Literatur

- Isola et al. 2017, *pix2pix*: Vorgänger mit gepaarten Daten, Herkunft des PatchGAN
- Kim et al. 2017 (*DiscoGAN*), Yi et al. 2017 (*DualGAN*): parallel entstandene Arbeiten mit derselben Idee
- Johnson et al. 2016: Generator-Architektur
- Mao et al. 2017: LSGAN
- Park et al. 2020: CUT
- Kim et al. 2020: U-GAT-IT
- Zhao et al. 2020: DiffAugment
- Pang et al. 2021: Survey zu Image-to-Image Translation

*(Jahreszahlen und Venues vor dem Zitieren auf arXiv oder Google Scholar prüfen.)*

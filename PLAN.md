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
- [ ] Gesichter: FFHQ small (Kaggle), etwa 3.100 Bilder
- [ ] Gurken: Jin 2023 (Zenodo), Ordner „Fruit Dataset“, etwa 1.800 Fotos (Quellen siehe README)
- [x] Gurkenbilder von Hand sichten
- [x] Exakte Duplikate entfernen (md5): FFHQ 0, Jin 0
- [ ] Split train/test etwa 90/10 pro **Originalfoto** über einen Hash des Dateinamens (stabil, auch beim späteren Neuerzeugen in 256 px)
- [ ] Gurken: Gurke finden, waagerecht drehen, quadratische Ausschnitte entlang der Gurke schneiden (Ausschnitte erben den Split ihres Fotos)
- [ ] Auf 128×128 skalieren

Reihenfolge: Duplikate entfernen → Split → Zuschneiden. Wie in den Original-Datensätzen von CycleGAN gibt es nur train und test (kein val), siehe Phase 7.

**Fertig, wenn:** Ein Bildgrid beider Domänen sauber aussieht und die Bildzahlen pro Split dokumentiert sind.

### Phase 2: Dataset und DataLoader
- [ ] `dataset.py` mit Normalisierung auf [-1, 1] und horizontalem Flip

**Fertig, wenn:** Ein Batch geladen, zurücktransformiert und angezeigt werden kann.

### Phase 3: Modelle
- [ ] Generator, Diskriminator, Init
- [ ] Shape-Tests für 64, 128 und 256 px: G gibt dieselbe Shape zurück, D eine Patch-Karte (bei 128 px: 1×1×14×14)
- [ ] Parameterzahlen mit Abschnitt 1.5 vergleichen

**Fertig, wenn:** Shapes und Parameterzahlen stimmen.

### Phase 4: Trainingsloop
- [ ] Losses, Optimizer, ImagePool, LR-Schedule
- [ ] Logging (TensorBoard), alle N Schritte Beispielbilder, Checkpoints mit Resume
- [ ] Config über argparse oder YAML
- [ ] **Sanity-Check:** auf etwa 20 Bildern overfitten, bis die Rekonstruktionen fast perfekt sind

**Fertig, wenn:** Ein Mini-Training auf der CPU durchläuft und sinnvolle Bilder speichert.

### Phase 5: Baseline-Training
- [ ] GPU wählen (eigener PC, Uni-Cluster, Kaggle oder Colab)
- [ ] Volles Training mit den Standardwerten
- [ ] Festen Satz von 16 Test-Gesichtern in jeder Epoche umwandeln und speichern

**Fertig, wenn:** Ein Checkpoint vorliegt, der erkennbar Gurkiges erzeugt.

### Phase 6: Evaluation
- [ ] `infer.py`
- [ ] KID (bei wenigen Bildern zuverlässiger) und FID auf test, z. B. mit `torchmetrics` oder `torch-fidelity`

**Fertig, wenn:** Für die Baseline eine Zahl und ein Bildgrid vorliegen.

### Phase 7: Experimente (je nur **eine** Änderung gegenüber der Baseline)
Da es kein val-Set gibt, werden die Varianten auf test verglichen. Die Werte der besten Variante sind dadurch leicht optimistisch; bei Bedarf später einen Teil von train als val abzweigen.

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
